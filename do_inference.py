import argparse
import os
from typing import Optional, Dict
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, r2_score
from torch.utils.data import DataLoader

from tabular.datasets.tabular_datasets import TabularDatasetID, CustomDatasetID, get_sid
from tabular.datasets.torch_dataset import get_data_dir, get_properties, HDF5Dataset, create_dataset, get_raw_dataset, fill_idx2text, save_data_splits, save_properties
from tabular.datasets.data_processing import TabularDataset
from tabular.datasets.properties import DatasetProperties
from tabular.evaluation.loss import apply_loss_fn
from tabular.evaluation.metrics import calculate_metric, calculate_ks_metric, PredictionsCache
from tabular.preprocessing.objects import SupervisedTask
from tabular.preprocessing.splits import DataSplit
from tabular.tabstar.arch.arch import TabStarModel
from tabular.tabstar.params.config import TabStarConfig
from tabular.tabstar.tabstar_trainer import TabStarTrainer
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.dataloaders import get_dataloader, tabular_collate_fn
from tabular.utils.gpus import get_device
from tabular.utils.paths import get_model_path, sanitize_filename_component, create_dir, dataset_run_properties_dir, properties_path
from tabular.utils.utils import cprint, verbose_print, fix_seed
from tabular.utils.logging import LOG_SEP
from peft import PeftModel


def load_finetuned_model(pretrain_exp: str, exp_name: str, lora_lr: float, 
                        lora_batch: int, lora_r: int, patience: int, device: torch.device, 
                        run_num: int, train_examples: int) -> TabStarModel:
    """Load the finetuned TabSTAR model (base model + LoRA adapter)"""
    
    # Load pretrain args to get the base model path
    pretrain_args = PretrainArgs.from_json(pretrain_exp=pretrain_exp)
    base_model_dir = get_model_path(pretrain_exp, is_pretrain=True)
    
    if not os.path.exists(base_model_dir):
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    
    # Construct finetune args to get the LoRA adapter path
    from tabular.tabstar.params.constants import LORA_BATCH, LORA_R
    
    # Create a mock args object for FinetuneArgs construction
    class MockArgs:
        def __init__(self):
            self.lora_lr = lora_lr
            self.lora_batch = lora_batch if lora_batch is not None else LORA_BATCH
            self.lora_r = lora_r if lora_r is not None else LORA_R
            self.downstream_keep_model = True
            self.downstream_patience = patience
    
    mock_args = MockArgs()
    finetune_args = FinetuneArgs.from_args(args=mock_args, exp_name=exp_name, pretrain_args=pretrain_args)
    
    # Now construct the additional ModelTrainer components (without exp_name to avoid duplication)
    dataset_id = CustomDatasetID.CUSTOM_CSV
    model_short_name = "Tab*"  # TabStarTrainer.SHORT_NAME
    
    run_name_components = [
        sanitize_filename_component(model_short_name),
        f"sid_{sanitize_filename_component(get_sid(dataset_id))}",
        f"run_{run_num}",
        f"examples_{train_examples}"
    ]
    run_name_suffix = LOG_SEP.join(run_name_components)
    
    # Combine finetune path with run_name suffix
    full_run_name = f"{finetune_args.full_exp_name}__{run_name_suffix}"
    adapter_dir = get_model_path(full_run_name, is_pretrain=False)
    
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")
    
    cprint(f"🔄 Loading base model from: {base_model_dir}")
    base_model = TabStarModel.from_pretrained(base_model_dir)
    
    cprint(f"🔄 Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    
    cprint(f"✅ Successfully loaded finetuned model")
    return model


def create_inference_dataset(data_dir: str, csv_path: str, target_column: str, pretrain_args: PretrainArgs, 
                           run_num: int, device: torch.device, custom_max_features: int = 3000,
                           custom_test_csv_path: str = None):
    """Create a dataset for inference where ALL data goes into the TRAIN split"""
    
    # Load raw dataset
    dataset_id = CustomDatasetID.CUSTOM_CSV
    raw_dataset = get_raw_dataset(dataset_id, custom_csv_path=csv_path, 
                                 custom_target_column=target_column, custom_max_features=custom_max_features,
                                 custom_test_csv_path=custom_test_csv_path)
    
    # Create artificial splits where ALL data goes to TRAIN
    n = len(raw_dataset.y)
    splits = [DataSplit.TRAIN] * n  # All data goes to train split
    
    # Process dataset for TabSTAR
    from tabular.preprocessing.target import process_y
    from tabular.tabstar.preprocessing.numerical import scale_x_num_and_add_categorical_bins
    from tabular.tabstar.preprocessing.textual import verbalize_x_txt
    from tabular.tabstar.preprocessing.target import add_target_tokens
    
    # Process targets
    targets = process_y(raw=raw_dataset, splits=splits, processing=TabStarTrainer.PROCESSING)
    feat_cnt = {feat_type.value: len(names) for feat_type, names in raw_dataset.feature_types.items()}
    
    # Process for TabSTAR
    x_txt, x_num = scale_x_num_and_add_categorical_bins(raw=raw_dataset, splits=splits,
                                                       number_verbalization=pretrain_args.numbers_verbalization)
    verbalize_x_txt(x_txt)
    
    # Create properties with all data in train split
    properties = DatasetProperties.create(raw=raw_dataset, splits=splits, feat_cnt=feat_cnt, 
                                         targets=targets, processing=TabStarTrainer.PROCESSING)
    
    # Add target tokens
    x_txt, x_num = add_target_tokens(x_txt=x_txt, x_num=x_num, data=properties)
    
    # Create dataset
    dataset = TabularDataset(properties=properties, x=x_txt, y=raw_dataset.y, splits=splits, x_num=x_num)
    
    # Fill text mappings
    fill_idx2text(dataset)
    
    # Save dataset
    cprint(f"Saving inference dataset {dataset.properties.sid} to {data_dir}")
    save_data_splits(dataset=dataset, data_dir=data_dir, processing=TabStarTrainer.PROCESSING)
    save_properties(data_dir=data_dir, dataset=dataset)
    cprint(f"✅ Saved inference dataset!")


def process_test_csv_for_inference(csv_path: str, target_column: str, pretrain_args: PretrainArgs, 
                                  run_num: int, device: torch.device, custom_max_features: int = 3000,
                                  custom_test_csv_path: str = None) -> str:
    """Process the test CSV specifically for inference, putting all data into train split"""
    
    # Create a custom dataset ID for the test data
    test_dataset_id = CustomDatasetID.CUSTOM_CSV
    
    # For inference, we'll use a different run_num to avoid conflicts with training data
    inference_run_num = run_num + 1000  # Offset to avoid conflicts
    
    # Create data directory manually
    sid = get_sid(test_dataset_id)
    data_dir = os.path.join(dataset_run_properties_dir(run_num=inference_run_num, train_examples=-1), 
                           TabStarTrainer.PROCESSING, sid)
    if pretrain_args.numbers_verbalization.value != "full":
        data_dir = os.path.join(data_dir, pretrain_args.numbers_verbalization.value)
    
    # Check if already exists
    if not os.path.exists(properties_path(data_dir)):
        create_dir(data_dir)
        try:
            create_inference_dataset(data_dir=data_dir, csv_path=csv_path, target_column=target_column,
                                   pretrain_args=pretrain_args, run_num=inference_run_num, device=device,
                                   custom_max_features=custom_max_features, custom_test_csv_path=custom_test_csv_path)
        except Exception as e:
            raise Exception(f"🚨🚨🚨 Error creating inference dataset due to: {e}")
    
    return data_dir


def process_test_csv(csv_path: str, target_column: str, pretrain_args: PretrainArgs, 
                    run_num: int, device: torch.device, custom_max_features: int = 3000,
                    custom_test_csv_path: str = None) -> str:
    """Process the test CSV using the same preprocessing pipeline as training"""
    
    # Use the new inference-specific function
    return process_test_csv_for_inference(csv_path, target_column, pretrain_args, run_num, device, 
                                         custom_max_features, custom_test_csv_path)


def run_inference(model: TabStarModel, data_dir: str, device: torch.device) -> Dict:
    """Run inference on the test data and calculate metrics"""
    
    # Get dataset properties
    properties = get_properties(data_dir)
    cprint(f"📊 Dataset info: {properties.sid}, Task: {properties.task_type.value}")
    cprint(f"📊 Available splits: {properties.split_sizes}")
    
    # For inference, all data should be in the TRAIN split
    if 'train' not in properties.split_sizes or properties.split_sizes['train'] == 0:
        raise ValueError("No training data found for inference. Expected all data to be in 'train' split.")
    
    total_samples = properties.split_sizes['train']
    cprint(f"📊 Total samples for inference: {total_samples}")
    
    # Load the train split which contains all our inference data
    # We need to create a custom dataloader for the train split
    dataset = HDF5Dataset(split_dir=join(data_dir, DataSplit.TRAIN))
    data_loader = DataLoader(dataset, shuffle=False, collate_fn=tabular_collate_fn, batch_size=32, num_workers=0)
    
    cprint(f"🔮 Running inference on {len(data_loader.dataset)} samples...")
    
    # Run inference on all data
    model.eval()
    
    with torch.no_grad():
        cache = PredictionsCache()
        for x_txt, x_num, y, batch_properties in data_loader:
            # Run model inference
            y_pred = model(x_txt=x_txt, x_num=x_num, sid=batch_properties.sid, 
                          d_output=batch_properties.d_effective_output)
            
            # Apply the same post-processing as during training
            predictions = apply_loss_fn(y_pred, batch_properties.task_type)
            cache.append(y=y, predictions=predictions)
    
    y_pred = cache.y_pred
    y_true = cache.y_true
    
    cprint(f"📊 Final dataset: {len(y_pred)} total predictions")
    
    # Calculate metrics
    results = {}
    
    if properties.task_type == SupervisedTask.BINARY:
        # For binary classification, calculate ROC AUC and KS
        roc_auc = roc_auc_score(y_true, y_pred)
        ks_statistic = calculate_ks_metric(y_true, y_pred)
        
        results['roc_auc'] = roc_auc
        results['ks_statistic'] = ks_statistic
        results['task_type'] = 'binary'
        
        cprint(f"📊 Binary Classification Results:")
        cprint(f"   ROC AUC: {roc_auc:.4f}")
        cprint(f"   KS Statistic: {ks_statistic:.4f}")
        
    elif properties.task_type == SupervisedTask.MULTICLASS:
        # For multiclass, calculate macro-averaged ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
            results['roc_auc'] = roc_auc
            results['task_type'] = 'multiclass'
            cprint(f"📊 Multiclass Classification Results:")
            cprint(f"   ROC AUC (macro): {roc_auc:.4f}")
        except Exception as e:
            cprint(f"⚠️ Could not calculate ROC AUC for multiclass: {e}")
            results['roc_auc'] = None
            results['task_type'] = 'multiclass'
    
    elif properties.task_type == SupervisedTask.REGRESSION:
        # For regression, calculate R²
        r2 = r2_score(y_true, y_pred)
        results['r2_score'] = r2
        results['task_type'] = 'regression'
        
        cprint(f"📊 Regression Results:")
        cprint(f"   R² Score: {r2:.4f}")
    
    # Add basic statistics
    results['n_samples'] = len(y_true)
    results['predictions_mean'] = float(np.mean(y_pred))
    results['predictions_std'] = float(np.std(y_pred))
    results['labels_mean'] = float(np.mean(y_true))
    results['labels_std'] = float(np.std(y_true))
    
    return results


def save_results_to_csv(dataset_name: str, finetuned_model: str, results: Dict, csv_path: str = "results.csv"):
    """Save inference results to CSV file, appending to existing data if present"""
    
    # Extract relevant metrics
    roc_auc = results.get('roc_auc', None)
    ks_statistic = results.get('ks_statistic', None)
    task_type = results.get('task_type', 'unknown')
    n_samples = results.get('n_samples', 0)
    
    # Create new result row
    new_result = {
        'dataset_name': dataset_name,
        'finetuned_model': finetuned_model,
        'task_type': task_type,
        'n_samples': n_samples,
        'roc_auc': roc_auc,
        'ks_statistic': ks_statistic,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Read existing results if file exists
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            cprint(f"📄 Found existing results file with {len(existing_df)} entries")
        except Exception as e:
            cprint(f"⚠️ Could not read existing results file: {e}")
            existing_df = pd.DataFrame()
    else:
        cprint(f"📄 Creating new results file: {csv_path}")
        existing_df = pd.DataFrame()
    
    # Append new result
    new_df = pd.DataFrame([new_result])
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(csv_path, index=False)
    cprint(f"💾 Saved results to {csv_path}")
    cprint(f"📊 Total entries in results file: {len(combined_df)}")
    
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run inference with a finetuned TabSTAR model")
    parser.add_argument('--pretrain_exp', type=str, required=True,
                       help='Name of the pretrained model experiment')
    parser.add_argument('--exp_name', type=str, default='custom_dataset_finetune',
                       help='Name of the finetuning experiment')
    parser.add_argument('--test_csv_path', type=str, required=True,
                       help='Path to the test CSV file')
    parser.add_argument('--target_column', type=str, required=True,
                       help='Name of the target column in the test CSV')
    parser.add_argument('--run_num', type=int, default=0,
                       help='Run number used during finetuning')
    parser.add_argument('--custom_max_features', type=int, default=3000,
                       help='Maximum number of features (should match finetuning)')
    parser.add_argument('--train_examples', type=int, default=10000,
                       help='Number of training examples used during finetuning')
    parser.add_argument('--results_csv', type=str, default='results.csv',
                       help='Path to save results CSV file')
    parser.add_argument('--custom_test_csv_path', type=str,
                       help='Path to additional test CSV file (for two CSV mode, optional)')
    
    # LoRA parameters (should match the finetuning run)
    parser.add_argument('--lora_lr', type=float, default=0.001,
                       help='LoRA learning rate used during finetuning')
    parser.add_argument('--lora_batch', type=int, default=None,
                       help='LoRA batch size used during finetuning')
    parser.add_argument('--lora_r', type=int, default=None,
                       help='LoRA rank used during finetuning')
    parser.add_argument('--patience', type=int, default=50,
                       help='Patience used during finetuning')

    args = parser.parse_args()
    
    # Set default values for LoRA parameters if not provided
    if args.lora_batch is None:
        from tabular.tabstar.params.constants import LORA_BATCH
        args.lora_batch = LORA_BATCH
    if args.lora_r is None:
        from tabular.tabstar.params.constants import LORA_R
        args.lora_r = LORA_R
    
    # Validate inputs
    if not os.path.exists(args.test_csv_path):
        raise FileNotFoundError(f"Test CSV file not found: {args.test_csv_path}")
    
    if args.custom_test_csv_path and not os.path.exists(args.custom_test_csv_path):
        raise FileNotFoundError(f"Additional test CSV file not found: {args.custom_test_csv_path}")
    
    # Set device
    device = torch.device(get_device())
    cprint(f"Using device: {device}")
    
    # Fix seed for reproducibility
    fix_seed()
    
    # Create model identifier for results
    dataset_name = os.path.basename(args.test_csv_path)
    if args.custom_test_csv_path:
        dataset_name += f"+{os.path.basename(args.custom_test_csv_path)}"
    finetuned_model = f"{args.pretrain_exp}__{args.exp_name}__lr_{args.lora_lr}__r_{args.lora_r}__examples_{args.train_examples}"
    
    cprint(f"🚀 Starting inference with finetuned TabSTAR model...")
    cprint(f"   Pretrain experiment: {args.pretrain_exp}")
    cprint(f"   Finetune experiment: {args.exp_name}")
    cprint(f"   Test CSV: {args.test_csv_path}")
    if args.custom_test_csv_path:
        cprint(f"   Additional test CSV: {args.custom_test_csv_path}")
    cprint(f"   Target column: {args.target_column}")
    cprint(f"   Model identifier: {finetuned_model}")
    
    try:
        # Load the finetuned model
        model = load_finetuned_model(
            pretrain_exp=args.pretrain_exp,
            exp_name=args.exp_name,
            lora_lr=args.lora_lr,
            lora_batch=args.lora_batch,
            lora_r=args.lora_r,
            patience=args.patience,
            device=device,
            run_num=args.run_num,
            train_examples=args.train_examples
        )
        
        # Load pretrain args for preprocessing
        pretrain_args = PretrainArgs.from_json(pretrain_exp=args.pretrain_exp)
        
        # Process the test CSV
        cprint("📝 Processing test CSV...")
        data_dir = process_test_csv(
            csv_path=args.test_csv_path,
            target_column=args.target_column,
            pretrain_args=pretrain_args,
            run_num=args.run_num,
            device=device,
            custom_max_features=args.custom_max_features,
            custom_test_csv_path=args.custom_test_csv_path
        )
        
        # Run inference and calculate metrics
        cprint("🔮 Running inference...")
        results = run_inference(model, data_dir, device)
        
        # Save results to CSV
        cprint("💾 Saving results...")
        save_results_to_csv(dataset_name=dataset_name, finetuned_model=finetuned_model, 
                           results=results, csv_path=args.results_csv)
        
        # Print summary
        cprint("🎉 Inference completed successfully!")
        cprint("=" * 50)
        
        if results['task_type'] == 'binary':
            cprint(f"📈 FINAL RESULTS:")
            cprint(f"   ROC AUC: {results['roc_auc']:.4f}")
            cprint(f"   KS Statistic: {results['ks_statistic']:.4f}")
        elif results['task_type'] == 'multiclass':
            if results['roc_auc'] is not None:
                cprint(f"📈 FINAL RESULTS:")
                cprint(f"   ROC AUC (macro): {results['roc_auc']:.4f}")
        elif results['task_type'] == 'regression':
            cprint(f"📈 FINAL RESULTS:")
            cprint(f"   R² Score: {results['r2_score']:.4f}")
        
        cprint(f"   Samples: {results['n_samples']}")
        cprint(f"   Results saved to: {args.results_csv}")
        cprint("=" * 50)
        
    except Exception as e:
        cprint(f"❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main() 