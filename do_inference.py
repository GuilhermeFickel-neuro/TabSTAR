import argparse
import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, r2_score

from tabular.datasets.tabular_datasets import TabularDatasetID, CustomDatasetID, get_sid
from tabular.datasets.torch_dataset import get_data_dir, get_properties, HDF5Dataset
from tabular.evaluation.loss import apply_loss_fn
from tabular.evaluation.metrics import calculate_metric, calculate_ks_metric, PredictionsCache
from tabular.preprocessing.objects import SupervisedTask
from tabular.preprocessing.splits import DataSplit
from tabular.tabstar.arch.arch import TabStarModel
from tabular.tabstar.params.config import TabStarConfig
from tabular.tabstar.tabstar_trainer import TabStarTrainer
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.dataloaders import get_dataloader
from tabular.utils.gpus import get_device
from tabular.utils.paths import get_model_path, sanitize_filename_component
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


def process_test_csv(csv_path: str, target_column: str, pretrain_args: PretrainArgs, 
                    run_num: int, device: torch.device, custom_max_features: int = 3000) -> str:
    """Process the test CSV using the same preprocessing pipeline as training"""
    
    # Create a custom dataset ID for the test data
    test_dataset_id = CustomDatasetID.CUSTOM_CSV
    
    # Get the data directory (this will create and cache the processed dataset)
    data_dir = get_data_dir(
        dataset=test_dataset_id,
        processing=TabStarTrainer.PROCESSING,
        run_num=run_num,
        train_examples=-1,  # Use all available data
        device=device,
        number_verbalization=pretrain_args.numbers_verbalization,
        custom_csv_path=csv_path,
        custom_target_column=target_column,
        custom_max_features=custom_max_features
    )
    
    return data_dir


def run_inference(model: TabStarModel, data_dir: str, device: torch.device) -> Dict:
    """Run inference on the test data and calculate metrics"""
    
    # Get dataset properties
    properties = get_properties(data_dir)
    cprint(f"📊 Dataset info: {properties.sid}, Task: {properties.task_type.value}")
    
    # Load test data (we'll use all available splits for inference)
    test_splits = []
    for split in [DataSplit.TRAIN, DataSplit.DEV, DataSplit.TEST]:
        try:
            data_loader = get_dataloader(data_dir=data_dir, split=split, batch_size=32)
            if len(data_loader.dataset) > 0:
                test_splits.append((split, data_loader))
                cprint(f"   {split.value}: {len(data_loader.dataset)} samples")
        except:
            # Skip if split doesn't exist
            pass
    
    if not test_splits:
        raise ValueError("No data found for inference")
    
    # Run inference on all available data
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for split_name, data_loader in test_splits:
            cprint(f"🔮 Running inference on {split_name.value} split...")
            
            cache = PredictionsCache()
            for x_txt, x_num, y, batch_properties in data_loader:
                # Run model inference
                y_pred = model(x_txt=x_txt, x_num=x_num, sid=batch_properties.sid, 
                              d_output=batch_properties.d_effective_output)
                
                # Apply the same post-processing as during training
                predictions = apply_loss_fn(y_pred, batch_properties.task_type)
                cache.append(y=y, predictions=predictions)
            
            all_predictions.append(cache.y_pred)
            all_labels.append(cache.y_true)
    
    # Concatenate all results
    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_labels)
    
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
    
    # Set device
    device = torch.device(get_device())
    cprint(f"Using device: {device}")
    
    # Fix seed for reproducibility
    fix_seed()
    
    cprint(f"🚀 Starting inference with finetuned TabSTAR model...")
    cprint(f"   Pretrain experiment: {args.pretrain_exp}")
    cprint(f"   Finetune experiment: {args.exp_name}")
    cprint(f"   Test CSV: {args.test_csv_path}")
    cprint(f"   Target column: {args.target_column}")
    
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
            custom_max_features=args.custom_max_features
        )
        
        # Run inference and calculate metrics
        cprint("🔮 Running inference...")
        results = run_inference(model, data_dir, device)
        
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
        cprint("=" * 50)
        
    except Exception as e:
        cprint(f"❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main() 