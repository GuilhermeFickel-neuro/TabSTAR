import argparse
import os
from typing import Optional, Dict, List
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, r2_score
from torch.utils.data import DataLoader

from tabular.datasets.tabular_datasets import TabularDatasetID, CustomDatasetID, get_sid
from tabular.datasets.torch_dataset import get_data_dir, get_properties, HDF5Dataset
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


def get_finetuned_data_dir(pretrain_exp: str, exp_name: str, run_num: int, train_examples: int) -> str:
    """Get the data directory where the finetuned model's processed data is stored"""
    
    # Load pretrain args to get the number verbalization
    pretrain_args = PretrainArgs.from_json(pretrain_exp=pretrain_exp)
    
    # Construct the data directory path (same as used during finetuning)
    dataset_id = CustomDatasetID.CUSTOM_CSV
    sid = get_sid(dataset_id)
    data_dir = join(dataset_run_properties_dir(run_num=run_num, train_examples=train_examples), 
                   TabStarTrainer.PROCESSING, sid)
    
    if pretrain_args.numbers_verbalization.value != "full":
        data_dir = join(data_dir, pretrain_args.numbers_verbalization.value)
    
    if not os.path.exists(properties_path(data_dir)):
        raise FileNotFoundError(f"Finetuned data directory not found: {data_dir}. "
                              f"Make sure you ran finetuning first with the same parameters.")
    
    return data_dir


def run_inference_on_test_split(model: TabStarModel, data_dir: str, test_split_name: str, device: torch.device) -> Dict:
    """Run inference on a specific test split and calculate metrics"""
    
    # Get dataset properties
    properties = get_properties(data_dir)
    cprint(f"📊 Dataset info: {properties.sid}, Task: {properties.task_type.value}")
    cprint(f"📊 Available splits: {properties.split_sizes}")
    
    # Check if the specific test split exists
    if test_split_name not in properties.split_sizes or properties.split_sizes[test_split_name] == 0:
        raise ValueError(f"Test split '{test_split_name}' not found or empty. Available splits: {list(properties.split_sizes.keys())}")
    
    total_samples = properties.split_sizes[test_split_name]
    cprint(f"📊 Total samples for test split '{test_split_name}': {total_samples}")
    
    # Load the specific test split
    dataset = HDF5Dataset(split_dir=join(data_dir, test_split_name))
    data_loader = DataLoader(dataset, shuffle=False, collate_fn=tabular_collate_fn, batch_size=32, num_workers=0)
    
    cprint(f"🔮 Running inference on test split '{test_split_name}' with {len(data_loader.dataset)} samples...")
    
    # Run inference
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
    
    cprint(f"📊 Final predictions for '{test_split_name}': {len(y_pred)} total predictions")
    
    # Calculate metrics
    results = {}
    
    if properties.task_type == SupervisedTask.BINARY:
        # For binary classification, calculate ROC AUC and KS
        roc_auc = roc_auc_score(y_true, y_pred)
        ks_statistic = calculate_ks_metric(y_true, y_pred)
        
        results['roc_auc'] = roc_auc
        results['ks_statistic'] = ks_statistic
        results['task_type'] = 'binary'
        
        cprint(f"📊 Binary Classification Results for '{test_split_name}':")
        cprint(f"   ROC AUC: {roc_auc:.4f}")
        cprint(f"   KS Statistic: {ks_statistic:.4f}")
        
    elif properties.task_type == SupervisedTask.MULTICLASS:
        # For multiclass, calculate macro-averaged ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
            results['roc_auc'] = roc_auc
            results['task_type'] = 'multiclass'
            cprint(f"📊 Multiclass Classification Results for '{test_split_name}':")
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
        
        cprint(f"📊 Regression Results for '{test_split_name}':")
        cprint(f"   R² Score: {r2:.4f}")
    
    # Add basic statistics
    results['n_samples'] = len(y_true)
    results['predictions_mean'] = float(np.mean(y_pred))
    results['predictions_std'] = float(np.std(y_pred))
    results['labels_mean'] = float(np.mean(y_true))
    results['labels_std'] = float(np.std(y_true))
    results['test_split_name'] = test_split_name
    
    return results


def save_results_to_csv(dataset_name: str, finetuned_model: str, results: Dict, csv_path: str = "results.csv"):
    """Save inference results to CSV file, appending to existing data if present"""
    
    # Extract relevant metrics
    roc_auc = results.get('roc_auc', None)
    ks_statistic = results.get('ks_statistic', None)
    r2_score = results.get('r2_score', None)
    task_type = results.get('task_type', 'unknown')
    n_samples = results.get('n_samples', 0)
    test_split_name = results.get('test_split_name', 'unknown')
    
    # Create new result row
    new_result = {
        'dataset_name': dataset_name,
        'test_split_name': test_split_name,
        'finetuned_model': finetuned_model,
        'task_type': task_type,
        'n_samples': n_samples,
        'roc_auc': roc_auc,
        'ks_statistic': ks_statistic,
        'r2_score': r2_score,
        'predictions_mean': results.get('predictions_mean'),
        'predictions_std': results.get('predictions_std'),
        'labels_mean': results.get('labels_mean'),
        'labels_std': results.get('labels_std'),
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
    parser.add_argument('--run_num', type=int, default=0,
                       help='Run number used during finetuning')
    parser.add_argument('--train_examples', type=int, default=10000,
                       help='Number of training examples used during finetuning')
    parser.add_argument('--results_csv', type=str, default='results.csv',
                       help='Path to save results CSV file')
    
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
    
    # Set device
    device = torch.device(get_device())
    cprint(f"Using device: {device}")
    
    # Fix seed for reproducibility
    fix_seed()
    
    # Create model identifier for results
    finetuned_model = f"{args.pretrain_exp}__{args.exp_name}__lr_{args.lora_lr}__r_{args.lora_r}__examples_{args.train_examples}"
    
    cprint(f"🚀 Starting inference with finetuned TabSTAR model...")
    cprint(f"   Pretrain experiment: {args.pretrain_exp}")
    cprint(f"   Finetune experiment: {args.exp_name}")
    cprint(f"   Run number: {args.run_num}")
    cprint(f"   Training examples: {args.train_examples}")
    cprint(f"   Model identifier: {finetuned_model}")
    
    try:
        # Load the finetuned model
        cprint("🔄 Loading finetuned model...")
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
        
        # Get the data directory where preprocessed data is stored
        cprint("📂 Finding finetuned data directory...")
        data_dir = get_finetuned_data_dir(
            pretrain_exp=args.pretrain_exp,
            exp_name=args.exp_name,
            run_num=args.run_num,
            train_examples=args.train_examples
        )
        cprint(f"📂 Using data directory: {data_dir}")
        
        # Get dataset properties to see what test splits are available
        properties = get_properties(data_dir)
        available_splits = list(properties.split_sizes.keys())
        test_splits = [split for split in available_splits if split.startswith('test')]
        
        if not test_splits:
            raise ValueError(f"No test splits found in the finetuned data. Available splits: {available_splits}")
        
        cprint(f"🔍 Found {len(test_splits)} test split(s): {test_splits}")
        
        # Run inference on each test split independently
        all_results = []
        for i, test_split_name in enumerate(test_splits):
            cprint(f"\n{'='*60}")
            cprint(f"🔮 RUNNING INFERENCE ON TEST SPLIT {i+1}/{len(test_splits)}: {test_split_name}")
            cprint(f"{'='*60}")
            
            try:
                # Run inference on this specific test split
                results = run_inference_on_test_split(model, data_dir, test_split_name, device)
                
                # Add dataset identifier to results
                results['test_split_name'] = test_split_name
                all_results.append(results)
                
                # Save results for this specific test split
                cprint("💾 Saving results...")
                save_results_to_csv(dataset_name=test_split_name, finetuned_model=finetuned_model, 
                                   results=results, csv_path=args.results_csv)
                
                # Print summary for this test split
                cprint(f"✅ Test split '{test_split_name}' completed successfully!")
                if results['task_type'] == 'binary':
                    cprint(f"   ROC AUC: {results['roc_auc']:.4f}")
                    cprint(f"   KS Statistic: {results['ks_statistic']:.4f}")
                elif results['task_type'] == 'multiclass':
                    if results['roc_auc'] is not None:
                        cprint(f"   ROC AUC (macro): {results['roc_auc']:.4f}")
                elif results['task_type'] == 'regression':
                    cprint(f"   R² Score: {results['r2_score']:.4f}")
                cprint(f"   Samples: {results['n_samples']}")
                
            except Exception as e:
                cprint(f"❌ Error processing test split '{test_split_name}': {e}")
                cprint("⏭️  Continuing with next test split...")
                continue
        
        # Print final summary
        cprint(f"\n{'='*60}")
        cprint("🎉 ALL INFERENCE COMPLETED!")
        cprint(f"{'='*60}")
        cprint(f"📊 Processed {len(all_results)} test split(s) successfully")
        
        for i, result in enumerate(all_results):
            cprint(f"\n📈 FINAL RESULTS FOR TEST SPLIT: {result['test_split_name']}")
            if result['task_type'] == 'binary':
                cprint(f"   ROC AUC: {result['roc_auc']:.4f}")
                cprint(f"   KS Statistic: {result['ks_statistic']:.4f}")
            elif result['task_type'] == 'multiclass':
                if result['roc_auc'] is not None:
                    cprint(f"   ROC AUC (macro): {result['roc_auc']:.4f}")
            elif result['task_type'] == 'regression':
                cprint(f"   R² Score: {result['r2_score']:.4f}")
            cprint(f"   Samples: {result['n_samples']}")
        
        cprint(f"\n💾 All results saved to: {args.results_csv}")
        cprint(f"{'='*60}")
        
    except Exception as e:
        cprint(f"❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main() 