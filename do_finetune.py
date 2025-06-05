import argparse
from typing import Optional

import torch

from tabular.datasets.tabular_datasets import TabularDatasetID, get_dataset_from_arg
from tabular.evaluation.constants import DOWNSTREAM_EXAMPLES, N_RUNS
from tabular.tabstar.tabstar_trainer import TabStarTrainer
from tabular.tabstar.params.constants import LORA_LR, LORA_BATCH, LORA_R
from tabular.trainers.finetune import do_finetune_run
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.early_stopping import FINETUNE_PATIENCE
from tabular.utils.gpus import get_device


def finetune_tabstar(finetune_args: FinetuneArgs,
                     dataset: TabularDatasetID,
                     run_num: int,
                     train_examples: int = DOWNSTREAM_EXAMPLES,
                     device: Optional[torch.device] = None,
                     custom_csv_path: str = None,
                     custom_target_column: str = None,
                     custom_max_features: int = 2500,
                     custom_test_csv_path: str = None):
    if device is None:
        device = torch.device(get_device())
    if dataset.value in finetune_args.pretrain_args.datasets:
        raise NotImplementedError(f"😱😱😱 Dataset {dataset} is already in pretrain datasets, beware!")
    do_finetune_run(dataset=dataset, model=TabStarTrainer, run_num=run_num, device=device,
                    finetune_args=finetune_args, exp_name=finetune_args.full_exp_name,
                    train_examples=train_examples, custom_csv_path=custom_csv_path,
                    custom_target_column=custom_target_column, custom_max_features=custom_max_features,
                    custom_test_csv_path=custom_test_csv_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_exp', type=str, required=True)
    parser.add_argument('--dataset_id', type=str, default="46667")
    parser.add_argument('--exp', type=str, default="default_finetune_exp")
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--downstream_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--downstream_keep_model', type=bool, default=True)
    parser.add_argument('--downstream_patience', type=int, default=FINETUNE_PATIENCE)
    parser.add_argument('--lora_lr', type=float, default=LORA_LR)
    parser.add_argument('--lora_batch', type=int, default=LORA_BATCH)
    parser.add_argument('--lora_r', type=int, default=LORA_R)
    # Custom dataset arguments
    parser.add_argument('--custom_csv_path', type=str, help='Path to custom CSV file (required when dataset_id is "custom")')
    parser.add_argument('--custom_target_column', type=str, help='Name of target column in custom CSV (required when dataset_id is "custom")')
    parser.add_argument('--custom_max_features', type=int, default=1000, help='Maximum number of features for custom datasets (default: 1000)')
    parser.add_argument('--custom_test_csv_path', type=str, help='Path to test CSV file (optional, for two CSV mode)')

    args = parser.parse_args()
    assert args.pretrain_exp, "Pretrain path is required"
    
    # Handle custom dataset validation
    if args.dataset_id == "custom":
        if not args.custom_csv_path or not args.custom_target_column:
            raise ValueError("For custom datasets, both --custom_csv_path and --custom_target_column must be provided")
        if not args.custom_csv_path.endswith('.csv'):
            raise ValueError("Custom dataset file must be a CSV file")
        if args.custom_test_csv_path and not args.custom_test_csv_path.endswith('.csv'):
            raise ValueError("Custom test dataset file must be a CSV file")
    
    data = get_dataset_from_arg(args.dataset_id)
    assert 0 <= args.run_num < N_RUNS, f"Invalid run number: {args.run_num}. Should be between 0 and {N_RUNS - 1}"

    pretrain_args = PretrainArgs.from_json(pretrain_exp=args.pretrain_exp)
    run_args = FinetuneArgs.from_args(args=args, pretrain_args=pretrain_args, exp_name=args.exp)
    finetune_tabstar(finetune_args=run_args, dataset=data,
                     run_num=args.run_num, train_examples=args.downstream_examples,
                     custom_csv_path=args.custom_csv_path, custom_target_column=args.custom_target_column,
                     custom_max_features=args.custom_max_features, custom_test_csv_path=args.custom_test_csv_path)
