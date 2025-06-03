import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

# TabSTAR imports
from tabular.datasets.tabular_datasets import TabularDatasetID
from tabular.datasets.properties import DatasetProperties
from tabular.datasets.paths import get_data_dir
from tabular.evaluation.metrics import calculate_ks_metric
from tabular.evaluation.loss import apply_loss_fn
from tabular.preprocessing.preprocessor import Preprocessor
from tabular.tabstar.tabstar_trainer import TabStarTrainer
from tabular.tabstar.params.constants import LORA_R, LORA_LR, LORA_BATCH # Default LoRA params
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.gpus import get_device
from tabular.preprocessing.splits import DataSplit
from tabular.utils.early_stopping import FINETUNE_PATIENCE
from tabular.evaluation.constants import DOWNSTREAM_EXAMPLES
from tabular.utils.utils import cprint
from tabular.preprocessing.objects import SupervisedTask, PreprocessingMethod


def run_inference():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned TabSTAR model.")
    parser.add_argument('--pretrain_exp', type=str, required=True, help="Name of the pretraining experiment (for base model).")
    parser.add_argument('--finetune_exp', type=str, required=True, help="Name of the fine-tuning experiment (for LoRA adapter and config).")
    
    parser.add_argument('--original_finetune_custom_csv_path', type=str, required=True, 
                        help="Path to the custom CSV file that was used for the original fine-tuning. Needed to load the correct preprocessor.")
    
    parser.add_argument('--inference_custom_csv_path', type=str, required=True, help="Path to the new custom CSV file for inference.")
    parser.add_argument('--inference_target_column', type=str, required=True, help="Name of the target column in the inference CSV.")
    
    parser.add_argument('--custom_max_features', type=int, default=2500, help="Maximum number of features for the preprocessor.")
    
    parser.add_argument('--lora_r', type=int, default=LORA_R, help="LoRA R parameter.")
    parser.add_argument('--lora_lr', type=float, default=LORA_LR, help="LoRA learning rate.")
    parser.add_argument('--lora_batch', type=int, default=LORA_BATCH, help="LoRA batch size.")
    parser.add_argument('--run_num', type=int, default=0, help="Run number (for FinetuneArgs consistency).")
    parser.add_argument('--downstream_examples', type=int, default=DOWNSTREAM_EXAMPLES, help="Downstream examples (for FinetuneArgs consistency).")
    parser.add_argument('--downstream_patience', type=int, default=FINETUNE_PATIENCE, help="Downstream patience (for FinetuneArgs consistency).")
    parser.add_argument('--downstream_keep_model', action='store_true', default=True, help="FinetuneArgs consistency, effectively true for loading.")

    args = parser.parse_args()

    if not os.path.exists(args.inference_custom_csv_path):
        raise FileNotFoundError(f"Inference CSV file not found: {args.inference_custom_csv_path}")
    if not args.inference_custom_csv_path.endswith('.csv'):
        raise ValueError("Inference dataset file must be a CSV file.")

    cprint(f"Using device: {get_device()}", "yellow")
    device = torch.device(get_device())

    cprint(f"Loading PretrainArgs from: {args.pretrain_exp}", "cyan")
    pretrain_args = PretrainArgs.from_json(pretrain_exp=args.pretrain_exp)

    finetune_config_ns = argparse.Namespace(
        pretrain_exp=args.pretrain_exp,
        dataset_id=args.original_finetune_custom_csv_path, 
        exp=args.finetune_exp,
        run_num=args.run_num,
        downstream_examples=args.downstream_examples,
        downstream_keep_model=args.downstream_keep_model,
        downstream_patience=args.downstream_patience,
        lora_lr=args.lora_lr,
        lora_batch=args.lora_batch,
        lora_r=args.lora_r,
        custom_csv_path=args.original_finetune_custom_csv_path,
        custom_target_column="dummy_original_target", # Not strictly needed if preprocessor loads OK
        custom_max_features=args.custom_max_features 
    )
    cprint(f"Creating FinetuneArgs for experiment: {args.finetune_exp}", "cyan")
    finetune_args = FinetuneArgs.from_args(args=finetune_config_ns, pretrain_args=pretrain_args, exp_name=args.finetune_exp)

    cprint(f"Loading preprocessor for original fine-tuning dataset: {args.original_finetune_custom_csv_path}", "cyan")
    original_dataset_id_obj = TabularDatasetID(name=args.original_finetune_custom_csv_path, is_custom=True)
    
    original_data_dir = get_data_dir(
        sid=original_dataset_id_obj.path_safe_name,
        args=finetune_args, 
        process_method_name=PreprocessingMethod.TABSTAR.value,
        custom_path=args.original_finetune_custom_csv_path
    )
    
    preprocessor_path = os.path.join(original_data_dir, "preprocessor.pkl")
    dataset_params_path = os.path.join(original_data_dir, "params.json")

    if not os.path.exists(preprocessor_path) or not os.path.exists(dataset_params_path):
        raise FileNotFoundError(f"Preprocessor or params.json not found in {original_data_dir}. "
                                "Ensure --original_finetune_custom_csv_path is correct and was processed during fine-tuning.")

    original_dataset_properties = DatasetProperties.from_json(dataset_params_path)
    preprocessor = Preprocessor.load(
        preprocessor_path, 
        properties=original_dataset_properties, 
        args=finetune_args, 
        data_dir=original_data_dir
    )
    cprint(f"Preprocessor loaded. Original task type: {original_dataset_properties.task_type.value}", "green")

    dummy_inference_ds_id = TabularDatasetID(name=args.inference_custom_csv_path, is_custom=True)
    trainer = TabStarTrainer(
        run_name=finetune_args.full_exp_name,
        dataset_ids=[dummy_inference_ds_id], 
        device=device,
        run_num=finetune_args.run_num,
        args=finetune_args,
        custom_csv_path=args.inference_custom_csv_path, 
        custom_target_column=args.inference_target_column,
        custom_max_features=args.custom_max_features
    )

    cprint(f"Loading fine-tuned model from directory: {trainer.model_path}", "cyan")
    trainer.load_model(cp_path=trainer.model_path)
    trainer.model.eval()
    cprint("Model loaded successfully.", "green")

    cprint(f"Loading inference data from: {args.inference_custom_csv_path}", "cyan")
    df_inference = pd.read_csv(args.inference_custom_csv_path)
    
    if args.inference_target_column not in df_inference.columns:
        raise ValueError(f"Target column '{args.inference_target_column}' not found in inference CSV: {args.inference_custom_csv_path}")
    
    X_infer = df_inference.drop(columns=[args.inference_target_column])
    y_infer_true_raw = df_inference[args.inference_target_column]

    cprint("Preprocessing inference data...", "cyan")
    processed_dict = preprocessor.transform(df=X_infer, y=y_infer_true_raw, split_type=DataSplit.TEST)
    
    x_txt_infer = processed_dict['x_cat_values'] 
    x_num_infer = processed_dict['x_num_values'] 
    y_infer_true_processed = processed_dict['y_values']

    if not isinstance(x_txt_infer, np.ndarray): x_txt_infer = np.array(x_txt_infer, dtype=object)
    if not isinstance(x_num_infer, np.ndarray): x_num_infer = np.array(x_num_infer, dtype=float)
    if not isinstance(y_infer_true_processed, np.ndarray): y_infer_true_processed = np.array(y_infer_true_processed)
    
    cprint("Inference data preprocessed.", "green")

    cprint("Performing inference...", "cyan")
    with torch.no_grad():
        inference_output = trainer.infer(x_txt_infer, x_num_infer, original_dataset_properties)
        y_pred_raw_model = inference_output.y_pred
        y_pred_final = apply_loss_fn(y_pred_raw_model, original_dataset_properties.task_type)
    
    y_pred_final_np = y_pred_final.cpu().numpy()
    cprint("Inference complete.", "green")

    if y_infer_true_processed.ndim > 1 and y_infer_true_processed.shape[1] == 1:
        y_infer_true_processed = y_infer_true_processed.ravel()

    if original_dataset_properties.task_type == SupervisedTask.BINARY:
        if y_pred_final_np.ndim > 1 and y_pred_final_np.shape[1] == 1:
             y_pred_final_np = y_pred_final_np.ravel()

        roc_auc = roc_auc_score(y_infer_true_processed, y_pred_final_np)
        ks_stat = calculate_ks_metric(y_true=y_infer_true_processed, y_pred_proba=y_pred_final_np)
        cprint(f"\\n--- Inference Metrics (Binary Classification) ---", "blue")
        cprint(f"ROC AUC: {roc_auc:.4f}", "green")
        cprint(f"KS Statistic: {ks_stat:.4f}", "green")
    elif original_dataset_properties.task_type == SupervisedTask.REGRESSION:
        if y_pred_final_np.ndim > 1 and y_pred_final_np.shape[1] == 1:
            y_pred_final_np = y_pred_final_np.ravel()
        mse = mean_squared_error(y_infer_true_processed, y_pred_final_np)
        cprint(f"\\n--- Inference Metrics (Regression) ---", "blue")
        cprint(f"Mean Squared Error: {mse:.4f}", "green")
    else: # Multiclass
        cprint(f"\\n--- Inference Metrics (Multiclass) ---", "blue")
        # Example: sklearn's roc_auc_score can handle multiclass with probabilities
        try:
            roc_auc_multi = roc_auc_score(y_infer_true_processed, y_pred_final_np, multi_class='ovr')
            cprint(f"ROC AUC (OvR): {roc_auc_multi:.4f}", "green")
        except ValueError as e:
            cprint(f"Could not compute multiclass ROC AUC: {e}", "yellow")
        cprint(f"Note: KS statistic is typically for binary classification.", "yellow")

if __name__ == "__main__":
    run_inference() 