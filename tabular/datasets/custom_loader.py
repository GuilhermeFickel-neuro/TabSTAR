import os
import pandas as pd
from pandas import DataFrame
from typing import List, Dict

from tabular.datasets.manual_curation_mapping import get_curated
from tabular.datasets.raw_dataset import RawDataset
from tabular.datasets.raw_loader import get_dataset_description, create_raw_dataset, get_dataframe_types, \
    set_target_drop_redundant_downsample_too_big
from tabular.datasets.tabular_datasets import get_sid, CustomDatasetID
from tabular.preprocessing.feature_type import get_feature_types


class TwoCSVRawDataset(RawDataset):
    """Special RawDataset that holds both train and test data separately for two CSV mode"""
    
    def __init__(self, train_dataset: RawDataset, test_dataset: RawDataset):
        # Use the train dataset as the base, but store test data separately
        super().__init__(
            sid=train_dataset.sid,
            x=train_dataset.x,
            y=train_dataset.y,
            task_type=train_dataset.task_type,
            feature_types=train_dataset.feature_types,
            curation=train_dataset.curation,
            desc=train_dataset.desc,
            source_name=train_dataset.source_name
        )
        self.test_x = test_dataset.x
        self.test_y = test_dataset.y
        self.is_two_csv_mode = True


class MultiTestCSVRawDataset(RawDataset):
    """Special RawDataset that holds one train CSV and multiple test CSVs"""
    
    def __init__(self, train_dataset: RawDataset, test_datasets: List[RawDataset]):
        # Use the train dataset as the base
        super().__init__(
            sid=train_dataset.sid,
            x=train_dataset.x,
            y=train_dataset.y,
            task_type=train_dataset.task_type,
            feature_types=train_dataset.feature_types,
            curation=train_dataset.curation,
            desc=train_dataset.desc,
            source_name=train_dataset.source_name
        )
        # Store test datasets as list of dictionaries
        self.test_datasets = []
        for i, test_dataset in enumerate(test_datasets):
            self.test_datasets.append({
                'x': test_dataset.x,
                'y': test_dataset.y,
                'index': i,
                'source_name': test_dataset.source_name
            })
        self.is_multi_test_csv_mode = True


class InferenceMultiTestCSVRawDataset(RawDataset):
    """Special RawDataset for inference that holds multiple test CSVs only"""
    
    def __init__(self, test_datasets: List[RawDataset]):
        # Use the first test dataset as the base for metadata
        first_test = test_datasets[0]
        super().__init__(
            sid=first_test.sid,
            x=first_test.x,  # This will be replaced with combined data
            y=first_test.y,  # This will be replaced with combined data
            task_type=first_test.task_type,
            feature_types=first_test.feature_types,
            curation=first_test.curation,
            desc=f"Inference dataset with {len(test_datasets)} test splits",
            source_name=f"inference_multi_test"
        )
        # Store test datasets as list of dictionaries
        self.test_datasets = []
        for i, test_dataset in enumerate(test_datasets):
            self.test_datasets.append({
                'x': test_dataset.x,
                'y': test_dataset.y,
                'index': i,
                'source_name': test_dataset.source_name
            })
        self.is_inference_multi_test_csv_mode = True


def load_custom_dataset(dataset_id: CustomDatasetID, csv_path: str, target_column: str, 
                       description: str = "Custom CSV dataset", max_features: int = 1000,
                       custom_test_csv_path: str = None, custom_test_csv_paths: List[str] = None) -> RawDataset:
    """
    Load a custom CSV dataset for TabSTAR training/finetuning.
    
    Args:
        dataset_id: CustomDatasetID enum value
        csv_path: Path to the training CSV file
        target_column: Name of the target column
        description: Optional description of the dataset
        max_features: Maximum number of features to include in the dataset
        custom_test_csv_path: Optional path to the test CSV file (for two CSV mode)
        custom_test_csv_paths: Optional list of paths to multiple test CSV files (for multi-test CSV mode)
    
    Returns:
        RawDataset object ready for TabSTAR processing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV file not found: {csv_path}")
    
    # Check mode: multi-test CSV takes precedence over two CSV mode
    if custom_test_csv_paths is not None and len(custom_test_csv_paths) > 0:
        # Multi-test CSV mode
        for test_csv_path in custom_test_csv_paths:
            if not os.path.exists(test_csv_path):
                raise FileNotFoundError(f"Test CSV file not found: {test_csv_path}")
        return _load_multi_test_csv_dataset(dataset_id, csv_path, custom_test_csv_paths, target_column, description, max_features)
    elif custom_test_csv_path is not None:
        # Two CSV mode (original)
        if not os.path.exists(custom_test_csv_path):
            raise FileNotFoundError(f"Test CSV file not found: {custom_test_csv_path}")
        return _load_two_csv_dataset(dataset_id, csv_path, custom_test_csv_path, target_column, description, max_features)
    else:
        # Single CSV mode (original)
        return _load_single_csv_dataset(dataset_id, csv_path, target_column, description, max_features)


def _load_single_csv_dataset(dataset_id: CustomDatasetID, csv_path: str, target_column: str, 
                            description: str, max_features: int) -> RawDataset:
    """Load a single CSV dataset (original logic)"""
    sid = get_sid(dataset_id)
    
    # Load the CSV file - try to detect separator
    df = _load_csv_file(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    return _create_raw_dataset_from_df(df, target_column, dataset_id, description, max_features, csv_path)


def _load_two_csv_dataset(dataset_id: CustomDatasetID, train_csv_path: str, test_csv_path: str, 
                         target_column: str, description: str, max_features: int) -> TwoCSVRawDataset:
    """Load two CSV datasets and create a special TwoCSVRawDataset"""
    
    # Load both CSV files
    train_df = _load_csv_file(train_csv_path)
    test_df = _load_csv_file(test_csv_path)
    
    # Validate target column exists in both
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training CSV. Available columns: {list(train_df.columns)}")
    
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test CSV. Available columns: {list(test_df.columns)}")
    
    # Validate that both CSVs have the same columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    if train_cols != test_cols:
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols
        error_msg = f"Column mismatch between training and test CSVs."
        if missing_in_test:
            error_msg += f" Missing in test: {missing_in_test}."
        if missing_in_train:
            error_msg += f" Missing in train: {missing_in_train}."
        raise ValueError(error_msg)
    
    # Create train dataset
    train_description = f"{description} (train split)"
    train_dataset = _create_raw_dataset_from_df(train_df, target_column, dataset_id, train_description, max_features, train_csv_path)
    
    # Create test dataset  
    test_description = f"{description} (test split)"
    test_dataset = _create_raw_dataset_from_df(test_df, target_column, dataset_id, test_description, max_features, test_csv_path)
    
    # Return special TwoCSVRawDataset
    return TwoCSVRawDataset(train_dataset, test_dataset)


def _load_multi_test_csv_dataset(dataset_id: CustomDatasetID, train_csv_path: str, test_csv_paths: List[str], 
                                target_column: str, description: str, max_features: int) -> MultiTestCSVRawDataset:
    """Load one train CSV and multiple test CSVs and create a MultiTestCSVRawDataset"""
    
    # Load train CSV
    train_df = _load_csv_file(train_csv_path)
    
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training CSV. Available columns: {list(train_df.columns)}")
    
    # Load all test CSVs and validate them
    test_dfs = []
    train_cols = set(train_df.columns)
    
    for i, test_csv_path in enumerate(test_csv_paths):
        test_df = _load_csv_file(test_csv_path)
        
        # Validate target column exists
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test CSV {i+1} ({test_csv_path}). Available columns: {list(test_df.columns)}")
        
        # Validate that test CSV has the same columns as train CSV
        test_cols = set(test_df.columns)
        
        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            error_msg = f"Column mismatch between training CSV and test CSV {i+1} ({test_csv_path})."
            if missing_in_test:
                error_msg += f" Missing in test: {missing_in_test}."
            if missing_in_train:
                error_msg += f" Missing in train: {missing_in_train}."
            raise ValueError(error_msg)
        
        test_dfs.append(test_df)
    
    # CRITICAL: Combine ALL DataFrames BEFORE any preprocessing
    # Track indices instead of using a temporary column that might get filtered out
    train_size = len(train_df)
    test_sizes = [len(test_df) for test_df in test_dfs]
    
    # Combine all DataFrames
    all_dfs = [train_df] + test_dfs
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Do preprocessing on the COMBINED dataset to ensure consistency
    combined_description = f"{description} (combined train + {len(test_csv_paths)} test CSVs)"
    combined_dataset = _create_raw_dataset_from_df(combined_df, target_column, dataset_id, combined_description, max_features, train_csv_path)
    
    # Now split the processed data back into train and test components using indices
    # Train data: indices 0 to train_size-1
    train_x = combined_dataset.x.iloc[:train_size].reset_index(drop=True)
    train_y = combined_dataset.y.iloc[:train_size].reset_index(drop=True)
    
    # Create train dataset with the processed data
    train_dataset = RawDataset(
        sid=combined_dataset.sid,
        x=train_x,
        y=train_y,
        task_type=combined_dataset.task_type,
        feature_types=combined_dataset.feature_types,
        curation=combined_dataset.curation,
        desc=f"{description} (train split)",
        source_name=f"train_{os.path.basename(train_csv_path)}"
    )
    
    # Extract test datasets using the known sizes
    test_datasets = []
    current_idx = train_size
    for i, (test_size, test_csv_path) in enumerate(zip(test_sizes, test_csv_paths)):
        # Test data: indices current_idx to current_idx+test_size-1
        test_x = combined_dataset.x.iloc[current_idx:current_idx+test_size].reset_index(drop=True)
        test_y = combined_dataset.y.iloc[current_idx:current_idx+test_size].reset_index(drop=True)
        
        test_dataset = RawDataset(
            sid=combined_dataset.sid,
            x=test_x,
            y=test_y,
            task_type=combined_dataset.task_type,
            feature_types=combined_dataset.feature_types,
            curation=combined_dataset.curation,
            desc=f"{description} (test split {i+1})",
            source_name=f"test_{i}_{os.path.basename(test_csv_path)}"
        )
        test_datasets.append(test_dataset)
        current_idx += test_size
    
    # Return special MultiTestCSVRawDataset
    return MultiTestCSVRawDataset(train_dataset, test_datasets)


def _load_csv_file(file_path: str) -> DataFrame:
    """Helper function to load CSV with automatic separator detection"""
    try:
        # First try with tab separator
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        if len(df.columns) == 1:
            # Likely not tab-separated, try comma
            df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        # Fallback to default comma separator
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e2:
            raise ValueError(f"Could not read CSV file {file_path}: {e2}")
    return df


def _create_raw_dataset_from_df(df: DataFrame, target_column: str, dataset_id: CustomDatasetID, 
                               description: str, max_features: int, csv_path: str) -> RawDataset:
    """Create a RawDataset from a DataFrame"""
    sid = get_sid(dataset_id)
    
    # Separate features and target
    x = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Infer task_type from y
    from tabular.preprocessing.objects import SupervisedTask
    unique_values = y.unique()
    num_unique = len(unique_values)

    if num_unique == 2:
        inferred_task_type = SupervisedTask.BINARY
    elif pd.api.types.is_numeric_dtype(y) and num_unique > 20: # Heuristic for regression
        inferred_task_type = SupervisedTask.REGRESSION
    elif num_unique <= 20 : # Heuristic for multiclass
        inferred_task_type = SupervisedTask.MULTICLASS
    else: # Default or fallback
        inferred_task_type = SupervisedTask.REGRESSION

    # Create a basic curation object (you can customize this)
    from tabular.datasets.manual_curation_obj import CuratedDataset, CuratedTarget, CuratedFeature
    
    # Create target specification
    target = CuratedTarget(
        raw_name=target_column,
        task_type=inferred_task_type # Use inferred task type
    )
    
    # Create basic feature specifications for all columns
    # For custom datasets, we'll create minimal curation to avoid validation issues
    features = []  # Keep this empty to avoid validation issues during downsampling
    
    csv_name = os.path.basename(csv_path)
    
    curation = CuratedDataset(
        name=f"custom_{csv_name}",
        target=target,
        features=features,  # Empty list to avoid column validation issues
        cols_to_drop=[],
        context=f"{description or 'Custom dataset for TabSTAR finetuning'} max_features:{max_features}",
        description=description
    )
    
    # Get dataset description
    dataset_description = get_dataset_description(
        name=f"Custom_{dataset_id.name}", 
        url=csv_path, 
        desc=description, 
        x=x, 
        y=y
    )
    
    # Process the dataset
    x, y, task_type, curation = set_target_drop_redundant_downsample_too_big(
        x=x, y=y, curation=curation, sid=sid
    )
    
    # Get feature types
    df_types = get_dataframe_types(x)
    feature_types = get_feature_types(x=x, curation=curation, feat_types=df_types)
    
    # Create the raw dataset
    raw = create_raw_dataset(
        x=x, y=y, curation=curation, desc=dataset_description, 
        feat_types=feature_types, sid=sid, task_type=task_type, 
        source_name=f"custom_{csv_name}"
    )
    
    return raw 


def load_inference_multi_test_dataset(dataset_id: CustomDatasetID, test_csv_paths: List[str], 
                                     target_column: str, description: str = "Inference multi-test dataset", 
                                     max_features: int = 1000) -> InferenceMultiTestCSVRawDataset:
    """Load multiple test CSVs for inference and create an InferenceMultiTestCSVRawDataset"""
    
    # Load all test CSVs and validate them
    test_dfs = []
    first_cols = None
    
    for i, test_csv_path in enumerate(test_csv_paths):
        test_df = _load_csv_file(test_csv_path)
        
        # Validate target column exists
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test CSV {i+1} ({test_csv_path}). Available columns: {list(test_df.columns)}")
        
        # Validate that all test CSVs have the same columns
        test_cols = set(test_df.columns)
        if first_cols is None:
            first_cols = test_cols
        elif first_cols != test_cols:
            missing_in_current = first_cols - test_cols
            missing_in_first = test_cols - first_cols
            error_msg = f"Column mismatch between test CSV 1 and test CSV {i+1} ({test_csv_path})."
            if missing_in_current:
                error_msg += f" Missing in current: {missing_in_current}."
            if missing_in_first:
                error_msg += f" Missing in first: {missing_in_first}."
            raise ValueError(error_msg)
        
        test_dfs.append(test_df)
    
    # Create test datasets
    test_datasets = []
    for i, (test_df, test_csv_path) in enumerate(zip(test_dfs, test_csv_paths)):
        test_description = f"{description} (test split {i+1})"
        test_dataset = _create_raw_dataset_from_df(test_df, target_column, dataset_id, test_description, max_features, test_csv_path)
        test_datasets.append(test_dataset)
    
    # Return special InferenceMultiTestCSVRawDataset
    return InferenceMultiTestCSVRawDataset(test_datasets) 