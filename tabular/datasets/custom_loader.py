import os
import pandas as pd
from pandas import DataFrame

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


def load_custom_dataset(dataset_id: CustomDatasetID, csv_path: str, target_column: str, 
                       description: str = "Custom CSV dataset", max_features: int = 1000,
                       custom_test_csv_path: str = None) -> RawDataset:
    """
    Load a custom CSV dataset for TabSTAR training/finetuning.
    
    Args:
        dataset_id: CustomDatasetID enum value
        csv_path: Path to the training CSV file
        target_column: Name of the target column
        description: Optional description of the dataset
        max_features: Maximum number of features to include in the dataset
        custom_test_csv_path: Optional path to the test CSV file (for two CSV mode)
    
    Returns:
        RawDataset object ready for TabSTAR processing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV file not found: {csv_path}")
    
    # Check if using two CSV mode
    if custom_test_csv_path is not None:
        if not os.path.exists(custom_test_csv_path):
            raise FileNotFoundError(f"Test CSV file not found: {custom_test_csv_path}")
        return _load_two_csv_dataset(dataset_id, csv_path, custom_test_csv_path, target_column, description, max_features)
    else:
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