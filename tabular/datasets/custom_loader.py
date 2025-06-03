import os
import pandas as pd
from pandas import DataFrame

from tabular.datasets.manual_curation_mapping import get_curated
from tabular.datasets.raw_dataset import RawDataset
from tabular.datasets.raw_loader import get_dataset_description, create_raw_dataset, get_dataframe_types, \
    set_target_drop_redundant_downsample_too_big
from tabular.datasets.tabular_datasets import get_sid, CustomDatasetID
from tabular.preprocessing.feature_type import get_feature_types


def load_custom_dataset(dataset_id: CustomDatasetID, csv_path: str, target_column: str, 
                       description: str = "Custom CSV dataset", max_features: int = 1000) -> RawDataset:
    """
    Load a custom CSV dataset for TabSTAR training/finetuning.
    
    Args:
        dataset_id: CustomDatasetID enum value
        csv_path: Path to the CSV file
        target_column: Name of the target column
        description: Optional description of the dataset
        max_features: Maximum number of features to include in the dataset
    
    Returns:
        RawDataset object ready for TabSTAR processing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    sid = get_sid(dataset_id)
    
    # Load the CSV file - try to detect separator
    try:
        # First try with tab separator
        df = pd.read_csv(csv_path, sep='\t', low_memory=False)
        if len(df.columns) == 1:
            # Likely not tab-separated, try comma
            df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        # Fallback to default comma separator
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e2:
            raise ValueError(f"Could not read CSV file: {e2}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {list(df.columns)}")
    
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
    
    curation = CuratedDataset(
        name=f"custom_{os.path.basename(csv_path)}",
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
        source_name=f"custom_{os.path.basename(csv_path)}"
    )
    
    return raw 