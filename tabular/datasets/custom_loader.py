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
                       description: str = "Custom CSV dataset") -> RawDataset:
    """
    Load a custom CSV dataset for TabSTAR training/finetuning.
    
    Args:
        dataset_id: CustomDatasetID enum value
        csv_path: Path to the CSV file
        target_column: Name of the target column
        description: Optional description of the dataset
    
    Returns:
        RawDataset object ready for TabSTAR processing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    sid = get_sid(dataset_id)
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Separate features and target
    x = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Create a basic curation object (you can customize this)
    from tabular.datasets.manual_curation_obj import CuratedDataset
    curation = CuratedDataset(
        target=target_column,
        description=description,
        drop_columns=[],
        categorical_columns=[],
        text_columns=[],
        ordinal_columns={},
        binary_columns=[],
        date_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        skewed_columns=[],
        outlier_columns=[],
        missing_columns=[],
        correlated_columns=[],
        redundant_columns=[],
        leaky_columns=[],
        noisy_columns=[],
        irrelevant_columns=[],
        task_type=None  # Will be auto-detected
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