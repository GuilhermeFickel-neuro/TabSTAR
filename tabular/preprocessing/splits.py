from collections import Counter
from enum import StrEnum
from typing import List, Dict, Tuple

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.objects import SupervisedTask, PreprocessingMethod, CV_METHODS
from tabular.utils.utils import SEED, verbose_print
from tabular.datasets.custom_loader import TwoCSVRawDataset, MultiTestCSVRawDataset

TEST_RATIO = 0.1
MAX_TEST_SIZE = 2000

NN_DEV_RATIO = 0.1
NN_PRETRAIN_DEV_RATIO = 0.05
MAX_DEV_SIZE = 1000

MIN_TOTAL_EXAMPLES = 100


class DataSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

    @classmethod
    def get_test_split_name(cls, test_index: int) -> str:
        """Generate test split name for multiple test datasets"""
        if test_index == 0:
            return cls.TEST  # First test split uses the standard "test" name
        else:
            return f"test{test_index + 1}"  # Additional test splits: test2, test3, etc.

    @classmethod
    def get_all_test_splits(cls, num_test_datasets: int) -> List[str]:
        """Get all test split names for the given number of test datasets"""
        return [cls.get_test_split_name(i) for i in range(num_test_datasets)]


def create_splits(raw: RawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> TabularDatasetSplits:
    if isinstance(raw, MultiTestCSVRawDataset):
        return create_multi_test_csv_splits(raw, run_num, train_examples, processing)
    else:
        # Default behavior: single CSV split into train/dev/test
        return create_regular_splits(raw, run_num, train_examples, processing)


def create_multi_test_csv_splits(raw: MultiTestCSVRawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> TabularDatasetSplits:
    """Create data splits when using multiple test CSVs."""
    
    # Train data is split into train/dev
    train_dev_splits = create_regular_splits(
        raw,  # The train part of the raw dataset
        run_num, 
        train_examples, 
        processing
    )

    all_splits = {
        DataSplit.TRAIN: train_dev_splits.train,
        DataSplit.DEV: train_dev_splits.dev,
    }

    # The first test dataset becomes the main 'test' split
    if raw.test_datasets:
        first_test_raw = raw.test_datasets[0]
        all_splits[DataSplit.TEST] = TabularDatasetSplit(
            x=first_test_raw['x'],
            y=first_test_raw['y'],
            processing=processing
        )
        
        # Additional test sets are named test2, test3, etc.
        for i, test_raw in enumerate(raw.test_datasets[1:], start=2):
            split_name = f"test{i}"
            all_splits[DataSplit(split_name)] = TabularDatasetSplit(
                x=test_raw['x'],
                y=test_raw['y'],
                processing=processing
            )

    return TabularDatasetSplits(all_splits)


def create_regular_splits(raw: RawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> TabularDatasetSplits:
    """Create train/dev/test splits from a single RawDataset."""
    n = len(raw.x)
    n_train_dev = get_train_dev_size(n, train_examples)
    splits = get_fixed_split(n, n_train_dev, run_num)
    
    split_array = _create_split_array(n, splits)
    
    return TabularDatasetSplits({
        DataSplit(s): TabularDatasetSplit(
            x=raw.x[split_array == s],
            y=raw.y[split_array == s],
            processing=processing
        )
        for s in [DataSplit.TRAIN, DataSplit.DEV, DataSplit.TEST]
        if np.any(split_array == s)
    })


def create_two_csv_splits(raw: RawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> List[DataSplit]:
    """Create splits for two CSV mode: train CSV -> train/dev splits, test CSV -> test split"""
    # raw is a TwoCSVRawDataset with train data in x/y and test data in test_x/test_y
    
    train_n = len(raw.y)  # Size of train CSV data
    test_n = len(raw.test_y)  # Size of test CSV data
    total_n = train_n + test_n
    
    if train_n < MIN_TOTAL_EXAMPLES:
        raise ValueError(f"Training dataset {raw.sid} has too few examples: {train_n}")
    
    is_pretrain = bool(train_examples < 0)
    use_dev = _uses_dev(processing)
    
    # Create indices for train data (0 to train_n-1)
    train_indices = list(range(train_n))
    
    if not is_pretrain:
        # Apply train_examples limit if specified
        train_indices, exclude = _get_exclude_two_csv(raw=raw, indices=train_indices, run_num=run_num, train_examples=train_examples)
    
    # Split train data into train/dev
    train_final, dev = _get_train_dev(raw=raw, indices=train_indices, use_dev=use_dev, run_num=run_num, is_pretrain=is_pretrain)
    
    # Test indices start after train data (train_n to train_n+test_n-1)
    test = list(range(train_n, train_n + test_n))
    
    # Combine the datasets: train data first, then test data
    combined_x = pd.concat([raw.x, raw.test_x], ignore_index=True)
    combined_y = pd.concat([raw.y, raw.test_y], ignore_index=True)
    
    # Update the raw dataset to contain combined data
    raw.x = combined_x
    raw.y = combined_y
    
    # Create split assignments
    splits = {DataSplit.TRAIN: train_final, DataSplit.DEV: dev, DataSplit.TEST: test}
    split_array = _create_split_array_two_csv(total_n, splits)
    
    verbose_print(f"Created two-CSV splits for {raw.sid}: train_data={train_n}, test_data={test_n}, {train_examples=}: {Counter(split_array)}")
    return split_array


def _get_exclude_two_csv(raw: RawDataset, indices: List[int], run_num: int, train_examples: int) -> Tuple[List[int], List[int]]:
    """Modified version of _get_exclude for two CSV mode"""
    if len(indices) < train_examples:
        return indices, []
    exclude_examples = len(indices) - train_examples
    return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=exclude_examples)


def _create_split_array_two_csv(total_n: int, splits: Dict[DataSplit, List[int]]) -> List[DataSplit]:
    """Create split array for two CSV mode without sampling (data is already combined)"""
    idx2split = {i: split for split, indices in splits.items() for i in indices}
    split_array = [idx2split.get(i) for i in range(total_n)]
    # All indices should be assigned in two CSV mode
    assert all(s is not None for s in split_array), "Some indices were not assigned to any split"
    return split_array


def _get_test(raw: RawDataset, indices: List[int], n: int, run_num: int) -> Tuple[List[int], List[int]]:
    test_size = int(n * TEST_RATIO)
    test_size = min(test_size, MAX_TEST_SIZE)
    return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=test_size)

def _get_exclude(raw: RawDataset, indices: List[int], run_num: int, train_examples: int) -> Tuple[List[int], List[int]]:
    if len(indices) < train_examples:
        return indices, []
    exclude_examples = len(indices) - train_examples
    return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=exclude_examples)


def _get_train_dev(raw: RawDataset, indices: List[int], use_dev: bool, run_num: int,
                   is_pretrain: bool) -> Tuple[List[int], List[int]]:
    if not use_dev:
        return indices, []
    nn_dev_ratio = NN_PRETRAIN_DEV_RATIO if is_pretrain else NN_DEV_RATIO
    dev_size = int(len(indices) * nn_dev_ratio)
    dev_size = min(dev_size, MAX_DEV_SIZE)
    return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=dev_size)


def get_x_train(x: DataFrame, splits: List[DataSplit]) -> DataFrame:
    return get_x_split(x=x, splits=splits, split=DataSplit.TRAIN)

def get_x_split(x: DataFrame, splits: List[DataSplit], split: DataSplit) -> DataFrame:
    indices = [i for i, s in enumerate(splits) if s == split]
    return x.iloc[indices].reset_index(drop=True)

def get_y_train(y: Series, splits: List[DataSplit]) -> Series:
    return get_y_split(y=y, splits=splits, split=DataSplit.TRAIN)

def get_y_split(y: Series, splits: List[DataSplit], split: DataSplit) -> Series:
    indices = [i for i, s in enumerate(splits) if s == split]
    return y.iloc[indices].reset_index(drop=True)


def _do_split(raw: RawDataset, indices: List[int], run_num: int, test_size: int) -> Tuple[List[int], List[int]]:
    random_state = SEED + run_num
    stratify = raw.y.iloc[indices] if raw.task_type != SupervisedTask.REGRESSION else None
    try:
        train, test = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError as e:
        assert raw.task_type != SupervisedTask.REGRESSION
        train, test = train_test_split(indices, test_size=test_size, random_state=random_state)
        train_classes = set(raw.y.iloc[train])
        missing_class_indices = [idx for idx in test if raw.y.iloc[idx] not in train_classes]
        if missing_class_indices:
            train.extend(missing_class_indices)
            test = [idx for idx in test if idx not in missing_class_indices]
    return train, test

def _sample_xy_and_get_array(raw: RawDataset, n: int, splits: Dict[DataSplit, List[int]]) -> List[DataSplit]:
    idx2split = {i: split for split, indices in splits.items() for i in indices}
    splits_array = [idx2split.get(i) for i in range(n)]
    valid_mask = [v is not None for v in splits_array]
    raw.x = raw.x[valid_mask].reset_index(drop=True)
    raw.y = raw.y[valid_mask].reset_index(drop=True)
    split_array = [s for s, valid in zip(splits_array, valid_mask) if valid]
    return split_array

def _uses_dev(processing: PreprocessingMethod) -> bool:
    if processing in CV_METHODS:
        return False
    process2dev = {PreprocessingMethod.TABSTAR: True,
                   PreprocessingMethod.CATBOOST: True,
                   PreprocessingMethod.TREES: True,
                   # TabPFN-v2 and CARTE don't use dev
                   PreprocessingMethod.TABPFNV2: False,
                   PreprocessingMethod.CARTE: False,
                   }
    return process2dev[processing]

def create_inference_multi_test_csv_splits(raw: RawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> List[DataSplit]:
    """Create splits for inference-only mode: multiple test CSVs -> test1, test2, ... splits (no train/dev)"""
    
    num_test_datasets = len(raw.test_datasets)
    test_sizes = [len(test_data['y']) for test_data in raw.test_datasets]
    total_n = sum(test_sizes)
    
    if total_n < MIN_TOTAL_EXAMPLES:
        raise ValueError(f"Total test dataset {raw.sid} has too few examples: {total_n}")
    
    # Create test indices for each test dataset (no train/dev splits needed for inference)
    test_splits = {}
    current_idx = 0
    for i, test_size in enumerate(test_sizes):
        test_split_name = DataSplit.get_test_split_name(i)
        test_indices = list(range(current_idx, current_idx + test_size))
        test_splits[test_split_name] = test_indices
        current_idx += test_size
    
    # Combine all test datasets
    combined_x_list = []
    combined_y_list = []
    
    for test_data in raw.test_datasets:
        combined_x_list.append(test_data['x'])
        combined_y_list.append(test_data['y'])
    
    combined_x = pd.concat(combined_x_list, ignore_index=True)
    combined_y = pd.concat(combined_y_list, ignore_index=True)
    
    # Update the raw dataset to contain combined data
    raw.x = combined_x
    raw.y = combined_y
    
    # Create split assignments (only test splits, no train/dev)
    splits = test_splits
    
    split_array = _create_split_array_multi_csv(total_n, splits)
    
    test_summary = ', '.join([f"{name}={len(indices)}" for name, indices in test_splits.items()])
    verbose_print(f"Created inference-only test splits for {raw.sid}: test_datasets=({test_summary}), total={total_n}: {Counter(split_array)}")
    return split_array


def _create_split_array_multi_csv(total_n: int, splits: Dict[str, List[int]]) -> List[str]:
    """Create split array for multi-test CSV mode without exclusions (data is already filtered)"""
    idx2split = {i: split for split, indices in splits.items() for i in indices}
    split_array = [idx2split.get(i) for i in range(total_n)]
    # All indices should be assigned in filtered multi-test CSV mode
    assert all(s is not None for s in split_array), "Some indices were not assigned to any split"
    return split_array