from collections import Counter
from enum import StrEnum
from typing import List, Dict, Tuple

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.objects import SupervisedTask, PreprocessingMethod, CV_METHODS
from tabular.utils.utils import SEED, verbose_print

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


def create_splits(raw: RawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> List[DataSplit]:
    # Check if this is a two CSV mode dataset
    if hasattr(raw, 'is_two_csv_mode') and raw.is_two_csv_mode:
        return create_two_csv_splits(raw, run_num, train_examples, processing)
    
    # Original single CSV logic
    n = len(raw.y)
    if n < MIN_TOTAL_EXAMPLES:
        raise ValueError(f"Dataset {raw.sid} has too few examples: {n}")
    indices = list(range(n))
    is_pretrain = bool(train_examples < 0)
    use_dev = _uses_dev(processing)
    if is_pretrain:
        test = []
    else:
        indices, test = _get_test(raw=raw, indices=indices, n=n, run_num=run_num)
        indices, exclude = _get_exclude(raw=raw, indices=indices, run_num=run_num, train_examples=train_examples)
    train, dev = _get_train_dev(raw=raw, indices=indices, use_dev=use_dev, run_num=run_num, is_pretrain=is_pretrain)
    splits = {DataSplit.TRAIN: train, DataSplit.DEV: dev, DataSplit.TEST: test}
    split_array = _sample_xy_and_get_array(raw=raw, n=n, splits=splits)
    verbose_print(f"Created splits for {raw.sid} of length {n} and {train_examples=}: {Counter(split_array)}")
    return split_array


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