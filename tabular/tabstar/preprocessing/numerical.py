from typing import Tuple, List

import numpy as np
from pandas import DataFrame
from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.splits import DataSplit, get_x_train
from tabular.tabstar.params.constants import NumberVerbalization
from tabular.tabstar.preprocessing.numerical_quantiles import verbalized_quantiles_og_values
from tabular.tabstar.preprocessing.numerical_scaling import standardize_series
from tabular.tabstar.preprocessing.numerical_utils import fit_scaler
from tabular.utils.utils import verbose_print


def scale_x_num_and_add_categorical_bins(raw: RawDataset, splits: List[DataSplit],
                                         number_verbalization: NumberVerbalization) -> Tuple[DataFrame, np.ndarray]:
    verbose_print(f"Scaling {len(raw.numerical)} numerical features for {raw.sid}")
    
    # sorted_cols defines the columns and their order for x_txt and x_num alignment
    sorted_cols = raw.numerical + raw.bool_cat_text
    
    # Initialize x_num with the same number of columns as x_txt will have.
    # Non-numerical feature columns in x_num will remain as zeros.
    x_num = np.zeros(shape=(raw.x.shape[0], len(sorted_cols)), dtype=np.float32)
    
    x_train = get_x_train(x=raw.x, splits=splits)
    
    # Create a mapping from column name to its index in sorted_cols for placing numerical values
    col_to_idx = {col_name: idx for idx, col_name in enumerate(sorted_cols)}

    for col_name in raw.numerical: # Iterate only over numerical columns for scaling
        train_col_data = x_train[col_name]
        # It's important to use .copy() when intending to modify a slice/subset of a DataFrame
        # if the original DataFrame (raw.x) should not be altered by intermediate scaling steps,
        # or to avoid SettingWithCopyWarning if original_col_data is just a view.
        # However, raw.x[col_name] is being updated with verbalized values for x_txt creation later.
        original_col_data = raw.x[col_name].copy() 

        # Verbalize numerical features and update them in raw.x (this will be used for x_txt)
        quantile_scaler = fit_scaler(train_col_data, for_verbalize=True)
        raw.x[col_name] = verbalized_quantiles_og_values(series=original_col_data, quantile_scaler=quantile_scaler,
                                                        number_verbalization=number_verbalization)
        
        # Scale numerical features for x_num
        standard_scaler = fit_scaler(train_col_data, for_verbalize=False)
        scaled_values = standardize_series(series=original_col_data, scaler=standard_scaler)
        
        # Place scaled numerical values into the correct column in x_num
        if col_name in col_to_idx: # Should always be true for raw.numerical if sorted_cols is built correctly
            num_col_idx = col_to_idx[col_name]
            x_num[:, num_col_idx] = scaled_values
        else:
            # This case should ideally not happen if raw.numerical is a subset of sorted_cols' components
            verbose_print(f"Warning: Numerical column {col_name} not found in sorted_cols map during x_num population.")

    # x_txt takes columns from raw.x, which now contains verbalized numerical features.
    # The order of columns in x_txt is determined by sorted_cols.
    x_txt = raw.x[sorted_cols].copy()
    
    verbose_print(f"Done scaling numerical!")
    return x_txt, x_num