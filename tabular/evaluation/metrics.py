from dataclasses import dataclass, field
from typing import List

import numpy as np
from pandas import Series
from sklearn.metrics import roc_auc_score, r2_score
from tabpfn_extensions.scoring.scoring_utils import safe_roc_auc_score
from torch import Tensor
from scipy.stats import ks_2samp

from tabular.preprocessing.objects import SupervisedTask

from tabular.utils.utils import verbose_print


@dataclass
class PredictionsCache:
    predictions: List[Tensor] = field(default_factory=list)
    labels: List[np.ndarray] = field(default_factory=list)

    def append(self, predictions: Tensor, y: np.ndarray):
        self.predictions.append(predictions)
        self.labels.append(y)

    @property
    def y_pred(self) -> np.ndarray:
        return np.concatenate([p.cpu().detach().numpy() for p in self.predictions])

    @property
    def y_true(self) -> np.ndarray:
        return np.concatenate(self.labels)


def calculate_metric(task_type: SupervisedTask, y_true: Series | np.ndarray, y_pred: Series | np.ndarray) -> float:
    if task_type == SupervisedTask.REGRESSION:
        score = r2_score(y_true=y_true, y_pred=y_pred)
    elif task_type == SupervisedTask.BINARY:
        score = roc_auc_score(y_true=y_true, y_score=y_pred)
    elif task_type == SupervisedTask.MULTICLASS:
        try:
            score = safe_roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr', average='macro')
        except ValueError as e:
            verbose_print(f"⚠️ Error calculating AUC. {y_true=}, {y_pred=}, {e=}")
            score = per_class_auc(y_true=y_true, y_pred=y_pred)
    else:
        raise ValueError(f"Unsupported data properties: {task_type}")
    return float(score)


def calculate_ks_metric(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculates the Kolmogorov-Smirnov statistic for binary classification.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class.

    Returns:
        The KS statistic.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred_proba, np.ndarray):
        y_pred_proba = np.array(y_pred_proba)

    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 1:
        y_pred_proba = y_pred_proba.ravel()
        
    if not (np.all(np.isin(y_true, [0, 1]))):
        raise ValueError("y_true must contain only binary labels (0 or 1).")
    if not (y_true.shape == y_pred_proba.shape):
        # This can happen if y_pred_proba is (n_samples, 2) for binary case from some models.
        # Assuming y_pred_proba[:, 1] is the prob for positive class as per roc_auc_score convention.
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        else:
            raise ValueError(f"y_true ({y_true.shape}) and y_pred_proba ({y_pred_proba.shape}) must have the same shape or y_pred_proba be (n_samples, 2).")
    if not (y_true.shape == y_pred_proba.shape): # re-check after potential reshape
         raise ValueError(f"Shape mismatch after attempting to reconcile: y_true ({y_true.shape}) and y_pred_proba ({y_pred_proba.shape}).")


    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        # KS statistic is not well-defined if only one class is present
        verbose_print(f"⚠️ KS metric is not well-defined for single-class data. Unique labels: {unique_labels}. Returning 0.0")
        return 0.0

    class0_probs = y_pred_proba[y_true == 0]
    class1_probs = y_pred_proba[y_true == 1]

    if len(class0_probs) == 0 or len(class1_probs) == 0:
        # KS statistic is not well-defined if one class has no samples
        verbose_print(f"⚠️ KS metric is not well-defined if one class has no samples. Class0 count: {len(class0_probs)}, Class1 count: {len(class1_probs)}. Returning 0.0")
        return 0.0
        
    ks_statistic, _ = ks_2samp(class0_probs, class1_probs)
    return float(ks_statistic)


def per_class_auc(y_true, y_pred) -> float:
    present_classes = np.unique(y_true)
    aucs = {}
    for cls in present_classes:
        # Binary ground truth: 1 for the current class, 0 for others
        y_true_binary = (y_true == cls).astype(int)
        # Predicted probabilities for the current class
        y_pred_scores = y_pred[:, cls]
        try:
            auc = roc_auc_score(y_true_binary, y_pred_scores)
            aucs[cls] = auc
        except ValueError as e:
            verbose_print(f"⚠️ Error calculating AUC for class {cls}. {e=}, {y_true_binary=}, {y_pred_scores=}")
    macro_avg = float(np.mean(list(aucs.values())))
    return macro_avg