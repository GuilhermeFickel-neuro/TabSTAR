from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Predictions:
    score: float
    predictions: np.ndarray
    labels: np.ndarray
    ks_score: Optional[float] = None