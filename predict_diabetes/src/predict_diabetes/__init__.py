__version__ = "0.1.0"

"""
predict_diabetes
================
A modular library for diabetes mellitus data analysis and prediction.
"""

from .load_and_split import load_data, split_data
from .preprocess import preprocess_data
from .train import train_model
from .predict import evaluate_model, add_prediction_probabilities
from .metrics import compute_roc_auc

__all__ = [
    "load_data",
    "split_data",
    "preprocess_data",
    "train_model",
    "evaluate_model",
    "add_prediction_probabilities",
    "compute_roc_auc",
]
