import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Tuple


def compute_roc_auc(
    y_train: pd.Series,
    y_test: pd.Series,
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[float, float]:
    """
    Compute and print ROC-AUC scores for training and testing datasets.

    Parameters
    ----------
    y_train : pd.Series
        True target values for training set.
    y_test : pd.Series
        True target values for test set.
    train : pd.DataFrame
        Training DataFrame with a 'predictions' column (predicted probabilities).
    test : pd.DataFrame
        Testing DataFrame with a 'predictions' column (predicted probabilities).

    Returns
    -------
    Tuple[float, float]
        (train_auc, test_auc) scores.
    """
    # Compute AUCs
    train_auc = roc_auc_score(y_train, train["predictions"])
    test_auc = roc_auc_score(y_test, test["predictions"])

    # Display results
    print(f"Train ROC-AUC: {train_auc:.3f}")
    print(f"Test  ROC-AUC: {test_auc:.3f}")

    return train_auc, test_auc
