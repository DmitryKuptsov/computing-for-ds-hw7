import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Tuple


def evaluate_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """
    Evaluate the model on the test dataset and print accuracy.

    Parameters
    ----------
    model : LogisticRegression
        Trained logistic regression model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True target values for test set.

    Returns
    -------
    float
        Accuracy score on the test set.
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {acc:.3f}")
    return acc


def add_prediction_probabilities(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add predicted probabilities as a new column 'predictions' 
    to both training and testing DataFrames.

    Parameters
    ----------
    model : LogisticRegression
        Trained logistic regression model.
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    train : pd.DataFrame
        Training DataFrame to modify.
    test : pd.DataFrame
        Testing DataFrame to modify.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Updated train and test DataFrames with 'predictions' column added.
    """
    train["predictions"] = model.predict_proba(X_train)[:, 1]
    test["predictions"] = model.predict_proba(X_test)[:, 1]
    return train, test
