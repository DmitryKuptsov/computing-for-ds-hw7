import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Tuple


def train_model(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train a logistic regression model to predict diabetes mellitus.

    Steps:
    1. Select relevant features (numeric + ethnicity one-hot encoded)
    2. Split into X_train, y_train, X_test, y_test
    3. Fit logistic regression model

    Parameters
    ----------
    train : pd.DataFrame
        Preprocessed training dataset.
    test : pd.DataFrame
        Preprocessed testing dataset.

    Returns
    -------
    Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Trained model, X_train, X_test, y_train, y_test.
    """
    # Features
    features = [
        "age", "height", "weight",
        "aids", "cirrhosis", "hepatic_failure",
        "immunosuppression", "leukemia", "lymphoma",
        "solid_tumor_with_metastasis"
    ] + [c for c in train.columns if c.startswith("ethnicity_")]

    # Split into features and target
    X_train = train[features]
    y_train = train["diabetes_mellitus"]
    X_test = test[features]
    y_test = test["diabetes_mellitus"]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
