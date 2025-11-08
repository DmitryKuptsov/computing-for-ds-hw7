import pandas as pd
from typing import Tuple, List


def drop_missing_demographics(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows with missing values in demographic columns: age, gender, ethnicity.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset.
    test : pd.DataFrame
        Testing dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames with NaNs dropped in key demographic columns.
    """
    cols_to_check = ["age", "gender", "ethnicity"]
    train = train.dropna(subset=cols_to_check)
    test = test.dropna(subset=cols_to_check)
    return train, test


def fill_missing_with_mean(
    train: pd.DataFrame, test: pd.DataFrame, cols: List[str] = ["height", "weight"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing values in numeric columns with the mean (computed from training data).

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset.
    test : pd.DataFrame
        Testing dataset.
    cols : list of str, default=["height", "weight"]
        Columns to fill with their respective means.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames with missing numeric values filled.
    """
    for col in cols:
        if col in train.columns:
            mean_val = train[col].mean()
            train[col] = train[col].fillna(mean_val)
            test[col] = test[col].fillna(mean_val)
    return train, test


def encode_ethnicity(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode the 'ethnicity' column using pandas get_dummies, 
    ensuring both train and test have the same columns.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset.
    test : pd.DataFrame
        Testing dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames with one-hot encoded ethnicity columns.
    """
    train = pd.get_dummies(train, columns=["ethnicity"], drop_first=True)
    test = pd.get_dummies(test, columns=["ethnicity"], drop_first=True)
    test = test.reindex(columns=train.columns, fill_value=0)
    return train, test


def encode_gender_binary(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode 'gender' as binary: M → 1, F → 0 (case-insensitive).

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset.
    test : pd.DataFrame
        Testing dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames with gender encoded as binary.
    """
    train["gender"] = train["gender"].apply(lambda x: 1 if str(x).upper().startswith("M") else 0)
    test["gender"] = test["gender"].apply(lambda x: 1 if str(x).upper().startswith("M") else 0)
    return train, test


def preprocess_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply full preprocessing pipeline:
    1. Drop NaN in demographic columns
    2. Fill numeric NaN with mean
    3. One-hot encode ethnicity
    4. Encode gender as binary

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset.
    test : pd.DataFrame
        Testing dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Cleaned and preprocessed train and test DataFrames.
    """
    train, test = drop_missing_demographics(train, test)
    train, test = fill_missing_with_mean(train, test)
    train, test = encode_ethnicity(train, test)
    train, test = encode_gender_binary(train, test)
    return train, test
