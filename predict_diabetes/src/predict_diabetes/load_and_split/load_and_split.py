import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load diabetes dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the diabetes data.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame with the dataset.
    """
    df = pd.read_csv(filepath)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing subsets.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset to be split.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Controls the shuffling applied before splitting.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing (train, test) DataFrames.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test