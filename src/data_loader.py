import pandas as pd

from src.config import RAW_DATA_PATH


def load_data(file_path=RAW_DATA_PATH):
    """
    Load the credit card transaction dataset.

    Returns:
        pd.DataFrame: Loaded transaction data.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Place creditcard.csv inside data/raw/."
        )

    return pd.read_csv(file_path)