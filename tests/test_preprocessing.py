import pandas as pd

from src.preprocessing import (
    remove_duplicates,
    separate_features_target,
)


def test_remove_duplicates():
    df = pd.DataFrame({
        "V1": [1.0, 1.0, 2.0],
        "Class": [0, 0, 1],
    })

    cleaned_df = remove_duplicates(df)

    assert len(cleaned_df) == 2


def test_separate_features_target():
    df = pd.DataFrame({
        "V1": [1.0, 2.0],
        "Amount": [10.0, 20.0],
        "Class": [0, 1],
    })

    X, y = separate_features_target(df)

    assert "Class" not in X.columns
    assert len(X) == len(y)