from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE


def remove_duplicates(df):
    """
    Remove exact duplicate rows from the dataset.
    """
    return df.drop_duplicates().copy()


def separate_features_target(df):
    """
    Separate the input features from the target column.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def split_data(X, y):
    """
    Split the dataset into stratified training and testing sets.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def scale_features(X_train, X_test):
    """
    Fit StandardScaler on training data and transform both datasets.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler