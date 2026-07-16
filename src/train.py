from src.config import (
    BATCH_SIZE,
    EPOCHS,
    FINAL_MODEL_PATH,
    MODERATE_CLASS_WEIGHTS,
    MODEL_DIR,
    VALIDATION_SPLIT,
)
from src.data_loader import load_data
from src.model import build_dense_model
from src.preprocessing import (
    remove_duplicates,
    scale_features,
    separate_features_target,
    split_data,
)


def train_final_model():
    """
    Train and save the final moderate class-weighted model.
    """
    df = load_data()
    df = remove_duplicates(df)

    X, y = separate_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test,
    )

    model = build_dense_model(X_train_scaled.shape[1])

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        class_weight=MODERATE_CLASS_WEIGHTS,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)

    return {
        "model": model,
        "history": history,
        "scaler": scaler,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
    }


if __name__ == "__main__":
    train_final_model()