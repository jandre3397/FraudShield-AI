from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def build_dense_model(input_features):
    """
    Build the FraudShield-AI Dense neural network.
    """
    model = Sequential([
        keras.Input(shape=(input_features,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model