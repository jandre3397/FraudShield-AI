from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODEL_DIR / "fraudshield_model.keras"

# Random seed
RANDOM_STATE = 42

# Training configuration
TEST_SIZE = 0.20
VALIDATION_SPLIT = 0.20
EPOCHS = 10
BATCH_SIZE = 256

# Final model class weights
MODERATE_CLASS_WEIGHTS = {
    0: 1.0,
    1: 10.0,
}