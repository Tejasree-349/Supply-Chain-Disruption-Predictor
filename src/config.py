from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset paths
RAW_DATA_PATH = (
    RAW_DATA_DIR / "DataCoSupplyChainDataset.csv"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "supply_chain_processed.csv"
)

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

# Application directory
APP_DIR = PROJECT_ROOT / "app"

# Test directory
TEST_DIR = PROJECT_ROOT / "tests"


# ============================================================
# TARGET CONFIGURATION
# ============================================================

# Binary classification target
TARGET_COLUMN = "Late_delivery_risk"

# Target values:
# 0 = No late-delivery risk according to the dataset label
# 1 = Late-delivery risk according to the dataset label

TARGET_VALUES = [0, 1]


# ============================================================
# DATA SPLITTING CONFIGURATION
# ============================================================


RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20


LEAKAGE_COLUMNS = [
    "Days_for_shipping_(real)",
    "Delivery_Status",
    "Late_delivery_risk",
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Classification models
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]


HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.40


# ============================================================
# OUTPUT FILE PATHS
# ============================================================

EVALUATION_REPORT_PATH = (
    REPORT_DIR / "model_evaluation.csv"
)


TEST_DATA_PATH = (
    REPORT_DIR / "test_data.csv"
)


FEATURE_IMPORTANCE_PATH = (
    REPORT_DIR / "permutation_importance.csv"
)


PREDICTIONS_PATH = (
    REPORT_DIR / "predictions.csv"
)


# ============================================================
# MODEL ARTIFACT PATHS
# ============================================================


LOGISTIC_REGRESSION_MODEL_PATH = (
    MODEL_DIR / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    MODEL_DIR / "random_forest.joblib"
)

GRADIENT_BOOSTING_MODEL_PATH = (
    MODEL_DIR / "gradient_boosting.joblib"
)


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

def create_project_directories():
    """
    Create the required directories if they do not exist.
    This function does not create or modify the dataset.
    """

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        PLOT_DIR,
        REPORT_DIR,
        APP_DIR,
        TEST_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config():
    """
    Validate configuration values and required paths.
    """

    if not 0 < TEST_SIZE < 1:
        raise ValueError(
            "TEST_SIZE must be between 0 and 1."
        )

    if not 0 < VALIDATION_SIZE < 1:
        raise ValueError(
            "VALIDATION_SIZE must be between 0 and 1."
        )

    if TARGET_COLUMN not in LEAKAGE_COLUMNS:
        raise ValueError(
            "The target column must be excluded from input features."
        )

    if not 0 <= MEDIUM_RISK_THRESHOLD <= 1:
        raise ValueError(
            "MEDIUM_RISK_THRESHOLD must be between 0 and 1."
        )

    if not 0 <= HIGH_RISK_THRESHOLD <= 1:
        raise ValueError(
            "HIGH_RISK_THRESHOLD must be between 0 and 1."
        )

    if MEDIUM_RISK_THRESHOLD > HIGH_RISK_THRESHOLD:
        raise ValueError(
            "The medium-risk threshold cannot exceed the high-risk threshold."
        )

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_project_directories()
    validate_config()

    print("=" * 60)
    print("SUPPLY CHAIN PROJECT CONFIGURATION")
    print("=" * 60)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Raw dataset: {RAW_DATA_PATH}")
    print(f"Processed dataset: {PROCESSED_DATA_PATH}")
    print(f"Models directory: {MODEL_DIR}")
    print(f"Reports directory: {REPORT_DIR}")
    print(f"Plots directory: {PLOT_DIR}")

    print(f"\nTarget column: {TARGET_COLUMN}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Validation size: {VALIDATION_SIZE}")

    print("\nConfiguration validation successful.")

    if RAW_DATA_PATH.exists():
        print("\nDataset file found.")
    else:
        print("\nWARNING: Dataset file not found.")
        print("Expected location:")
        print(RAW_DATA_PATH)

    print("\n" + "=" * 60)