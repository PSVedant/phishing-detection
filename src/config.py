"""Central configuration: paths, seeds, column names and model hyper-parameters."""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "emails.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# ----------------------------------------------------------------------------
# Dataset schema: one row per email, raw RFC-822 text (headers + body) + label
# ----------------------------------------------------------------------------
TEXT_COLUMN = "email_text"
LABEL_COLUMN = "label"          # phishing = 1 (positive class), legitimate = 0
POSITIVE_LABEL = 1

# ----------------------------------------------------------------------------
# Experiment settings
# ----------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.30                # stratified 70/30 split
CV_FOLDS = 5                    # only used with --tune
PRIMARY_METRIC = "recall"       # FN (missed phishing) is costlier than FP

# ----------------------------------------------------------------------------
# Default model hyper-parameters
# ----------------------------------------------------------------------------
LOGREG_PARAMS = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
GNB_PARAMS = {"var_smoothing": 1e-9}
TREE_PARAMS = {
    "criterion": "gini",
    "max_depth": 5,             # depth control (prevents overfitting)
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "ccp_alpha": 0.0,
}

# Grids searched when --tune is given (scored on recall, see PRIMARY_METRIC)
PARAM_GRIDS = {
    "logistic_regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "gaussian_nb": {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
    "decision_tree": {
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__min_samples_leaf": [5, 10, 20],
    },
}
