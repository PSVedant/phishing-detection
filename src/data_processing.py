"""Load the raw email CSV, extract features and create the stratified 70/30 split.

    python -m src.data_processing --data data/raw/emails.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.feature_extraction import FEATURE_NAMES, extract_feature_frame

logger = logging.getLogger(__name__)


def load_raw_dataset(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read and validate the raw dataset (columns: email_text, label)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Provide a CSV with columns "
            f"'{config.TEXT_COLUMN}' and '{config.LABEL_COLUMN}', or run "
            "`python -m src.generate_synthetic_data` to create demo data."
        )
    df = pd.read_csv(path)
    missing = {config.TEXT_COLUMN, config.LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    before = len(df)
    df = df.dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN]).copy()
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(int)
    if not set(df[config.LABEL_COLUMN].unique()) <= {0, 1}:
        raise ValueError("Labels must be binary: phishing=1, legitimate=0")
    if len(df) < before:
        logger.warning("Dropped %d rows with missing values", before - len(df))

    counts = df[config.LABEL_COLUMN].value_counts().to_dict()
    logger.info("Loaded %d emails; class counts: %s", len(df), counts)
    if len(counts) == 2 and abs(counts[0] - counts[1]) / len(df) > 0.1:
        logger.warning("Dataset is not balanced (>10%% gap) - interpret metrics with care")
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = extract_feature_frame(df[config.TEXT_COLUMN].tolist())
    y = df[config.LABEL_COLUMN].reset_index(drop=True)
    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series):
    """Stratified 70/30 split with a fixed seed (reproducible)."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )


def prepare_dataset(raw_path: Path = config.RAW_DATA_PATH, save: bool = True):
    """Full data stage: raw CSV -> features -> stratified split (optionally saved)."""
    df = load_raw_dataset(raw_path)
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.assign(**{config.LABEL_COLUMN: y_train.values}).to_csv(config.TRAIN_PATH, index=False)
        X_test.assign(**{config.LABEL_COLUMN: y_test.values}).to_csv(config.TEST_PATH, index=False)
        logger.info("Saved splits: train=%d, test=%d -> %s",
                    len(X_train), len(X_test), config.PROCESSED_DIR)
    return X_train, X_test, y_train, y_test


def load_split(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a processed split saved by prepare_dataset()."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data_processing` first.")
    df = pd.read_csv(path)
    return df[FEATURE_NAMES], df[config.LABEL_COLUMN]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(config.RAW_DATA_PATH))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_dataset(Path(args.data))


if __name__ == "__main__":
    main()
