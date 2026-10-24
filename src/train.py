"""Train the three models on the training split and persist them with joblib.

    python -m src.train            # default hyper-parameters
    python -m src.train --tune     # stratified CV grid search optimising recall
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.data_processing import load_split
from src.models import build_models

logger = logging.getLogger(__name__)


def train_models(X_train: pd.DataFrame, y_train: pd.Series, tune: bool = False,
                 save: bool = True) -> Dict[str, Pipeline]:
    trained: Dict[str, Pipeline] = {}
    for name, pipe in build_models().items():
        if tune:
            cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
            search = GridSearchCV(pipe, config.PARAM_GRIDS[name],
                                  scoring=config.PRIMARY_METRIC, cv=cv, n_jobs=-1)
            search.fit(X_train, y_train)
            trained[name] = search.best_estimator_
            logger.info("%-20s best params=%s (CV %s=%.4f)", name, search.best_params_,
                        config.PRIMARY_METRIC, search.best_score_)
        else:
            pipe.fit(X_train, y_train)
            trained[name] = pipe
            logger.info("%-20s trained (default params)", name)

    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, model in trained.items():
            joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
        logger.info("Saved models to %s", config.MODELS_DIR)
    return trained


def load_models() -> Dict[str, Pipeline]:
    models = {}
    for name in build_models():
        path = config.MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python -m src.train` first.")
        models[name] = joblib.load(path)
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune", action="store_true", help="CV grid search scored on recall")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    X_train, y_train = load_split(config.TRAIN_PATH)
    train_models(X_train, y_train, tune=args.tune)


if __name__ == "__main__":
    main()
