"""End-to-end workflow: data -> features -> split -> train -> evaluate.

    python -m src.pipeline                       # uses data/raw/emails.csv
    python -m src.pipeline --data my_emails.csv --tune
    python -m src.pipeline --synthetic 3000      # generate demo data first
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import config
from src.data_processing import prepare_dataset
from src.evaluate import run_evaluation
from src.generate_synthetic_data import generate
from src.build_showcase import build as build_showcase
from src.train import train_models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(config.RAW_DATA_PATH))
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--synthetic", type=int, metavar="N",
                    help="generate N synthetic emails and write them to --data first")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_path = Path(args.data)
    if args.synthetic:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate(args.synthetic, config.RANDOM_SEED).to_csv(data_path, index=False)

    X_train, X_test, y_train, y_test = prepare_dataset(data_path)
    models = train_models(X_train, y_train, tune=args.tune)
    run_evaluation(models, X_test, y_test)
    build_showcase()


if __name__ == "__main__":
    main()
