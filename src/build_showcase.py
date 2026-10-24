"""Inject real results (reports/metrics.json + model coefficients) into the HTML showcase.

    python -m src.build_showcase     # then open showcase/index.html
"""
from __future__ import annotations

import base64
import json
import logging

import pandas as pd

from src import config
from src.feature_extraction import FEATURE_NAMES
from src.train import load_models

logger = logging.getLogger(__name__)
SHOW_DIR = config.ROOT_DIR / "showcase"
IMAGES = [  # (file, title)
    ("confusion_matrices.png", "Confusion matrices"),
    ("metric_comparison.png", "Recall / precision / F1"),
    ("roc_curves.png", "ROC curves"),
    ("feature_importance.png", "Feature importance"),
    ("decision_tree.png", "Decision tree rules"),
]
GROUPS = {
    "URL": ["num_links", "num_ip_urls", "num_http_links", "all_links_https"],
    "Header": ["sender_replyto_mismatch"],
    "Link": ["num_mismatched_links", "has_mismatched_link"],
    "Content": ["kw_verify", "kw_suspended", "kw_password", "kw_suspicious_total", "urgency_score"],
    "Structure": ["body_length", "has_attachment"],
}


def build() -> None:
    m = pd.read_csv(config.REPORTS_DIR / "metrics.csv")
    metrics = [{**r, **{k: int(r[k]) for k in ("TN", "FP", "FN", "TP")}}
               for r in m.to_dict("records")]
    coef = pd.Series(load_models()["logistic_regression"].named_steps["clf"].coef_[0],
                     index=FEATURE_NAMES).sort_values(key=abs, ascending=False).head(6)
    images = []
    for fname, title in IMAGES:
        f = config.REPORTS_DIR / fname
        if f.exists():  # embedded as base64 -> index.html is fully self-contained
            images.append([title, "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()])
    data = {"images": images, "metrics": metrics, "groups": GROUPS, "seed": config.RANDOM_SEED,
            "test_size": int(sum(metrics[0][k] for k in ("TN", "FP", "FN", "TP"))),
            "coefs": [[k, float(v)] for k, v in coef.items()]}
    html = (SHOW_DIR / "template.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data))
    (SHOW_DIR / "index.html").write_text(html, encoding="utf-8")
    logger.info("Showcase written to %s", SHOW_DIR / "index.html")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
