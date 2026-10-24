"""
Evaluate trained models on the held-out test set (phishing = positive class).

Recall is the PRIMARY metric: a missed phishing email (false negative) is far
costlier than a false alarm (false positive). Models are ranked by recall, with
F1 and then precision as tie-breakers.

    python -m src.evaluate
"""
from __future__ import annotations

import json
import logging
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)
from sklearn.tree import export_text, plot_tree

from src import config
from src.data_processing import load_split
from src.feature_extraction import FEATURE_NAMES
from src.models import DISPLAY_NAMES
from src.train import load_models

logger = logging.getLogger(__name__)
POS = config.POSITIVE_LABEL


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    # labels=[0, 1] -> matrix layout [[TN, FP], [FN, TP]] with phishing positive
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_test, y_pred, pos_label=POS, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=POS, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=POS, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def evaluate_all(models: Dict[str, object], X_test, y_test) -> pd.DataFrame:
    rows = {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "model"
    return df.sort_values([config.PRIMARY_METRIC, "f1", "precision"], ascending=False)


def plot_confusion_matrices(results: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results.iterrows()):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], dtype=int)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{DISPLAY_NAMES.get(name, name)}\nRecall={r.recall:.3f}")
        ax.set_xticks([0, 1], ["Legit (0)", "Phish (1)"])
        ax.set_yticks([0, 1], ["Legit (0)", "Phish (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(results: pd.DataFrame, path) -> None:
    """Grouped bars: recall (primary), precision, F1 per model."""
    labels = [DISPLAY_NAMES.get(n, n) for n in results.index]
    x, w = np.arange(len(labels)), 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (m, c) in enumerate([("recall", "#000000"), ("precision", "#979797"), ("f1", "#d1ffca")]):
        bars = ax.bar(x + (i - 1) * w, results[m], w, label=m.capitalize(), color=c, edgecolor="black")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.8, 1.05)
    ax.set_title("Model comparison (phishing = positive class)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(models, X_test, y_test, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{DISPLAY_NAMES.get(name, name)} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(models, path) -> None:
    lr = pd.Series(models["logistic_regression"].named_steps["clf"].coef_[0], index=FEATURE_NAMES)
    dt = pd.Series(models["decision_tree"].named_steps["clf"].feature_importances_, index=FEATURE_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lr.sort_values().plot.barh(ax=axes[0], color="black")
    axes[0].set_title("Logistic Regression coefficients (scaled)")
    dt.sort_values().plot.barh(ax=axes[1], color="#979797")
    axes[1].set_title("Decision Tree importance (Gini)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decision_tree(models, path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(models["decision_tree"].named_steps["clf"], feature_names=FEATURE_NAMES,
              class_names=["legit", "phish"], filled=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title("Decision Tree (top 3 levels)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(results: pd.DataFrame, models: Dict[str, object]) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.REPORTS_DIR / "metrics.csv")
    (config.REPORTS_DIR / "metrics.json").write_text(
        json.dumps(results.to_dict(orient="index"), indent=2))
    plot_confusion_matrices(results, config.REPORTS_DIR / "confusion_matrices.png")

    lines = ["# Phishing detection - evaluation report", "",
             "Positive class = phishing (1). **Primary metric: Recall** "
             "(false negatives = missed phishing are the costliest error).", "",
             "| Rank | Model | Recall | Precision | F1 | TN | FP | FN | TP |",
             "|---|---|---|---|---|---|---|---|---|"]
    for rank, (name, r) in enumerate(results.iterrows(), 1):
        lines.append(f"| {rank} | {DISPLAY_NAMES.get(name, name)} | **{r.recall:.4f}** | "
                     f"{r.precision:.4f} | {r.f1:.4f} | {int(r.TN)} | {int(r.FP)} | "
                     f"{int(r.FN)} | {int(r.TP)} |")
    best = results.index[0]
    lines += ["", f"**Best by recall:** {DISPLAY_NAMES.get(best, best)}", "",
              "## Interpretability", ""]

    lr = models["logistic_regression"]
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=FEATURE_NAMES).sort_values(
        key=abs, ascending=False)
    lines += ["Logistic regression coefficients (standardised features):", "", "```",
              coefs.round(3).to_string(), "```", "",
              "Decision tree rules:", "", "```",
              export_text(models["decision_tree"].named_steps["clf"], feature_names=FEATURE_NAMES),
              "```"]
    (config.REPORTS_DIR / "report.md").write_text("\n".join(lines))


def print_summary(results: pd.DataFrame) -> None:
    show = results[["recall", "precision", "f1", "TN", "FP", "FN", "TP"]].copy()
    show.index = [DISPLAY_NAMES.get(i, i) for i in show.index]
    print("\n=== Test-set results (phishing = positive class; ranked by RECALL) ===")
    for col in ("recall", "precision", "f1"):
        show[col] = show[col].map("{:.4f}".format)
    for col in ("TN", "FP", "FN", "TP"):
        show[col] = show[col].astype(int)
    print(show.to_string())
    print()


def run_evaluation(models, X_test, y_test) -> pd.DataFrame:
    results = evaluate_all(models, X_test, y_test)
    write_report(results, models)
    plot_metric_comparison(results, config.REPORTS_DIR / "metric_comparison.png")
    plot_roc_curves(models, X_test, y_test, config.REPORTS_DIR / "roc_curves.png")
    plot_feature_importance(models, config.REPORTS_DIR / "feature_importance.png")
    plot_decision_tree(models, config.REPORTS_DIR / "decision_tree.png")
    print_summary(results)
    logger.info("Reports written to %s", config.REPORTS_DIR)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    X_test, y_test = load_split(config.TEST_PATH)
    run_evaluation(load_models(), X_test, y_test)


if __name__ == "__main__":
    main()
