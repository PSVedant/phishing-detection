"""Model factory. Every model is an sklearn Pipeline whose last step is named 'clf'."""
from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config


def build_models(seed: int = config.RANDOM_SEED) -> Dict[str, Pipeline]:
    """Return the three untrained models keyed by name."""
    return {
        # Linear baseline. Scaling is essential: body_length (~1e3) would otherwise
        # dominate binary flags. predict_proba gives calibrated-ish probabilities.
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **config.LOGREG_PARAMS)),
        ]),
        # Probabilistic baseline (Gaussian likelihood per feature). Scale-invariant.
        "gaussian_nb": Pipeline([
            ("clf", GaussianNB(**config.GNB_PARAMS)),
        ]),
        # Rule-based model: Gini splits, depth + leaf-size controls against overfitting.
        "decision_tree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=seed, **config.TREE_PARAMS)),
        ]),
    }


DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "gaussian_nb": "Gaussian Naive Bayes",
    "decision_tree": "Decision Tree",
}
