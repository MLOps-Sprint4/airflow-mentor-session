"""
Utilities for the Telco Churn Airflow pipeline.

Imported inside DAG task bodies (not at DAG parse time).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "logistic": {
        "class": LogisticRegression,
        "defaults": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "flavour": "sklearn",
    },
    "rf": {
        "class": RandomForestClassifier,
        "defaults": {"n_estimators": 100, "max_depth": 10},
        "flavour": "sklearn",
    },
    "xgb": {
        "class": XGBClassifier,
        "defaults": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
        },
        "flavour": "xgboost",
    },
}


def build_model(model_type: str, user_params: dict):
    """Instantiate a model, merging user params over defaults."""
    entry = MODEL_REGISTRY[model_type]
    params = {**entry["defaults"], **user_params}
    params["random_state"] = params.get("random_state", 42)
    model = entry["class"](**params)
    return model, params, entry["flavour"]


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_plots(model, X_test, y_test):
    """Create and log confusion matrix and ROC curve plots to MLflow."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)
