"""Tracks model performance over time using labeled transactions."""
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_performance(y_true, y_proba, y_pred) -> dict:
    """Standard classification metrics for a window of predictions."""
    if len(y_true) < 10:
        return {}

    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
        "auc_pr": float(average_precision_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }