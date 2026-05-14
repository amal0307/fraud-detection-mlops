"""Train XGBoost fraud detector and register it in MLflow."""
import os

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.training.features import FEATURE_COLS, engineer_features, get_feature_target

load_dotenv(override=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fraud_detection"
DATA_PATH = "data/raw/creditcard.csv"
REFERENCE_PATH = "data/reference/reference_data.pkl"
MODEL_NAME = "fraud_xgboost"


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ─── 1. Load and engineer features ──────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    X, y = get_feature_target(df)
    print(f"Loaded {len(df):,} rows. Fraud rate: {y.mean() * 100:.3f}%")

    # ─── 2. Stratified train/test split ─────────────────────────
    # `stratify=y` ensures both splits keep the same fraud ratio.
    # Without this, a random split could put all frauds in one side.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ─── 3. Compute class weight for imbalance ─────────────────
    # `scale_pos_weight` tells XGBoost "treat each fraud case as N times
    # more important than each legit case." Better than SMOTE for trees.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # ─── 4. Train with MLflow tracking ─────────────────────────
    with mlflow.start_run() as run:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",  # AUC of Precision-Recall curve
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "tree_method": "hist",  # Fastest training mode
            "early_stopping_rounds": 20,
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ─── 5. Evaluate ────────────────────────────────────────
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "auc_pr": average_precision_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        print("\nResults:")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")

        # ─── 6. Register model in MLflow registry ──────────────
        signature = mlflow.models.infer_signature(X_train, y_pred_proba)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        # ─── 7. Save reference data for drift detection ────────
        # We'll compare future data against this snapshot to detect drift.
        os.makedirs("data/reference", exist_ok=True)
        reference = X_train.copy()
        reference["target"] = y_train.values
        reference.to_pickle(REFERENCE_PATH)
        mlflow.log_artifact(REFERENCE_PATH)

        print(f"\nRun ID:           {run.info.run_id}")
        print(f"Model registered: {MODEL_NAME}")
        print(f"Reference saved:  {REFERENCE_PATH}")


if __name__ == "__main__":
    train()