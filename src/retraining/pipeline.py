"""Auto-retraining pipeline. Trains challenger, evaluates vs. champion, promotes if better."""
import logging
import os
from datetime import datetime, timedelta, timezone

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
from sqlalchemy import text

from src.api.db import engine, get_db
from src.retraining.promote import should_promote
from src.training.features import FEATURE_COLS, engineer_features

load_dotenv(override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

MODEL_NAME = "fraud_xgboost"
EXPERIMENT_NAME = "fraud_detection"
RETRAIN_LOOKBACK_DAYS = int(os.getenv("RETRAIN_LOOKBACK_DAYS", "30"))
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "5000"))
MIN_FRAUD_SAMPLES = int(os.getenv("MIN_FRAUD_SAMPLES", "20"))

logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test) -> dict:
    """Same evaluation function used in training."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "auc_pr": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


def fetch_labeled_data(days: int) -> pd.DataFrame:
    """Pull recent transactions that have a true_label populated."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = """
        SELECT amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
               v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
               v21, v22, v23, v24, v25, v26, v27, v28, true_label
        FROM transactions
        WHERE true_label IS NOT NULL AND timestamp >= %(since)s
    """
    return pd.read_sql(query, engine, params={"since": since})


def load_current_champion():
    """Load the currently registered Production model (or latest if none staged)."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        return None, None

    # Prefer the version tagged "Production"; fall back to latest.
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    chosen = prod_versions[0] if prod_versions else max(versions, key=lambda v: int(v.version))
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{chosen.version}")
    return model, chosen.version


def log_retraining_event(reason, old_version, new_version, promoted, old_auc, new_auc, notes):
    with get_db() as db:
        db.execute(
            text("""
                INSERT INTO retraining_events
                    (trigger_reason, old_model_version, new_model_version,
                     new_model_promoted, old_auc, new_auc, notes)
                VALUES (:reason, :old, :new, :promoted, :old_auc, :new_auc, :notes)
            """),
            {
                "reason": reason,
                "old": old_version,
                "new": new_version,
                "promoted": promoted,
                "old_auc": old_auc,
                "new_auc": new_auc,
                "notes": notes,
            },
        )


def retrain(trigger_reason: str = "drift_detected") -> dict:
    """Full retraining pipeline. Returns summary dict."""
    logger.info(f"=== Retraining triggered: {trigger_reason} ===")

    # ─── 1. Pull recent labeled data ─────────────────────────────
    df = fetch_labeled_data(RETRAIN_LOOKBACK_DAYS)
    logger.info(f"Fetched {len(df)} labeled rows from last {RETRAIN_LOOKBACK_DAYS} days")

    if len(df) < MIN_TRAINING_SAMPLES:
        msg = f"Insufficient data: {len(df)} < {MIN_TRAINING_SAMPLES}"
        logger.warning(msg)
        log_retraining_event(trigger_reason, None, None, False, None, None, msg)
        return {"status": "skipped", "reason": msg}

    fraud_count = df["true_label"].sum()
    if fraud_count < MIN_FRAUD_SAMPLES:
        msg = f"Insufficient fraud examples: {fraud_count} < {MIN_FRAUD_SAMPLES}"
        logger.warning(msg)
        log_retraining_event(trigger_reason, None, None, False, None, None, msg)
        return {"status": "skipped", "reason": msg}

    # ─── 2. Prep features ────────────────────────────────────────
    df = df.rename(columns={"true_label": "class"})
    df = engineer_features(df)
    X = df[FEATURE_COLS]
    y = df["class"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ─── 3. Load champion, evaluate on holdout ───────────────────
    champion, champion_version = load_current_champion()
    if champion is None:
        logger.warning("No existing champion — promoting challenger by default.")
        champion_metrics = {"auc_pr": 0.0, "recall": 0.0}
    else:
        champion_metrics = evaluate(champion, X_holdout, y_holdout)
        logger.info(f"Champion v{champion_version} metrics: {champion_metrics}")

    # ─── 4. Train challenger ─────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"challenger_{datetime.now():%Y%m%d_%H%M%S}") as run:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "tree_method": "hist",
            "early_stopping_rounds": 20,
        }
        mlflow.log_params(params)
        mlflow.log_param("trigger_reason", trigger_reason)
        mlflow.log_param("training_rows", len(X_train))

        challenger = xgb.XGBClassifier(**params)
        challenger.fit(X_train, y_train, eval_set=[(X_holdout, y_holdout)], verbose=False)

        challenger_metrics = evaluate(challenger, X_holdout, y_holdout)
        mlflow.log_metrics(challenger_metrics)
        logger.info(f"Challenger metrics: {challenger_metrics}")

        # ─── 5. Champion/challenger decision ─────────────────────
        promote, reason = should_promote(champion_metrics, challenger_metrics)
        mlflow.log_param("promoted", promote)
        mlflow.log_param("decision_reason", reason)
        logger.info(f"Decision: {'PROMOTE' if promote else 'REJECT'} — {reason}")

        # ─── 6. Register the challenger regardless (audit trail) ─
        signature = mlflow.models.infer_signature(X_train, challenger.predict_proba(X_train)[:, 1])
        result = mlflow.xgboost.log_model(
            challenger,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )
        new_version = result.registered_model_version

        # ─── 7. If promoting, transition stages in registry ──────
        if promote:
            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_version, stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"✓ Challenger v{new_version} promoted to Production.")

        # ─── 8. Audit log ────────────────────────────────────────
        log_retraining_event(
            trigger_reason,
            champion_version,
            new_version,
            promote,
            champion_metrics.get("auc_pr"),
            challenger_metrics.get("auc_pr"),
            reason,
        )

        return {
            "status": "promoted" if promote else "rejected",
            "old_version": champion_version,
            "new_version": new_version,
            "champion_metrics": champion_metrics,
            "challenger_metrics": challenger_metrics,
            "reason": reason,
        }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = retrain(trigger_reason="manual")
    print("\nResult:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()