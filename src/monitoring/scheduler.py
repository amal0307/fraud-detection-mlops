"""Periodic drift monitoring job."""
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from src.api.db import engine, get_db
from src.monitoring.drift_detector import compute_drift
from src.monitoring.performance_monitor import compute_performance
from src.training.features import FEATURE_COLS

load_dotenv(override=True)

REFERENCE_PATH = "data/reference/reference_data.pkl"
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
CHECK_INTERVAL_SECONDS = int(os.getenv("DRIFT_CHECK_INTERVAL", "3600"))  # default 1h
LOOKBACK_HOURS = int(os.getenv("DRIFT_LOOKBACK_HOURS", "24"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_reference() -> pd.DataFrame:
    """Load the snapshot of training data saved during training."""
    return pd.read_pickle(REFERENCE_PATH)


def fetch_recent_transactions(hours: int) -> pd.DataFrame:
    """Pull transactions from the last N hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = """
        SELECT t.amount, t.v1, t.v2, t.v3, t.v4, t.v5, t.v6, t.v7, t.v8, t.v9, t.v10,
               t.v11, t.v12, t.v13, t.v14, t.v15, t.v16, t.v17, t.v18, t.v19, t.v20,
               t.v21, t.v22, t.v23, t.v24, t.v25, t.v26, t.v27, t.v28, t.true_label,
               p.fraud_probability, p.prediction
        FROM transactions t
        LEFT JOIN predictions p ON p.transaction_id = t.transaction_id
        WHERE t.timestamp >= %(since)s
    """
    return pd.read_sql(query, engine, params={"since": since})


def write_drift_results(drift_rows: list[dict]):
    """Persist drift metrics."""
    if not drift_rows:
        return
    with get_db() as db:
        for row in drift_rows:
            db.execute(
                text(
                    """
                    INSERT INTO drift_metrics
                        (feature_name, psi_score, ks_statistic, ks_pvalue, drift_detected)
                    VALUES (:feature_name, :psi_score, :ks_statistic, :ks_pvalue, :drift_detected)
                    """
                ),
                row,
            )


def write_performance(perf: dict, model_version: str, window_start, window_end, n: int):
    if not perf:
        return
    with get_db() as db:
        db.execute(
            text(
                """
                INSERT INTO model_performance
                    (model_version, window_start, window_end, precision_score,
                     recall_score, f1_score, auc_score, n_predictions)
                VALUES (:mv, :ws, :we, :prec, :rec, :f1, :auc, :n)
                """
            ),
            {
                "mv": model_version,
                "ws": window_start,
                "we": window_end,
                "prec": perf.get("precision"),
                "rec": perf.get("recall"),
                "f1": perf.get("f1"),
                "auc": perf.get("auc_pr"),
                "n": n,
            },
        )


def run_once():
    """Single drift + performance check. Triggers retraining if drift fires."""
    logger.info("Running drift check...")

    reference = load_reference().drop(columns=["target"], errors="ignore")
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=LOOKBACK_HOURS)
    current = fetch_recent_transactions(LOOKBACK_HOURS)

    if len(current) < 50:
        logger.info(f"Only {len(current)} recent transactions — skipping this cycle.")
        return

    logger.info(f"Comparing {len(current)} recent txns against {len(reference)} reference rows.")

    drift_rows = compute_drift(reference, current, FEATURE_COLS, PSI_THRESHOLD)
    write_drift_results(drift_rows)

    drifted = [r["feature_name"] for r in drift_rows if r["drift_detected"]]
    if drifted:
        logger.warning(f"DRIFT DETECTED on features: {drifted}")
        # Auto-trigger retraining
        try:
            from src.retraining.pipeline import retrain
            result = retrain(trigger_reason=f"drift:{','.join(drifted[:3])}")
            logger.info(f"Retraining result: {result['status']}")
        except Exception:
            logger.exception("Retraining failed.")
    else:
        logger.info("No drift detected.")

    labeled = current.dropna(subset=["true_label"])
    if len(labeled) >= 30:
        perf = compute_performance(
            labeled["true_label"].astype(int),
            labeled["fraud_probability"],
            labeled["prediction"].astype(int),
        )
        write_performance(perf, "current", window_start, window_end, len(labeled))
        logger.info(f"Perf metrics ({len(labeled)} labeled): {perf}")


def main():
    logger.info(f"Drift monitor starting. Interval: {CHECK_INTERVAL_SECONDS}s")
    while True:
        try:
            run_once()
        except Exception:
            logger.exception("Drift check failed; continuing.")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()