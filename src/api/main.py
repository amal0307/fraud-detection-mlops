"""Fraud detection API — scores transactions and persists results."""
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from src.api.db import get_db
from src.api.predictor import FraudPredictor
from src.api.schemas import PredictionResponse, TransactionRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

predictor: FraudPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup; loads the model into memory."""
    global predictor
    logger.info("Loading model from MLflow registry...")
    predictor = FraudPredictor()
    logger.info(f"Model v{predictor.model_version} loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Used by Docker, Kubernetes, load balancers to check if we're alive."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_version": predictor.model_version if predictor else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.perf_counter()

    features = txn.model_dump()

    try:
        result = predictor.predict(features)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000

    # Persist the transaction and prediction. Both writes happen in one txn.
    with get_db() as db:
        row = db.execute(
            text(
                """
                INSERT INTO transactions (
                    amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                    v21, v22, v23, v24, v25, v26, v27, v28
                ) VALUES (
                    :amount, :v1, :v2, :v3, :v4, :v5, :v6, :v7, :v8, :v9, :v10,
                    :v11, :v12, :v13, :v14, :v15, :v16, :v17, :v18, :v19, :v20,
                    :v21, :v22, :v23, :v24, :v25, :v26, :v27, :v28
                )
                RETURNING transaction_id
                """
            ),
            features,
        ).fetchone()
        transaction_id = row[0]

        db.execute(
            text(
                """
                INSERT INTO predictions (
                    transaction_id, model_version, fraud_probability,
                    prediction, latency_ms, shap_top_features
                ) VALUES (
                    :tid, :mv, :proba, :pred, :lat, :shap
                )
                """
            ),
            {
                "tid": transaction_id,
                "mv": result["model_version"],
                "proba": result["fraud_probability"],
                "pred": result["prediction"],
                "lat": latency_ms,
                "shap": json.dumps(result["shap_top_features"]),
            },
        )

    return PredictionResponse(
        transaction_id=transaction_id,
        prediction=result["prediction"],
        fraud_probability=result["fraud_probability"],
        model_version=result["model_version"],
        latency_ms=round(latency_ms, 2),
        shap_top_features=result["shap_top_features"],
    )