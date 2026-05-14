"""Request/response schemas for the prediction API."""
from typing import Dict

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """One transaction to score. Matches the 29 features the model expects."""
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    v1: float; v2: float; v3: float; v4: float; v5: float; v6: float; v7: float
    v8: float; v9: float; v10: float; v11: float; v12: float; v13: float; v14: float
    v15: float; v16: float; v17: float; v18: float; v19: float; v20: float; v21: float
    v22: float; v23: float; v24: float; v25: float; v26: float; v27: float; v28: float


class PredictionResponse(BaseModel):
    """What we return to the caller."""
    model_config = {"protected_namespaces": ()}
    transaction_id: int
    prediction: int = Field(..., description="1 = fraud flagged, 0 = legitimate")
    fraud_probability: float = Field(..., ge=0, le=1)
    model_version: str
    latency_ms: float
    shap_top_features: Dict[str, float] = Field(
        ..., description="Top 3 features driving this decision, by SHAP value"
    )