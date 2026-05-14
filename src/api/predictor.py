"""Loads model from MLflow registry and produces predictions + SHAP explanations."""
import mlflow
import mlflow.xgboost
import pandas as pd
import shap

from src.training.features import FEATURE_COLS


class FraudPredictor:
    def __init__(self, model_name: str = "fraud_xgboost"):
        self.model_name = model_name
        self.model = None
        self.model_version: str = ""
        self.explainer: shap.TreeExplainer | None = None
        self.load()

    def load(self):
        """Pull latest registered version from MLflow."""
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{self.model_name}'")
        if not versions:
            raise RuntimeError(f"No registered versions for '{self.model_name}'")

        # Sort by version number descending, take the latest
        latest = max(versions, key=lambda v: int(v.version))
        self.model_version = latest.version

        model_uri = f"models:/{self.model_name}/{latest.version}"
        self.model = mlflow.xgboost.load_model(model_uri)

        # TreeExplainer is the fast SHAP implementation specific to tree models
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, features: dict) -> dict:
        """Score one transaction and explain it."""
        df = pd.DataFrame([features])[FEATURE_COLS]

        # Probability of fraud
        proba = float(self.model.predict_proba(df)[0, 1])
        prediction = int(proba >= 0.5)

        # SHAP values for this one prediction (one row, 29 features)
        shap_values = self.explainer.shap_values(df)[0]
        impacts = dict(zip(FEATURE_COLS, shap_values.tolist()))

        # Top 3 by absolute impact — these are the "why was this flagged" features
        top_3 = dict(
            sorted(impacts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        )

        return {
            "prediction": prediction,
            "fraud_probability": proba,
            "model_version": self.model_version,
            "shap_top_features": top_3,
        }