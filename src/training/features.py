"""Feature engineering for fraud detection."""
import numpy as np
import pandas as pd

# The Kaggle dataset has 28 PCA-anonymized features (V1-V28) + Amount + Time + Class
FEATURE_COLS = [f"v{i}" for i in range(1, 29)] + ["amount"]
TARGET_COL = "class"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform raw transaction data."""
    df = df.copy()

    # Normalize column names to lowercase (matches our Postgres schema)
    df.columns = [c.lower() for c in df.columns]

    # Log-transform Amount: highly right-skewed (most txns small, few huge)
    # log1p handles zeros gracefully: log(1 + x)
    df["amount"] = np.log1p(df["amount"])

    # Drop Time column — it's seconds since first txn, not useful for our use case.
    # In production we'll use real timestamps from the streaming layer.
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    return df


def get_feature_target(df: pd.DataFrame):
    """Split DataFrame into features (X) and target (y)."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y