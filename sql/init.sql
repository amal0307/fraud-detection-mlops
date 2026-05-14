-- Table 1: Every transaction we score
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    SERIAL PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    amount            NUMERIC(12, 2) NOT NULL,
    -- The 28 anonymized features from the Kaggle dataset
    v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT, v5 FLOAT, v6 FLOAT, v7 FLOAT,
    v8 FLOAT, v9 FLOAT, v10 FLOAT, v11 FLOAT, v12 FLOAT, v13 FLOAT, v14 FLOAT,
    v15 FLOAT, v16 FLOAT, v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT, v21 FLOAT,
    v22 FLOAT, v23 FLOAT, v24 FLOAT, v25 FLOAT, v26 FLOAT, v27 FLOAT, v28 FLOAT,
    true_label        INTEGER  -- 1 if known fraud, 0 if known legit, NULL if unknown
);

-- Table 2: Every prediction the model made
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     SERIAL PRIMARY KEY,
    transaction_id    INTEGER REFERENCES transactions(transaction_id),
    model_version     VARCHAR(50) NOT NULL,
    fraud_probability FLOAT NOT NULL,
    prediction        INTEGER NOT NULL,  -- 1 = flagged as fraud, 0 = legit
    latency_ms        FLOAT,
    shap_top_features JSONB,  -- {"V14": -3.2, "V12": -2.1, "V10": -1.8}
    predicted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table 3: Drift monitoring results (logged hourly)
CREATE TABLE IF NOT EXISTS drift_metrics (
    metric_id         SERIAL PRIMARY KEY,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feature_name      VARCHAR(50) NOT NULL,
    psi_score         FLOAT,
    ks_statistic      FLOAT,
    ks_pvalue         FLOAT,
    drift_detected    BOOLEAN NOT NULL DEFAULT FALSE
);

-- Table 4: Model performance over time
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id    SERIAL PRIMARY KEY,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version     VARCHAR(50) NOT NULL,
    window_start      TIMESTAMPTZ NOT NULL,
    window_end        TIMESTAMPTZ NOT NULL,
    precision_score   FLOAT,
    recall_score      FLOAT,
    f1_score          FLOAT,
    auc_score         FLOAT,
    n_predictions     INTEGER
);

-- Table 5: Retraining audit log
CREATE TABLE IF NOT EXISTS retraining_events (
    event_id          SERIAL PRIMARY KEY,
    triggered_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_reason    VARCHAR(100),  -- 'drift_detected', 'scheduled', 'manual'
    old_model_version VARCHAR(50),
    new_model_version VARCHAR(50),
    new_model_promoted BOOLEAN NOT NULL DEFAULT FALSE,
    old_auc           FLOAT,
    new_auc           FLOAT,
    notes             TEXT
);

-- Indexes to make queries fast
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions(predicted_at);
CREATE INDEX IF NOT EXISTS idx_drift_computed_at ON drift_metrics(computed_at);
CREATE INDEX IF NOT EXISTS idx_perf_computed_at ON model_performance(computed_at);