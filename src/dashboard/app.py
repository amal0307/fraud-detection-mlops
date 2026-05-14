"""Streamlit dashboard for the fraud detection system."""
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv(override=True)

# ─── DB connection (same as the API) ──────────────────────────────
DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB')}"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

st.set_page_config(
    page_title="Fraud Detection — Live Ops",
    page_icon="🛡️",
    layout="wide",
)

# Auto-refresh every 10 seconds
REFRESH_SECONDS = 10
st.markdown(
    f"""<meta http-equiv="refresh" content="{REFRESH_SECONDS}">""",
    unsafe_allow_html=True,
)


# ─── Data fetchers (cached briefly) ───────────────────────────────
@st.cache_data(ttl=5)
def fetch_summary(hours: int = 24) -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = """
        SELECT
            COUNT(*) AS n_txns,
            COUNT(*) FILTER (WHERE p.prediction = 1) AS n_flagged,
            AVG(p.latency_ms) AS avg_latency,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY p.latency_ms) AS p99_latency,
            MAX(p.model_version) AS model_version
        FROM transactions t
        JOIN predictions p ON p.transaction_id = t.transaction_id
        WHERE t.timestamp >= %(since)s
    """
    return pd.read_sql(q, engine, params={"since": since}).iloc[0].to_dict()


@st.cache_data(ttl=5)
def fetch_throughput(hours: int = 24) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = """
        SELECT date_trunc('minute', t.timestamp) AS minute,
               COUNT(*) AS n_txns,
               COUNT(*) FILTER (WHERE p.prediction = 1) AS n_flagged
        FROM transactions t
        JOIN predictions p ON p.transaction_id = t.transaction_id
        WHERE t.timestamp >= %(since)s
        GROUP BY 1
        ORDER BY 1
    """
    return pd.read_sql(q, engine, params={"since": since})


@st.cache_data(ttl=5)
def fetch_proba_distribution(hours: int = 24) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = """
        SELECT p.fraud_probability
        FROM predictions p
        JOIN transactions t ON t.transaction_id = p.transaction_id
        WHERE t.timestamp >= %(since)s
    """
    return pd.read_sql(q, engine, params={"since": since})


@st.cache_data(ttl=10)
def fetch_drift_latest() -> pd.DataFrame:
    q = """
        SELECT DISTINCT ON (feature_name)
               feature_name, psi_score, ks_statistic, ks_pvalue,
               drift_detected, computed_at
        FROM drift_metrics
        ORDER BY feature_name, computed_at DESC
    """
    return pd.read_sql(q, engine)


@st.cache_data(ttl=10)
def fetch_drift_timeline(feature: str) -> pd.DataFrame:
    q = """
        SELECT computed_at, psi_score
        FROM drift_metrics
        WHERE feature_name = %(feature)s
        ORDER BY computed_at
    """
    return pd.read_sql(q, engine, params={"feature": feature})


@st.cache_data(ttl=10)
def fetch_retraining_events() -> pd.DataFrame:
    q = """
        SELECT triggered_at, trigger_reason, old_model_version,
               new_model_version, new_model_promoted,
               old_auc, new_auc, notes
        FROM retraining_events
        ORDER BY triggered_at DESC
        LIMIT 25
    """
    return pd.read_sql(q, engine)


@st.cache_data(ttl=10)
def fetch_recent_predictions(limit: int = 20) -> pd.DataFrame:
    q = """
        SELECT p.predicted_at, t.amount,
               p.fraud_probability, p.prediction, p.latency_ms,
               p.shap_top_features::text AS shap_top_features
        FROM predictions p
        JOIN transactions t ON t.transaction_id = p.transaction_id
        ORDER BY p.predicted_at DESC
        LIMIT %(limit)s
    """
    return pd.read_sql(q, engine, params={"limit": limit})


# ─── UI ───────────────────────────────────────────────────────────
st.title("🛡️  Fraud Detection — Live Operations")
st.caption(
    f"Auto-refreshing every {REFRESH_SECONDS}s · "
    f"Last update: {datetime.now().strftime('%H:%M:%S')}"
)

hours = st.sidebar.slider("Lookback window (hours)", 1, 168, 24)

# ─── Top-row KPIs ─────────────────────────────────────────────────
summary = fetch_summary(hours)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions", f"{int(summary['n_txns'] or 0):,}")
c2.metric("Fraud Flagged", f"{int(summary['n_flagged'] or 0):,}")
fraud_rate = (
    (summary["n_flagged"] or 0) / summary["n_txns"] * 100
    if summary["n_txns"] else 0
)
c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
c4.metric("Avg Latency", f"{summary['avg_latency'] or 0:.1f} ms")
c5.metric("p99 Latency", f"{summary['p99_latency'] or 0:.1f} ms")

st.divider()

# ─── Throughput chart ─────────────────────────────────────────────
st.subheader("Transaction throughput")
throughput = fetch_throughput(hours)
if not throughput.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=throughput["minute"], y=throughput["n_txns"],
        name="Total", mode="lines", line=dict(color="#3b82f6", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=throughput["minute"], y=throughput["n_flagged"],
        name="Flagged as fraud", mode="lines", line=dict(color="#ef4444", width=2)
    ))
    fig.update_layout(
        height=300, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="", yaxis_title="Transactions per minute",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No transactions yet. Start the producer + consumer to see data flowing.")

# ─── Prediction probability histogram ─────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("Fraud probability distribution")
    proba_df = fetch_proba_distribution(hours)
    if not proba_df.empty:
        fig = px.histogram(
            proba_df, x="fraud_probability", nbins=50,
            color_discrete_sequence=["#6366f1"],
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Threshold")
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="P(fraud)", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet.")

with right:
    st.subheader("Drift status per feature")
    drift = fetch_drift_latest()
    if not drift.empty:
        drift["status"] = drift["drift_detected"].map({True: "⚠️ DRIFT", False: "✓ OK"})
        drift_display = drift[["feature_name", "psi_score", "status"]].copy()
        drift_display["psi_score"] = drift_display["psi_score"].round(4)
        drift_display.columns = ["Feature", "PSI", "Status"]
        st.dataframe(
            drift_display.sort_values("PSI", ascending=False),
            use_container_width=True, height=300, hide_index=True,
        )
    else:
        st.info("No drift checks yet. Run the monitor.")

st.divider()

# ─── Retraining audit log ─────────────────────────────────────────
st.subheader("Retraining events (audit trail)")
events = fetch_retraining_events()
if not events.empty:
    events["decision"] = events["new_model_promoted"].map(
        {True: "✅ PROMOTED", False: "❌ REJECTED"}
    )
    events["auc_change"] = (events["new_auc"] - events["old_auc"]).round(4)
    display = events[[
        "triggered_at", "trigger_reason", "decision",
        "old_model_version", "new_model_version",
        "old_auc", "new_auc", "auc_change", "notes",
    ]]
    display.columns = [
        "When", "Trigger", "Decision",
        "Old v.", "New v.", "Old AUC-PR", "New AUC-PR", "Δ", "Notes",
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.info("No retraining events yet.")

# ─── Recent predictions with SHAP ─────────────────────────────────
st.subheader("Recent predictions (with SHAP top features)")
recent = fetch_recent_predictions(20)
if not recent.empty:
    def format_shap(s):
        try:
            d = json.loads(s)
            return " · ".join(f"{k}: {v:+.2f}" for k, v in d.items())
        except Exception:
            return s

    recent["shap_top_features"] = recent["shap_top_features"].apply(format_shap)
    recent["fraud_probability"] = recent["fraud_probability"].round(4)
    recent["latency_ms"] = recent["latency_ms"].round(1)
    recent["flagged"] = recent["prediction"].map({1: "🚩", 0: ""})
    recent = recent[[
        "predicted_at", "amount", "fraud_probability",
        "flagged", "latency_ms", "shap_top_features",
    ]]
    recent.columns = ["Time", "Amount", "P(fraud)", "", "Latency (ms)", "Top SHAP features"]
    st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.info("No predictions yet.")