"""Drift detection: PSI and KS test on feature distributions."""
import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """
    PSI compares two distributions across bins.

    PSI = sum( (actual% - expected%) * ln(actual% / expected%) )

    Lower = more similar. >0.2 = significant drift.
    """
    # Bin the data using quantiles of the *expected* (training) data
    # so bin edges are stable regardless of what current data looks like.
    breakpoints = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to percentages, add tiny epsilon to avoid log(0)
    eps = 1e-6
    expected_pct = expected_counts / max(len(expected), 1) + eps
    actual_pct = actual_counts / max(len(actual), 1) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov 2-sample test. Returns (statistic, p_value)."""
    statistic, p_value = stats.ks_2samp(expected, actual)
    return float(statistic), float(p_value)


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = 0.2,
) -> list[dict]:
    """Compute PSI + KS for every feature. Return one row per feature."""
    results = []
    for col in feature_cols:
        if col not in current.columns or len(current[col].dropna()) < 30:
            # Skip if too few samples — drift stats are unreliable
            continue

        ref_values = reference[col].dropna().values
        cur_values = current[col].dropna().values

        psi = population_stability_index(ref_values, cur_values)
        ks_stat, ks_p = ks_test(ref_values, cur_values)

        results.append({
            "feature_name": col,
            "psi_score": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "drift_detected": psi > psi_threshold,
        })

    return results