"""Data drift: PSI and KS test for numeric features."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    e_hist, _ = np.histogram(expected, breakpoints)
    a_hist, _ = np.histogram(actual, breakpoints)
    e_pct = (e_hist + 1e-10) / (e_hist.sum() + 1e-10)
    a_pct = (a_hist + 1e-10) / (a_hist.sum() + 1e-10)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test statistic and p-value."""
    res = stats.ks_2samp(x, y)
    return float(res.statistic), float(res.pvalue)


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Compute PSI and KS for numeric features."""
    feature_cols = feature_cols or [
        c
        for c in reference.columns
        if reference[c].dtype in ("float64", "int64") and c not in ("target",)
    ]
    results = {}
    for col in feature_cols:
        if col not in current.columns:
            continue
        ref = reference[col].dropna().values
        cur = current[col].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue
        results[col] = {
            "psi": psi(ref, cur),
            "ks_statistic": ks_statistic(ref, cur)[0],
            "ks_pvalue": ks_statistic(ref, cur)[1],
        }
    return results


def compute_prediction_drift(
    ref_pred: np.ndarray,
    cur_pred: np.ndarray,
) -> dict[str, float]:
    """Drift metrics for predictions."""
    return {
        "psi": psi(ref_pred, cur_pred),
        "ks_statistic": ks_statistic(ref_pred, cur_pred)[0],
        "ks_pvalue": ks_statistic(ref_pred, cur_pred)[1],
    }
