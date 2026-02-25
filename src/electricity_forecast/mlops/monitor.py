"""Monitoring: compute drift and write summary JSON."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.mlops.drift import compute_feature_drift, compute_prediction_drift


def run_drift_monitor(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    ref_predictions: pd.Series | None = None,
    cur_predictions: pd.Series | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute feature and prediction drift, write summary JSON.
    """
    feature_drift = compute_feature_drift(reference_df, current_df)
    summary = {"feature_drift": feature_drift}

    if ref_predictions is not None and cur_predictions is not None:
        pred_drift = compute_prediction_drift(ref_predictions.values, cur_predictions.values)
        summary["prediction_drift"] = pred_drift

    if output_path is None:
        cfg = get_config()
        output_path = Path(cfg["data"]["interim_dir"]) / "drift_summary.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
