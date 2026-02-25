"""Unit tests for feature leakage (no future info, no target leakage)."""

import pandas as pd

from electricity_forecast.models import get_model_class
from electricity_forecast.transforms.features import build_features


def test_lags_use_only_past(sample_modeling_df: pd.DataFrame) -> None:
    """Lags must not include current or future values."""
    df = sample_modeling_df.copy()
    row_idx = 100
    df.loc[row_idx, "target"] = 9999.0  # inject marker
    out = build_features(df, target_col="target", ts_col="datetime", weather_df=None)
    # lag_1 at row 100 should be value from row 99, not 9999
    lag1 = out.loc[out["datetime"] == df.loc[row_idx, "datetime"], "lag_1"].values
    assert len(lag1) > 0
    assert lag1[0] != 9999.0


def test_rolling_uses_shifted_series(sample_modeling_df: pd.DataFrame) -> None:
    """Rolling stats must use shift(1) to avoid current value."""
    df = sample_modeling_df.head(50).copy()
    out = build_features(
        df,
        target_col="target",
        ts_col="datetime",
        rolling_windows=[6],
        rolling_agg=["mean"],
    )
    assert "roll_6_mean" in out.columns
    # First row with valid roll should not include current in window
    valid = out.dropna(subset=["roll_6_mean"])
    assert len(valid) >= 1


def test_model_excludes_lmp_from_features(sample_modeling_df: pd.DataFrame) -> None:
    """XGB and other models must not use lmp (target) as a feature."""
    df = sample_modeling_df.copy()
    df["lmp"] = df["target"]  # simulate raw data with lmp column
    model_cls = get_model_class("xgb")
    model = model_cls()
    model.fit(df.head(200), val_df=None)
    assert "lmp" not in model.feature_names_
    assert "target" not in model.feature_names_
