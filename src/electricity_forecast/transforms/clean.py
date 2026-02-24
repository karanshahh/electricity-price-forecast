"""Standardize timestamps, handle duplicates, missing values, outliers."""

from typing import Literal

import numpy as np
import pandas as pd

from electricity_forecast.config import get_config


def clean_lmp(
    df: pd.DataFrame,
    ts_col: str | None = None,
    value_col: str | None = None,
    duplicate_strategy: Literal["first", "last", "mean"] | None = None,
    missing_strategy: Literal["interpolate", "forward_fill", "drop"] | None = None,
    outlier_method: Literal["iqr", "zscore", "none"] | None = None,
    outlier_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Clean LMP time series: UTC timestamps, dedupe, fill/drop missing, clip outliers.
    Configurable via config.transform.clean.
    """
    cfg = get_config()
    clean_cfg = cfg.get("transform", {}).get("clean", {})
    dup = duplicate_strategy or clean_cfg.get("duplicate_strategy", "first")
    miss = missing_strategy or clean_cfg.get("missing_strategy", "interpolate")
    out_m = outlier_method or clean_cfg.get("outlier_method", "iqr")
    out_t = outlier_threshold if outlier_threshold is not None else clean_cfg.get("outlier_threshold", 3.0)

    df = df.copy()
    ts_col = ts_col or _find_col(df, ["datetime_begin", "datetime", "timestamp", "time"])
    value_col = value_col or _find_col(df, ["lmp", "total_lmp", "price", "value"])

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.sort_values(ts_col).reset_index(drop=True)
        if value_col and ts_col != value_col:
            df = df[[ts_col, value_col] + [c for c in df.columns if c not in (ts_col, value_col)]]

    if value_col:
        df = _dedupe(df, ts_col or 0, value_col, dup)
        df = _handle_missing(df, value_col, miss)
        df = _handle_outliers(df, value_col, out_m, out_t)

    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0] if len(df.columns) > 0 else None


def _dedupe(
    df: pd.DataFrame, ts_col: str | int, value_col: str, strategy: str
) -> pd.DataFrame:
    if ts_col is None or strategy == "none":
        return df
    duped = df.duplicated(subset=[ts_col], keep=False)
    if not duped.any():
        return df
    if strategy == "first":
        return df.drop_duplicates(subset=[ts_col], keep="first")
    if strategy == "last":
        return df.drop_duplicates(subset=[ts_col], keep="last")
    if strategy == "mean":
        return df.groupby(ts_col, as_index=False)[value_col].mean()
    return df


def _handle_missing(df: pd.DataFrame, value_col: str, strategy: str) -> pd.DataFrame:
    if df[value_col].notna().all():
        return df
    if strategy == "drop":
        return df.dropna(subset=[value_col])
    if strategy == "forward_fill":
        df[value_col] = df[value_col].ffill().bfill()
        return df
    df[value_col] = df[value_col].interpolate(method="linear").ffill().bfill()
    return df


def _handle_outliers(
    df: pd.DataFrame, value_col: str, method: str, threshold: float
) -> pd.DataFrame:
    if method == "none":
        return df
    s = df[value_col]
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
    else:
        mean, std = s.mean(), s.std()
        if std == 0:
            return df
        lo = mean - threshold * std
        hi = mean + threshold * std
    df[value_col] = s.clip(lo, hi)
    return df
