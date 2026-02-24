"""Time-based train/val/test splits and rolling backtest folds."""

from dataclasses import dataclass

import pandas as pd

from electricity_forecast.config import get_config


@dataclass
class SplitResult:
    """Train/val/test DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def time_split(
    df: pd.DataFrame,
    ts_col: str = "datetime",
    train_days: int | None = None,
    val_days: int | None = None,
    test_days: int | None = None,
) -> SplitResult:
    """
    Split by date boundaries: train (oldest), val, test (newest).
    Uses last train_days+val_days+test_days from df.
    """
    cfg = get_config()
    bt = cfg.get("backtest", {})
    train_days = train_days or bt.get("train_days", 365)
    val_days = val_days or bt.get("val_days", 30)
    test_days = test_days or bt.get("test_days", 30)

    df = df.copy()
    if ts_col not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col).reset_index(drop=True)

    dates = df[ts_col].dt.date.unique()
    n = len(dates)
    if n < train_days + val_days + test_days:
        train_days = max(1, n // 3)
        val_days = max(1, (n - train_days) // 2)
        test_days = n - train_days - val_days

    t_end = dates[-(val_days + test_days)]
    v_end = dates[-test_days]
    train = df[df[ts_col].dt.date < t_end]
    val = df[(df[ts_col].dt.date >= t_end) & (df[ts_col].dt.date < v_end)]
    test = df[df[ts_col].dt.date >= v_end]
    return SplitResult(train=train, val=val, test=test)


def rolling_folds(
    df: pd.DataFrame,
    ts_col: str = "datetime",
    train_days: int = 365,
    test_days: int = 7,
    step_days: int = 7,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Rolling-origin folds: (train, test) pairs.
    Each fold steps forward by step_days.
    """
    cfg = get_config()
    bt = cfg.get("backtest", {})
    train_days = bt.get("train_days", train_days)
    test_days = bt.get("test_days", test_days)
    step_days = bt.get("step_days", step_days)

    df = df.copy()
    if ts_col not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col).reset_index(drop=True)
    dates = sorted(df[ts_col].dt.date.unique())
    n_dates = len(dates)
    if n_dates < train_days + test_days:
        train_days = max(1, n_dates // 2)
        test_days = max(1, min(test_days, n_dates - train_days))
        step_days = max(1, step_days)

    folds = []
    i = train_days
    while i + test_days <= len(dates):
        train_dates = set(dates[i - train_days : i])
        test_dates = set(dates[i : i + test_days])
        train = df[df[ts_col].dt.date.isin(train_dates)]
        test = df[df[ts_col].dt.date.isin(test_dates)]
        if len(train) > 0 and len(test) > 0:
            folds.append((train, test))
        i += step_days
    return folds
