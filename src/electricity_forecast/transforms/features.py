"""Time-series feature engineering: lags, rolling, calendar, weather."""


import numpy as np
import pandas as pd

from electricity_forecast.config import get_config

US_HOLIDAYS = {
    "01-01",
    "07-04",
    "12-25",
    "11-28",
    "11-29",
    "12-24",
    "12-31",
    "01-02",
    "09-01",
    "11-11",
    "12-26",
}


def build_features(
    df: pd.DataFrame,
    target_col: str = "lmp",
    ts_col: str = "datetime",
    weather_df: pd.DataFrame | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    rolling_agg: list[str] | None = None,
    rolling_quantiles: list[float] | None = None,
    include_weather: bool = True,
) -> pd.DataFrame:
    """
    Create modeling table: lags, rolling stats, calendar, price change, optional weather.
    No future leakage: only past values used.
    """
    cfg = get_config()
    feat_cfg = cfg.get("transform", {}).get("features", {})
    lags = lags or feat_cfg.get("lags", [1, 2, 24, 48, 168])
    rolling_windows = rolling_windows or feat_cfg.get("rolling_windows", [6, 24, 168])
    rolling_agg = rolling_agg or feat_cfg.get("rolling_agg", ["mean", "std", "min", "max"])
    rolling_quantiles = rolling_quantiles or feat_cfg.get("rolling_quantiles", [0.25, 0.5, 0.75])
    include_weather = (
        include_weather if weather_df is not None else feat_cfg.get("include_weather", False)
    )

    out = df.copy()
    if ts_col not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        ts_col = out.columns[0]
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
    out = out.sort_values(ts_col).reset_index(drop=True)

    price = out[target_col] if target_col in out.columns else out.iloc[:, 1]
    out["target"] = price

    for lag in lags:
        out[f"lag_{lag}"] = price.shift(lag)

    _agg_map = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}
    for w in rolling_windows:
        shifted = price.shift(1)
        for agg in rolling_agg:
            fn = _agg_map.get(agg)
            if fn:
                out[f"roll_{w}_{agg}"] = shifted.rolling(w, min_periods=1).apply(fn, raw=True)
        for q in rolling_quantiles:
            out[f"roll_{w}_q{int(q * 100)}"] = price.shift(1).rolling(w, min_periods=1).quantile(q)

    out["hour"] = out[ts_col].dt.hour
    out["dow"] = out[ts_col].dt.dayofweek
    out["month"] = out[ts_col].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_holiday"] = out[ts_col].dt.strftime("%m-%d").isin(US_HOLIDAYS).astype(int)

    out["price_change_1"] = price.shift(1).diff(1)
    out["price_change_24"] = price.shift(1).diff(24)
    out["volatility_24"] = price.shift(1).rolling(24, min_periods=1).std()

    if include_weather and weather_df is not None:
        out = _merge_weather(out, weather_df, ts_col)

    return out.dropna(subset=[f"lag_{lags[0]}"]).reset_index(drop=True)


def _merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Align weather by hour; no future info."""
    w = weather_df.copy()
    if isinstance(w.index, pd.DatetimeIndex):
        w = w.reset_index()
        w = w.rename(columns={w.columns[0]: "wt"})
    else:
        w["wt"] = pd.to_datetime(w.get("datetime", w.iloc[:, 0]), utc=True)
    w["wt"] = w["wt"].dt.tz_localize(None) if w["wt"].dt.tz else w["wt"]
    df["_merge_key"] = pd.to_datetime(df[ts_col], utc=True).dt.tz_localize(None)
    w["_merge_key"] = w["wt"]
    merge_cols = [
        c
        for c in w.columns
        if c not in ("wt", "_merge_key") and w[c].dtype in (np.floating, np.int64)
    ]
    out = df.merge(
        w[["_merge_key"] + merge_cols], on="_merge_key", how="left", suffixes=("", "_weather")
    )
    out = out.drop(columns=["_merge_key"], errors="ignore")
    return out
