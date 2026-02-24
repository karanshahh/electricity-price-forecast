"""Naive baselines: last value, seasonal naive (same hour previous day/week)."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from electricity_forecast.models.base import ForecastModel

FEATURE_COLS = ["lag_1", "lag_24", "lag_168"]


class NaiveLast(ForecastModel):
    """Predict last observed value."""

    def __init__(self) -> None:
        self.last_value_: float = 0.0

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, **kwargs: Any
    ) -> "NaiveLast":
        target = train_df["target"] if "target" in train_df.columns else train_df.iloc[:, -1]
        self.last_value_ = float(target.iloc[-1])
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        n = len(df)
        return pd.Series([self.last_value_] * n, index=df.index)

    def save(self, path: str | Path) -> None:
        joblib.dump({"last_value": self.last_value_}, path)

    @classmethod
    def load(cls, path: str | Path) -> "NaiveLast":
        m = cls()
        data = joblib.load(path)
        m.last_value_ = data["last_value"]
        return m


class SeasonalNaive(ForecastModel):
    """Predict same hour from previous day (lag_24) or week (lag_168)."""

    def __init__(self, period: int = 24) -> None:
        self.period = period
        self.fallback_: float = 0.0

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, **kwargs: Any
    ) -> "SeasonalNaive":
        lag_col = f"lag_{self.period}"
        if lag_col in train_df.columns:
            valid = train_df[lag_col].dropna()
            self.fallback_ = float(valid.mean()) if len(valid) > 0 else 0.0
        else:
            target = train_df["target"] if "target" in train_df.columns else train_df.iloc[:, -1]
            self.fallback_ = float(target.mean())
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        lag_col = f"lag_{self.period}"
        if lag_col in df.columns:
            return df[lag_col].fillna(self.fallback_)
        return pd.Series([self.fallback_] * len(df), index=df.index)

    def save(self, path: str | Path) -> None:
        joblib.dump({"period": self.period, "fallback": self.fallback_}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SeasonalNaive":
        data = joblib.load(path)
        m = cls(period=data["period"])
        m.fallback_ = data["fallback"]
        return m
