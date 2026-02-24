"""SARIMAX baseline with exogenous regressors support."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from electricity_forecast.config import get_config
from electricity_forecast.models.base import ForecastModel

EXCLUDE_COLS = {"target", "datetime", "datetime_begin", "timestamp"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in ("float64", "int64")]


class SARIMAXForecast(ForecastModel):
    """SARIMAX model for electricity price forecasting."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
        **kwargs: Any,
    ) -> None:
        cfg = get_config()
        sar_cfg = cfg.get("model", {}).get("sarimax", {})
        self.order = order or tuple(sar_cfg.get("order", [1, 0, 1]))
        self.seasonal_order = seasonal_order or tuple(sar_cfg.get("seasonal_order", [1, 0, 1, 24]))
        self.model_: SARIMAXResults | None = None
        self.feature_names_: list[str] = []
        self.last_endog_: np.ndarray | None = None

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, **kwargs: Any
    ) -> "SARIMAXForecast":
        y = train_df["target"].values
        exog = None
        feats = _feature_cols(train_df)
        if feats:
            self.feature_names_ = feats
            exog = train_df[feats].fillna(0).values

        mod = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_ = mod.fit(disp=False)
        self.last_endog_ = y
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        steps = len(df)
        exog = None
        if self.feature_names_ and all(c in df.columns for c in self.feature_names_):
            exog = df[self.feature_names_].fillna(0).values
        fcast = self.model_.forecast(steps=steps, exog=exog)
        return pd.Series(fcast, index=df.index)

    def save(self, path: str | Path) -> None:
        joblib.dump({
            "model": self.model_,
            "feature_names": self.feature_names_,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "SARIMAXForecast":
        data = joblib.load(path)
        m = cls()
        m.model_ = data["model"]
        m.feature_names_ = data.get("feature_names", [])
        return m
