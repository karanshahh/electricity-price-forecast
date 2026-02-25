"""Quantile regression / conformal prediction for prediction intervals."""

from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from electricity_forecast.models.base import ForecastModel
from electricity_forecast.models.xgb import _feature_cols

EXCLUDE_COLS = {"target", "lmp", "datetime", "datetime_begin", "timestamp"}


class QuantileXGB(ForecastModel):
    """
    XGBoost quantile regression for prediction intervals.
    Fits separate models for lower/upper quantiles.
    """

    def __init__(
        self,
        quantiles: tuple[float, float] = (0.1, 0.9),
        **xgb_kwargs: Any,
    ) -> None:
        self.quantiles = quantiles
        self.xgb_kwargs = xgb_kwargs
        self.models_: dict[float, xgb.XGBRegressor] = {}
        self.feature_names_: list[str] = []

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, **kwargs: Any
    ) -> "QuantileXGB":
        feats = _feature_cols(train_df)
        self.feature_names_ = feats
        X = train_df[feats].fillna(0)
        y = train_df["target"]

        for q in self.quantiles:
            m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **self.xgb_kwargs)
            m.fit(X, y)
            self.models_[q] = m
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        feats = [c for c in self.feature_names_ if c in df.columns]
        X = df[feats].fillna(0) if feats else df
        out = pd.DataFrame(index=df.index)
        for q, m in self.models_.items():
            out[f"q{int(q * 100)}"] = m.predict(X)
        out["point"] = out[[c for c in out.columns if c.startswith("q")]].mean(axis=1)
        return out

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "models": self.models_,
                "feature_names": self.feature_names_,
                "quantiles": self.quantiles,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "QuantileXGB":
        import joblib

        data = joblib.load(path)
        m = cls(quantiles=tuple(data["quantiles"]))
        m.models_ = data["models"]
        m.feature_names_ = data["feature_names"]
        return m
