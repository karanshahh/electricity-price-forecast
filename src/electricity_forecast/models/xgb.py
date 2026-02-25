"""XGBoost regression with early stopping and feature importance."""

from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from electricity_forecast.config import get_config
from electricity_forecast.models.base import ForecastModel

EXCLUDE_COLS = {"target", "lmp", "datetime", "datetime_begin", "timestamp"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in ("float64", "int64")]


class XGBForecast(ForecastModel):
    """XGBoost regressor for electricity price forecasting."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> None:
        cfg = get_config()
        xgb_cfg = cfg.get("model", {}).get("xgb", {})
        self.n_estimators = n_estimators or xgb_cfg.get("n_estimators", 500)
        self.max_depth = max_depth or xgb_cfg.get("max_depth", 6)
        self.learning_rate = learning_rate or xgb_cfg.get("learning_rate", 0.05)
        self.early_stopping = early_stopping_rounds or xgb_cfg.get("early_stopping_rounds", 50)
        self.model_: xgb.XGBRegressor | None = None
        self.feature_names_: list[str] = []

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, **kwargs: Any
    ) -> "XGBForecast":
        feats = _feature_cols(train_df)
        self.feature_names_ = feats
        X_train = train_df[feats].fillna(0)
        y_train = train_df["target"]

        eval_set = None
        early_stop = self.early_stopping
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[feats].fillna(0)
            y_val = val_df["target"]
            eval_set = [(X_val, y_val)]
        else:
            early_stop = None  # No early stopping without validation set

        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            early_stopping_rounds=early_stop,
            **kwargs,
        )
        self.model_.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series:
        X = df.reindex(columns=self.feature_names_).fillna(0)
        return pd.Series(self.model_.predict(X), index=df.index)

    def feature_importance(self) -> pd.Series:
        """Return feature importance (gain) if model is fitted."""
        if self.model_ is None:
            return pd.Series()
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {
                "model": self.model_,
                "feature_names": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "XGBForecast":
        import joblib

        data = joblib.load(path)
        m = cls()
        m.model_ = data["model"]
        m.feature_names_ = data["feature_names"]
        return m
