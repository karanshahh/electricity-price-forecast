"""Base model interface: fit, predict, save, load."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ForecastModel(ABC):
    """Common interface for all forecast models."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, **kwargs: Any) -> "ForecastModel":
        """Fit on training data. Optional validation set for early stopping."""
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, **kwargs: Any) -> pd.Series | pd.DataFrame:
        """Return point forecasts. May return Series or DataFrame with interval columns."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "ForecastModel":
        """Load model from disk."""
        ...
