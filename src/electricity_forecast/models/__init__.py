"""Model training and inference module."""

from electricity_forecast.models.base import ForecastModel
from electricity_forecast.models.baselines import NaiveLast, SeasonalNaive

# Lazy imports for heavy deps (xgboost, torch) - use get_model_class() to avoid loading until needed
_MODEL_REGISTRY: dict[str, type] = {
    "naive": NaiveLast,
    "seasonal": SeasonalNaive,
}


def get_model_class(name: str) -> type:
    """Lazy-load model class by name. Avoids importing xgboost/torch until needed."""
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name]
    if name == "xgb":
        from electricity_forecast.models.xgb import XGBForecast
        return XGBForecast
    if name == "sarimax":
        from electricity_forecast.models.sarimax import SARIMAXForecast
        return SARIMAXForecast
    if name == "lstm":
        from electricity_forecast.models.lstm import LSTMForecast
        return LSTMForecast
    raise ValueError(f"Unknown model: {name}")


def __getattr__(name: str):
    """Lazy attr access for XGBForecast, etc."""
    _map = {"XGBForecast": "xgb", "SARIMAXForecast": "sarimax", "LSTMForecast": "lstm", "QuantileXGB": "calibrate"}
    if name in _map:
        key = _map[name]
        if key == "calibrate":
            from electricity_forecast.models.calibrate import QuantileXGB
            return QuantileXGB
        return get_model_class(key)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ForecastModel",
    "NaiveLast",
    "SeasonalNaive",
    "XGBForecast",
    "SARIMAXForecast",
    "LSTMForecast",
    "QuantileXGB",
    "get_model_class",
]
