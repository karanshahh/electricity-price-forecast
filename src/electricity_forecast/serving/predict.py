"""Load production model, validate input via pydantic, return forecasts."""

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from electricity_forecast.config import get_config
from electricity_forecast.models import get_model_class
from electricity_forecast.transforms.features import build_features


class PredictRequest(BaseModel):
    """Request schema for /predict."""

    timestamps: list[str] = Field(..., min_length=1, description="ISO8601 timestamps for forecast")
    temperature_2m: list[float] | None = Field(None, description="Optional hourly temp")
    latitude: float | None = Field(None, description="Lat for weather")
    longitude: float | None = Field(None, description="Lon for weather")
    include_intervals: bool = Field(False, description="Include prediction intervals")


class PredictResponse(BaseModel):
    """Response schema for /predict."""

    forecasts: list[float] = Field(..., description="Point forecasts")
    lower: list[float] | None = Field(None, description="Lower interval bounds")
    upper: list[float] | None = Field(None, description="Upper interval bounds")


class Predictor:
    """Wrapper that loads model and produces forecasts."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        cfg = get_config()
        p = model_path or Path(cfg["data"]["processed_dir"]) / "model.pt"
        p = Path(p)
        if not Path(p).exists():
            p = Path(cfg["data"]["interim_dir"]) / "model.pt"
        self.model_path = Path(p) if Path(p).exists() else None
        self.model: Any = None
        if self.model_path:
            XGBForecast = get_model_class("xgb")
            self.model = XGBForecast.load(self.model_path)

    def predict(self, request: PredictRequest) -> PredictResponse:
        if self.model is None:
            raise ValueError("No model loaded. Train and save model first.")
        if not request.timestamps:
            return PredictResponse(forecasts=[])

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(request.timestamps, utc=True),
                "lmp": 0.0,  # placeholder for feature building (model expects lmp)
            }
        )
        df = build_features(df, target_col="lmp", ts_col="datetime", weather_df=None)
        pred = self.model.predict(df)
        if isinstance(pred, pd.DataFrame):
            forecasts = (
                pred["point"].tolist() if "point" in pred.columns else pred.iloc[:, 0].tolist()
            )
            lower = pred.get("q10", pred.get("q1", None))
            upper = pred.get("q90", pred.get("q9", None))
            lower = lower.tolist() if lower is not None else None
            upper = upper.tolist() if upper is not None else None
        else:
            forecasts = pred.tolist()
            lower = upper = None

        return PredictResponse(
            forecasts=forecasts,
            lower=lower if request.include_intervals else None,
            upper=upper if request.include_intervals else None,
        )


def load_predictor(model_path: str | Path | None = None) -> Predictor:
    """Load predictor from MLflow or local path."""
    return Predictor(model_path=model_path)
