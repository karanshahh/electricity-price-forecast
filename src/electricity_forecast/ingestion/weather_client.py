"""Open-Meteo API client for hourly weather data."""

import time

import pandas as pd
import requests

from electricity_forecast.config import get_config
from electricity_forecast.ingestion.schemas import WeatherFetchParams

HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,"
    "apparent_temperature,precipitation,cloud_cover,"
    "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
    "surface_pressure,shortwave_radiation"
)


def _retry_request(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int,
    backoff_factor: float,
    timeout: int,
) -> requests.Response:
    """Execute request with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(backoff_factor**attempt)
    raise last_exc  # type: ignore


class WeatherClient:
    """
    Fetches hourly weather from Open-Meteo (archive for past, forecast for future).
    No API key required for non-commercial use.
    """

    def __init__(
        self,
        base_url: str | None = None,
        forecast_url: str | None = None,
        timeout_sec: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        cfg = get_config()
        w = cfg.get("ingestion", {}).get("weather", {})
        self.base_url = base_url or w.get(
            "base_url", "https://archive-api.open-meteo.com/v1/archive"
        )
        self.forecast_url = forecast_url or w.get(
            "forecast_url", "https://api.open-meteo.com/v1/forecast"
        )
        self.timeout = timeout_sec or w.get("timeout_sec", 30)
        self.max_retries = max_retries or w.get("max_retries", 3)

    def fetch_hourly(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        latitude: float,
        longitude: float,
        hourly_vars: str = HOURLY_VARS,
    ) -> pd.DataFrame:
        """
        Fetch hourly weather for date range and lat/lon.
        Uses archive API for past dates. Returns DataFrame with datetime (UTC) index.
        """
        params = WeatherFetchParams(
            start_date=pd.Timestamp(start_date).date(),
            end_date=pd.Timestamp(end_date).date(),
            latitude=latitude,
            longitude=longitude,
        )
        req_params = {
            "latitude": params.latitude,
            "longitude": params.longitude,
            "start_date": params.start_date.isoformat(),
            "end_date": params.end_date.isoformat(),
            "hourly": hourly_vars,
            "timezone": "UTC",
        }
        session = requests.Session()
        resp = _retry_request(session, self.base_url, req_params, self.max_retries, 2, self.timeout)
        data = resp.json()
        if "hourly" not in data:
            return pd.DataFrame()
        h = data["hourly"]
        df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
        df["datetime"] = pd.to_datetime(h["time"], utc=True)
        df = df.set_index("datetime")
        return df
