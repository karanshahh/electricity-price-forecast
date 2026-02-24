"""PJM Data Miner API client for day-ahead LMP time series."""

import time
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console
from rich.logging import RichHandler

from electricity_forecast.config import get_config
from electricity_forecast.ingestion.schemas import PJMFetchParams

console = Console()
logger = console


def _retry_request(
    session: requests.Session,
    url: str,
    params: dict,
    headers: dict | None,
    max_retries: int,
    backoff_factor: float,
    timeout: int,
) -> requests.Response:
    """Execute request with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = backoff_factor ** attempt
                time.sleep(wait)
    raise last_exc  # type: ignore


class PJMClient:
    """
    Downloads day-ahead LMP time series from PJM Data Miner API.
    Requires PJM_API_KEY env var if API mandates authentication.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int | None = None,
        max_retries: int | None = None,
        retry_backoff_factor: float | None = None,
    ) -> None:
        cfg = get_config()
        pjm_cfg = cfg.get("ingestion", {}).get("pjm", {})
        self.base_url = base_url or pjm_cfg.get("base_url", "https://api.pjm.com/api/v1")
        self.api_key = api_key
        self.timeout = timeout_sec or pjm_cfg.get("timeout_sec", 60)
        self.max_retries = max_retries or pjm_cfg.get("max_retries", 3)
        self.backoff = retry_backoff_factor or pjm_cfg.get("retry_backoff_factor", 2)

    def _ensure_api_key(self) -> None:
        """Raise clear error if API key is required but missing."""
        if self.api_key:
            return
        import os

        key = os.environ.get("PJM_API_KEY")
        if key:
            self.api_key = key
            return
        raise ValueError(
            "PJM API requires authentication. Set PJM_API_KEY in .env or pass api_key. "
            "Register at https://apiportal.pjm.com/ for a free API key."
        )

    def fetch_day_ahead_lmp(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        node_or_zone: str,
    ) -> pd.DataFrame:
        """
        Fetch day-ahead LMP for date range and node/zone.
        Returns DataFrame with datetime_begin (UTC), node/zone, lmp columns.
        """
        params = PJMFetchParams(
            start_date=pd.Timestamp(start_date).date(),
            end_date=pd.Timestamp(end_date).date(),
            node_or_zone=node_or_zone,
        )
        self._ensure_api_key()
        url = f"{self.base_url.rstrip('/')}/day_ahead_lmp"
        req_params = {
            "datetime_begin": params.start_date.isoformat(),
            "datetime_end": params.end_date.isoformat(),
            "node_id": params.node_or_zone,
        }
        headers = {"Ocp-Apim-Key": self.api_key} if self.api_key else {}
        session = requests.Session()
        resp = _retry_request(
            session, url, req_params, headers, self.max_retries, self.backoff, self.timeout
        )
        data = resp.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data) if data else pd.DataFrame()
        if df.empty:
            return df
        ts_col = next((c for c in df.columns if "datetime" in c.lower() or "time" in c.lower()), None)
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        return df
