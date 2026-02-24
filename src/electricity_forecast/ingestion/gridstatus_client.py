"""
Fetch LMP via gridstatus — no API key for CAISO.
Use --iso caiso when PJM_API_KEY is not available.
"""

import pandas as pd


def fetch_caiso_lmp(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch day-ahead LMP from CAISO (California) using gridstatus.
    No API key required. Returns DataFrame with datetime_begin (UTC) and lmp.
    """
    try:
        import gridstatus
    except ImportError:
        raise ImportError(
            "Install gridstatus for CAISO data (no key): pip install gridstatus"
        ) from None

    iso = gridstatus.CAISO()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    df = iso.get_lmp(start=start, end=end, market="DAY_AHEAD_HOURLY")

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["datetime_begin"] = pd.to_datetime(df["Time"], utc=True)
    df = df.rename(columns={"LMP": "lmp"})
    out = df.groupby("datetime_begin", as_index=False)["lmp"].mean()
    return out[["datetime_begin", "lmp"]]
