#!/usr/bin/env python3
"""CLI to fetch PJM LMP and weather data, save to data/raw as parquet."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.ingestion.pjm_client import PJMClient
from electricity_forecast.ingestion.weather_client import WeatherClient


def _parse_args() -> argparse.Namespace:
    end = datetime.now().date()
    start = end - timedelta(days=30)
    parser = argparse.ArgumentParser(description="Fetch electricity and weather data")
    parser.add_argument("--start", type=str, default=str(start), help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=str(end), help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--node",
        type=str,
        default="PJM-RTO",
        help="PJM node or zone ID (default: PJM-RTO)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=39.95,
        help="Latitude for weather (default: 39.95, Philadelphia)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-75.17,
        help="Longitude for weather (default: -75.17)",
    )
    parser.add_argument("--pjm-only", action="store_true", help="Fetch only LMP data")
    parser.add_argument("--weather-only", action="store_true", help="Fetch only weather")
    parser.add_argument(
        "--iso",
        type=str,
        default="pjm",
        choices=["pjm", "caiso"],
        help="ISO: pjm (needs PJM_API_KEY) or caiso (no key)",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    raw_dir = Path(args.out_dir or cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    if not args.weather_only:
        df_lmp = None
        try:
            if args.iso == "caiso":
                from electricity_forecast.ingestion.gridstatus_client import fetch_caiso_lmp

                df_lmp = fetch_caiso_lmp(start, end)
            else:
                import os

                if os.environ.get("PJM_API_KEY"):
                    client = PJMClient()
                    df_lmp = client.fetch_day_ahead_lmp(start, end, args.node)
                else:
                    raise ValueError("PJM requires PJM_API_KEY. Use --iso caiso for no-key data.")
        except (ValueError, ImportError) as e:
            print(f"LMP fetch skipped: {e}")
            print(
                "Use --iso caiso for CAISO (no key), or set PJM_API_KEY for PJM, or --weather-only"
            )
        if df_lmp is not None and not df_lmp.empty:
            out = raw_dir / f"lmp_{args.iso}_{start.date()}_{end.date()}.parquet"
            df_lmp.to_parquet(out, index=False)
            print(f"Saved {args.iso.upper()} LMP to {out}")
        elif df_lmp is not None:
            print("LMP returned empty data. Check date range.")

    if not args.pjm_only:
        client = WeatherClient()
        df_weather = client.fetch_hourly(start, end, args.lat, args.lon)
        if not df_weather.empty:
            out = raw_dir / f"weather_{args.lat}_{args.lon}_{start.date()}_{end.date()}.parquet"
            df_weather.to_parquet(out)
            print(f"Saved weather to {out}")


if __name__ == "__main__":
    main()
