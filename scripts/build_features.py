#!/usr/bin/env python3
"""Load raw parquet, build features, save to data/processed with feature manifest."""

import argparse
import json
from pathlib import Path

import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.transforms import clean_lmp, build_features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature table from raw data")
    parser.add_argument("--input", type=str, help="Input parquet path (raw LMP)")
    parser.add_argument("--weather", type=str, default=None, help="Optional weather parquet")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def _find_raw_files(raw_dir: Path) -> tuple[Path | None, Path | None]:
    """Find latest LMP and weather parquet in raw dir."""
    lmp = list(raw_dir.glob("pjm_lmp_*.parquet")) or list(raw_dir.glob("lmp_*.parquet"))
    weather = list(raw_dir.glob("weather_*.parquet"))
    return (max(lmp, key=lambda p: p.stat().st_mtime) if lmp else None,
            max(weather, key=lambda p: p.stat().st_mtime) if weather else None)


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(args.out_dir or cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        lmp_path = Path(args.input)
        weather_path = Path(args.weather) if args.weather else None
    else:
        lmp_path, weather_path = _find_raw_files(raw_dir)
        if lmp_path is None:
            lmp_path, weather_path = _find_sample()
        if args.weather:
            weather_path = Path(args.weather)

    if lmp_path is None or not lmp_path.exists():
        print("No raw LMP file found. Run: make sample (for CI) or make fetch (for real data)")
        return

    df = pd.read_parquet(lmp_path)
    ts_col = next((c for c in df.columns if "datetime" in c.lower() or "time" in c.lower()), df.columns[0])
    val_col = next((c for c in df.columns if "lmp" in c.lower() or "price" in c.lower()), df.columns[-1])
    df = clean_lmp(df, ts_col=ts_col, value_col=val_col)

    weather_df = None
    if weather_path and weather_path.exists():
        weather_df = pd.read_parquet(weather_path)

    df = build_features(df, target_col=val_col, ts_col=ts_col, weather_df=weather_df)

    out_path = processed_dir / "modeling_table.parquet"
    df.to_parquet(out_path, index=False)

    manifest = {
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "n_rows": len(df),
        "date_range": [str(df[ts_col].min()), str(df[ts_col].max())],
    }
    manifest_path = processed_dir / "feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {out_path} ({len(df)} rows)")
    print(f"Feature manifest: {manifest_path}")


def _find_sample() -> tuple[Path | None, Path | None]:
    """Check data/interim for CI sample."""
    interim = Path("data/interim")
    if interim.exists():
        pjm = list(interim.glob("*lmp*.parquet")) or list(interim.glob("*pjm*.parquet"))
        w = list(interim.glob("*weather*.parquet"))
        return (pjm[0] if pjm else None, w[0] if w else None)
    return (None, None)
