#!/usr/bin/env python3
"""Generate small sample data in data/interim for CI tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.transforms import build_features, clean_lmp


def main() -> None:
    cfg = get_config()
    interim = Path(cfg["data"]["interim_dir"])
    interim.mkdir(parents=True, exist_ok=True)

    n = 400
    rng = np.random.default_rng(42)
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    lmp = 35 + 15 * np.sin(np.arange(n) / 24) + rng.standard_normal(n) * 4
    df = pd.DataFrame({"datetime_begin": ts, "lmp": lmp})

    lmp_path = interim / "sample_lmp.parquet"
    df.to_parquet(lmp_path, index=False)
    print(f"Saved {lmp_path}")

    df = clean_lmp(df, ts_col="datetime_begin", value_col="lmp")
    df = build_features(df, target_col="lmp", ts_col="datetime_begin", weather_df=None)

    modeling_path = interim / "sample_modeling.parquet"
    df.to_parquet(modeling_path, index=False)
    print(f"Saved {modeling_path}")


if __name__ == "__main__":
    main()
