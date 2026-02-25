"""Pytest fixtures and sample data for CI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_lmp_df() -> pd.DataFrame:
    """Small LMP-like dataframe for tests."""
    n = 500
    rng = np.random.default_rng(42)
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    lmp = 30 + 20 * np.sin(np.arange(n) / 24) + rng.standard_normal(n) * 5
    return pd.DataFrame({"datetime_begin": ts, "lmp": lmp})


@pytest.fixture
def sample_weather_df() -> pd.DataFrame:
    """Small weather dataframe for tests."""
    n = 500
    rng = np.random.default_rng(43)
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "datetime": ts,
            "temperature_2m": 15 + 10 * np.sin(np.arange(n) / 24) + rng.standard_normal(n) * 2,
            "cloud_cover": rng.integers(0, 100, n),
        }
    ).set_index("datetime")


@pytest.fixture
def sample_modeling_df() -> pd.DataFrame:
    """Modeling table with features for tests."""
    n = 400
    rng = np.random.default_rng(44)
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    target = 40 + 15 * np.sin(np.arange(n) / 24) + rng.standard_normal(n) * 3
    df = pd.DataFrame({"datetime": ts, "target": target})
    for lag in [1, 24, 168]:
        df[f"lag_{lag}"] = df["target"].shift(lag)
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    return df.dropna().reset_index(drop=True)


@pytest.fixture
def interim_sample_dir() -> Path:
    """Create data/interim with sample parquet for CI."""
    d = Path("data/interim")
    d.mkdir(parents=True, exist_ok=True)
    n = 300
    rng = np.random.default_rng(45)
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    lmp = 35 + 15 * np.sin(np.arange(n) / 24) + rng.standard_normal(n) * 4
    df = pd.DataFrame({"datetime_begin": ts, "lmp": lmp})
    out = d / "sample_lmp.parquet"
    df.to_parquet(out, index=False)
    return d
