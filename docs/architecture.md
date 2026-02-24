# Architecture

## Overview

Day-ahead electricity price forecasting pipeline for PJM and CAISO markets. Uses LMP (locational marginal price) and optional weather data.

## Pipeline

```
fetch → features → train → backtest → run_api / run_app
```

1. **Fetch**: CAISO LMP (no key) or PJM LMP (API key) + Open-Meteo weather → `data/raw/*.parquet`
2. **Features**: Clean, transform, build features → `data/processed/modeling_table.parquet`
3. **Train**: Fit model, log to MLflow → `mlruns/`
4. **Backtest**: Rolling-origin evaluation → `docs/backtest_report.md`
5. **Serve**: FastAPI + Streamlit

## Project Structure

```
electricity-price-forecast/
├── src/electricity_forecast/
│   ├── ingestion/     # pjm_client, gridstatus_client (CAISO), weather_client, schemas
│   ├── transforms/    # clean, features, splits
│   ├── models/        # baselines, xgb, sarimax, lstm, calibrate
│   ├── evaluation/    # metrics, backtest, plots
│   ├── mlops/         # tracking, registry, drift, monitor
│   ├── serving/       # api, predict
│   ├── app/           # streamlit_app
│   └── config/        # YAML loader
├── configs/
├── data/raw/, data/interim/, data/processed/
├── scripts/           # fetch_data, build_features, train, backtest
├── docker/
└── tests/
```

## Components

| Component | Purpose |
|-----------|---------|
| **ingestion** | CAISO (gridstatus, no key), PJM (API key), Open-Meteo (weather) |
| **transforms** | Clean (UTC, dedupe, outliers), features (lags, rolling, calendar), splits |
| **models** | Naive, SeasonalNaive, XGB, SARIMAX, LSTM, QuantileXGB |
| **evaluation** | MAE, RMSE, MAPE, SMAPE, pinball; rolling backtest |
| **mlops** | MLflow tracking, registry, drift (PSI/KS) |
| **serving** | FastAPI /predict, Streamlit dashboard |

## Config

- `configs/config.yaml`: base config
- `configs/config.local.yaml`: local overrides
- `.env`: PJM_API_KEY, MLFLOW_TRACKING_URI, etc.
