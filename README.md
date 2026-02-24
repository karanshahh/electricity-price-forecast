# ⚡ Electricity Price Forecast

**Production-grade day-ahead electricity price forecasting** for PJM and CAISO wholesale markets. End-to-end ML pipeline with ingestion, feature engineering, multiple model families, rigorous backtesting, MLOps tooling, and API/dashboard serving.

---

## Why This Project

Electricity prices are highly volatile and driven by demand, weather, and generation mix. Accurate day-ahead forecasts enable utilities, traders, and large consumers to optimize procurement and hedging. This project demonstrates a **full ML lifecycle** from raw market data to deployed predictions—the kind of system you’d see in a trading or energy analytics team.

---

## Highlights

| Area | What’s Implemented |
|------|--------------------|
| **Data Ingestion** | CAISO (no key), PJM (API key), Open-Meteo (weather), retry/backoff |
| **Feature Engineering** | Lags (1, 2, 24, 48, 168h), rolling stats/quantiles, calendar, holiday/weekend flags, volatility proxies—**no future leakage** |
| **Models** | Naive baselines, XGBoost, SARIMAX, PyTorch LSTM, quantile regression for prediction intervals |
| **Evaluation** | MAE, RMSE, MAPE, SMAPE, pinball loss; rolling-origin backtest; leakage unit tests |
| **MLOps** | MLflow tracking, model registry, drift detection (PSI, KS), monitoring summary |
| **Serving** | FastAPI `/predict` with Pydantic validation, Streamlit dashboard, Docker |
| **Config** | YAML configs, `.env` for secrets, no hardcoded values |

---

## Quickstart

```bash
# 1. Install
make setup

# 2. Fetch real CAISO data (no API key) or make sample for synthetic
make fetch     # Real CAISO LMP + weather
make features

# 3. Train and evaluate
make train --model naive   # or make train (XGB) after brew install libomp
make backtest

# 4. Serve
make run_api   # FastAPI on :8000
make run_app   # Streamlit on :8501
```

**macOS:** `brew install libomp` for XGBoost.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Fetch     │───▶│  Features   │───▶│   Train     │───▶│  Backtest   │───▶│   Serve     │
│ CAISO/PJM   │    │ Clean+Feat  │    │ MLflow log  │    │ Rolling     │    │ API + App   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                    │                   │                  │
      ▼                    ▼                   ▼                  ▼
  data/raw/          data/processed/       mlruns/          docs/report.md
```

---

## Project Structure

```
electricity-price-forecast/
├── src/electricity_forecast/
│   ├── ingestion/          # Data sources
│   │   ├── pjm_client.py    # PJM LMP API (requires key)
│   │   ├── gridstatus_client.py  # CAISO LMP via gridstatus (no key)
│   │   ├── weather_client.py# Open-Meteo archive
│   │   └── schemas.py       # Pydantic validation
│   ├── transforms/
│   │   ├── clean.py         # UTC, dedupe, missing, outliers
│   │   ├── features.py      # Lags, rolling, calendar, no leakage
│   │   └── splits.py       # Time-based train/val/test, rolling folds
│   ├── models/
│   │   ├── baselines.py     # Naive, seasonal naive
│   │   ├── xgb.py           # XGBoost + early stopping + importance
│   │   ├── sarimax.py       # Statsmodels SARIMAX
│   │   ├── lstm.py          # PyTorch LSTM, GPU optional
│   │   └── calibrate.py     # Quantile XGB for intervals
│   ├── evaluation/
│   │   ├── metrics.py      # MAE, RMSE, MAPE, SMAPE, pinball
│   │   ├── backtest.py     # Rolling-origin, MLflow logging
│   │   └── plots.py        # Forecast vs actual, error dist, calibration
│   ├── mlops/
│   │   ├── tracking.py     # MLflow experiment setup
│   │   ├── registry.py     # Production model registration
│   │   ├── drift.py        # PSI, KS for feature/prediction drift
│   │   └── monitor.py      # Drift summary JSON
│   ├── serving/
│   │   ├── api.py          # FastAPI /health, /predict
│   │   └── predict.py      # Model load, Pydantic I/O
│   ├── app/
│   │   └── streamlit_app.py# Dashboard: forecasts, backtest, scenario
│   └── config/             # YAML + env loader
├── configs/config.yaml
├── scripts/
│   ├── fetch_data.py       # CLI: date range, node, lat/lon
│   ├── build_features.py   # Raw → processed + manifest
│   ├── train.py            # End-to-end train + MLflow
│   └── backtest.py         # Rolling backtest + report
├── docker/                 # Dockerfile, docker-compose
├── docs/
│   ├── architecture.md
│   └── data_dictionary.md
└── tests/
    ├── test_features_leakage.py   # No future info in features
    └── test_predict_contract.py   # API contract
```

---

## Key Findings & Design Decisions

### 1. **No Future Leakage**
- All lags, rolling stats, and price-change features use `shift(1)` so only past information is used.
- Unit tests (`test_features_leakage.py`) assert that injected future values never appear in features.

### 2. **Time-Series Splits**
- Train/val/test by date boundaries (no shuffling).
- Rolling-origin backtest: fixed train window, step forward; mimics real deployment.

### 3. **Config-Driven**
- YAML for lags, rolling windows, model hyperparameters.
- `.env` for `PJM_API_KEY`, `MLFLOW_TRACKING_URI`.
- No hardcoded magic numbers; easy to tune without code changes.

### 4. **Data Sources**
- **CAISO** (California): No API key; `make fetch` uses CAISO by default.
- **PJM**: Requires API key (free at [apiportal.pjm.com](https://apiportal.pjm.com)); use `--iso pjm`.

### 5. **Model Interface**
- Common `fit`, `predict`, `save`, `load` across baselines, XGB, SARIMAX, LSTM.
- Quantile regression for prediction intervals (e.g., 10th–90th percentile).

### 6. **Drift Monitoring**
- PSI and KS for feature and prediction drift.
- Summary JSON for downstream alerting or dashboards.

---

## Make Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install package + dev deps |
| `make lint` | Ruff linter |
| `make format` | Ruff format + fix |
| `make test` | Pytest (auto-generates sample data if missing) |
| `make sample` | Generate sample parquet in `data/interim/` |
| `make fetch` | Fetch CAISO LMP + weather (no key; use `--iso pjm` with `PJM_API_KEY` for PJM) |
| `make features` | Build modeling table + feature manifest |
| `make train` | Train model, log to MLflow |
| `make backtest` | Rolling backtest, write `docs/backtest_report.md` |
| `make run_api` | Start FastAPI on :8000 |
| `make run_app` | Start Streamlit on :8501 |

---

## Real Data (No PJM API Key?)

**Option A: CAISO (California) — no API key**

```bash
# Fetch real CAISO day-ahead LMP (California grid)
python scripts/fetch_data.py --iso caiso --start 2024-01-01 --end 2024-01-31
```

**Option B: PJM (requires free API key)**

```bash
# Add to .env: PJM_API_KEY=your_key (register at https://apiportal.pjm.com/)
python scripts/fetch_data.py --start 2024-01-01 --end 2024-01-31 --node PJM-RTO
```

**Option C: Weather only (no LMP)**

```bash
python scripts/fetch_data.py --weather-only --start 2024-01-01 --end 2024-01-31
```

---

## API

```bash
# Health
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamps": ["2024-01-15T12:00:00Z", "2024-01-15T13:00:00Z"]}'
```

Response: `{"forecasts": [...], "lower": null, "upper": null}` (intervals when using quantile model).

---

## Docker

```bash
cd docker && docker-compose up
```

- **API:** http://localhost:8000  
- **Streamlit:** http://localhost:8501  

Volumes: `data/`, `mlruns/` for persistence.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | pandas, numpy, pyarrow, gridstatus (CAISO) |
| ML | scikit-learn, xgboost, statsmodels, PyTorch |
| MLOps | MLflow |
| API | FastAPI, Pydantic, uvicorn |
| App | Streamlit, Plotly |
| Config | PyYAML, pydantic-settings, python-dotenv |
| Dev | pytest, ruff, rich |

---

## Where Results Are Stored

| Output | Location | Description |
|--------|----------|-------------|
| **Sample data** | `data/interim/sample_lmp.parquet`, `sample_modeling.parquet` | Generated by `make sample` |
| **Raw data** | `data/raw/lmp_*.parquet`, `weather_*.parquet` | CAISO/PJM LMP + weather from `make fetch` |
| **Modeling table** | `data/processed/modeling_table.parquet` | Feature table from `make features` |
| **Feature manifest** | `data/processed/feature_manifest.json` | Column list, dtypes, date range |
| **Trained model** | `data/processed/model.pt` | Saved by `make train` |
| **Backtest report** | `docs/backtest_report.md` | Markdown report from `make backtest` |
| **Backtest JSON** | `data/interim/backtest_results.json` | Per-fold metrics (also in MLflow) |
| **MLflow runs** | `mlruns/` | Params, metrics, artifacts per run |

---

## Latest Results (Real CAISO Data)

Backtest on real CAISO day-ahead LMP (Feb 2026, 95 hourly observations), Naive Last model:

| Metric | Value |
|--------|-------|
| MAE | 25.57 $/MWh |
| RMSE | 28.75 $/MWh |
| MAPE | 99.49% |
| SMAPE | 196.68% |

These metrics reflect a weak baseline on real market data. XGBoost or LSTM (after `brew install libomp`) should improve performance.

---

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — Pipeline and components
- **[docs/data_dictionary.md](docs/data_dictionary.md)** — Field and feature definitions
- **[docs/results_analysis.md](docs/results_analysis.md)** — Results analysis and interpretation

---

## License

MIT
