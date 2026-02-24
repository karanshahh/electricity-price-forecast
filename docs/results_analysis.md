# Results Analysis

## Summary: Did We Get Results?

**Yes.** The pipeline ran end-to-end on **real CAISO data** and produced forecasts, metrics, and artifacts. Results are from a **Naive Last** baseline on real California day-ahead LMP (Feb 2026). They demonstrate the system works; the baseline is weak by design.

---

## Data Overview

| Metric | Value |
|--------|-------|
| **Source** | Real CAISO (California) day-ahead LMP |
| **Date range** | 2026-02-20 to 2026-02-24 (~5 days) |
| **Observations** | 95 hourly LMP values |
| **Features** | 37 columns (lags, rolling stats, calendar, volatility) |

Data fetched via `gridstatus` from CAISO OASIS API—no API key required.

---

## Model: Naive Last

The **Naive Last** model predicts the last observed price for every future hour. It is a minimal baseline used to validate the pipeline.

---

## Backtest Results (Real CAISO Data)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 25.57 $/MWh | Average absolute error per hour |
| **RMSE** | 28.75 $/MWh | Root mean squared error |
| **MAPE** | 99.49% | Mean absolute percentage error |
| **SMAPE** | 196.68% | Symmetric MAPE |

### Interpretation

- **MAE 25.57 $/MWh** — Naive baseline struggles with real price volatility.
- **MAPE 99.49%** — High; naive model does not adapt to price changes.
- **1 fold** — Limited by ~5 days of data; more data would yield more folds.
- **SMAPE > 100%** — Can occur when predictions and actuals diverge significantly.

---

## What This Tells Us

1. **Pipeline works** — Fetch (CAISO) → features → train → backtest → artifacts all complete.
2. **Real data** — CAISO LMP is real market data, not synthetic.
3. **Baseline is weak** — Naive Last is a sanity check, not a useful forecaster.
4. **Next steps** — Train XGBoost or LSTM (`brew install libomp` on macOS) for better forecasts.

---

## Limitations

- **Short horizon** — 5 days of data; more history improves backtest robustness.
- **Naive model** — No learning; just repeats the last value.
- **Single fold** — One train/test split; more data enables rolling evaluation.
