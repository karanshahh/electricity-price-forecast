# Results Analysis

## Summary: Did We Get Results?

**Yes.** The pipeline ran end-to-end and produced forecasts, metrics, and artifacts. The results are from a **Naive Last** baseline model trained on synthetic sample data (~17 days, 399 hourly observations). They are not production-ready but demonstrate that the system works.

---

## Data Overview

| Metric | Value |
|--------|-------|
| **Date range** | 2023-01-01 to 2023-01-17 (~17 days) |
| **Observations** | 399 hourly LMP values |
| **Target (LMP)** | Mean 36.4 $/MWh, Std 11.3, Range [13.3, 57.8] |
| **Features** | 37 columns (lags, rolling stats, calendar, volatility) |

The sample data is synthetic (sinusoidal + noise), not real PJM data. Real LMPs are more volatile and have spikes; this sample is smoother.

---

## Model: Naive Last

The **Naive Last** model predicts the last observed price for every future hour. It is a minimal baseline used to validate the pipeline.

---

## Backtest Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 14.00 $/MWh | Average absolute error per hour |
| **RMSE** | 17.33 $/MWh | Root mean squared error (penalizes large errors) |
| **MAPE** | 54.67% | Mean absolute percentage error |
| **SMAPE** | 36.74% | Symmetric MAPE (bounded 0–200%) |

### Interpretation

- **MAE 14 $/MWh** on a mean price of ~36 $/MWh implies errors of ~38% of the mean on average. For a naive baseline, this is expected.
- **MAPE 54.67%** is high; naive models struggle when prices change.
- **1 fold** only: with ~17 days of data, the rolling backtest produced a single fold (train on first half, test on second). More data would yield more folds and more stable metrics.

---

## What This Tells Us

1. **Pipeline works** — Fetch → features → train → backtest → artifacts all complete.
2. **Baseline is weak** — Naive Last is a sanity check, not a useful forecaster.
3. **Next steps** — Train XGBoost or LSTM on real PJM data (with `brew install libomp` for XGB on macOS) to get meaningful forecasts.
4. **Data scale** — With real data (often 100+ $/MWh spikes), expect higher MAE/RMSE in absolute terms; MAPE/SMAPE are more comparable across scales.

---

## Limitations

- **Sample data** — Synthetic; real PJM LMPs have different dynamics.
- **Single fold** — Limited backtest; more data needed for robust evaluation.
- **Naive model** — No learning; just repeats the last value.
- **No XGBoost/LSTM** — Not run due to libomp on this machine; those models should outperform the baseline.
