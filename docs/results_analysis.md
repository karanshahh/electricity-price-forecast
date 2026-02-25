# Results Analysis

## Summary

The pipeline runs end-to-end on **real CAISO data** with **no target leakage**. XGBoost achieves realistic forecast accuracy and a simple directional strategy shows positive PnL with strong risk-adjusted returns.

---

## Data Overview

| Metric | Value |
|--------|-------|
| **Source** | Real CAISO (California) day-ahead LMP |
| **Date range** | 2025-09-01 to 2026-02-24 (~6 months) |
| **Observations** | 2736 hourly rows |
| **Features** | 36 columns (lags, rolling stats, calendar, volatility, weather) |
| **Target** | Next-hour LMP ($/MWh) |

Data fetched via `gridstatus` from CAISO OASIS API—no API key required.

---

## Leakage Fix

Previously, the raw price column `lmp` was used as a feature. Since `lmp` equals the target for each row, this caused severe leakage (MAE ~0.16, unrealistically low). **`lmp` is now excluded** from all model features. Forecasts use only lagged prices, rolling stats, calendar, and weather.

---

## Forecast Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 2.99 $/MWh | Avg absolute error per hour |
| **RMSE** | 4.52 $/MWh | Penalizes large errors |
| **SMAPE** | 12.2% | Symmetric % error (robust to scale) |
| **Directional Accuracy** | 70.0% | % of correct up/down calls |

MAE ~3 $/MWh is realistic for next-hour electricity price forecasting. Directional accuracy of 70% is strong for trading—better than random (50%).

---

## Strategy Metrics (No Look-Ahead)

Position is set using **forecast vs previous price** (known at decision time). No future information is used.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **PnL (mean/fold)** | 1,335 $ | Total profit per fold |
| **Sharpe (mean)** | 9.37 | Risk-adjusted return (annualized) |
| **Win Rate** | 71.5% | % of profitable trades |
| **Max Drawdown** | 18.6% | Peak-to-trough decline |

---

## What This Tells Us

1. **Pipeline is sound** — Leakage removed; metrics are meaningful.
2. **Real data** — CAISO LMP is real market data, not synthetic.
3. **Baseline context** — XGB improves over lag_24/lag_168 but not over lag_1; lag_1 is a strong benchmark.
4. **Overfitting risk** — Large train/test MAE gap warrants regularization or simpler models.

---

## Baseline Comparison

| Model | MAE ($/MWh) | vs XGB |
|-------|-------------|-------|
| **XGB** | 2.99 | — |
| lag_1 (naive last) | 2.76 | **−8%** (lag_1 better) |
| lag_24 (seasonal) | 5.53 | +46% |
| lag_168 (weekly) | 8.75 | +66% |

**Note:** lag_1 is the strongest baseline for next-hour prediction (correlation 0.92 with target). XGB beats lag_24/lag_168 but does not beat lag_1 on MAE. Strategy PnL: lag_1 also outperforms XGB (~2.9k vs 1.3k mean/fold). This suggests the model may be overfitting (train MAE ~0.15 vs test ~3.0) or that lag_1 captures most of the predictable signal.

---

## Limitations

- **lag_1 baseline** — Naive last-hour predictor beats XGB on MAE and strategy PnL; consider lag_1 as production fallback.
- **Overfitting** — Train MAE ~0.15 vs test ~3.0 (≈20× gap); regularization or feature reduction may help.
- **Transaction costs** — Not included; would reduce PnL by ~8–15% at 0.5–1.0 $/MWh.
- **Strategy** — Simple threshold; production would add position sizing, costs, regime filters.
- **Sample** — 6 months; longer history would improve robustness.
- **Distribution shift** — Adversarial validation (AUC 0.75) suggests train/test distribution difference.
