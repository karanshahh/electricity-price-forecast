# Backtest Report
**Model:** xgb
**Folds:** 5

## Forecast Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 2.99 $/MWh | Avg absolute error per hour |
| RMSE | 4.52 $/MWh | Penalizes large errors |
| SMAPE | 12.16% | Symmetric % error (robust) |
| Directional Accuracy | 70.0% | Correct up/down calls |

## Strategy Metrics (no look-ahead)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| PnL (mean/fold) | 1335.24 $ | Total profit/loss |
| Sharpe (mean) | 9.37 | Risk-adjusted return |
| Win Rate | 71.5% | % profitable trades |
| Max Drawdown | 18.6% | Peak-to-trough decline |

## Baseline Comparison
| Model | MAE ($/MWh) |
|-------|-------------|
| xgb | 2.99 |
| lag_1 | 2.76 |
| lag_24 | 5.53 |
| lag_168 | 8.75 |

## Per-Fold Metrics
| Fold | MAE | Dir% | PnL | Sharpe | Win% | MaxDD% |
|------|-----|------|-----|--------|------|--------|
| 0 | 2.70 | 73.5 | 1671.88 | 19.37 | 74.9% | 4.0 |
| 1 | 2.66 | 70.1 | 1392.15 | 16.02 | 72.6% | 11.4 |
| 2 | 3.03 | 69.3 | 1185.20 | 3.16 | 71.5% | 20.7 |
| 3 | 3.03 | 70.1 | 1206.26 | 3.34 | 70.6% | 6.0 |
| 4 | 3.51 | 67.2 | 1220.72 | 4.94 | 68.2% | 50.7 |