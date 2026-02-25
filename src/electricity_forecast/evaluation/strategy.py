"""Trading strategy backtest: forecast → direction → PnL, Sharpe, drawdown.

No look-ahead: position based on pred vs previous price (known at decision time).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    """Strategy backtest output."""

    total_pnl: float
    sharpe_ratio: float
    n_trades: int
    win_rate: float
    max_drawdown_pct: float
    directional_accuracy: float


def run_strategy_backtest(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prev: pd.Series | np.ndarray | None = None,
    threshold_pct: float = 0.01,
    cost_per_mwh: float = 0.0,
) -> StrategyResult:
    """
    Strategy: go long when forecast > prev_price (expect rise), short when below.
    Position set at decision time using only pred and prev_price (no look-ahead).
    PnL = (actual - prev) * position - costs.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    n = min(len(y_t), len(y_p))
    y_t, y_p = y_t[:n], y_p[:n]

    # Previous price: known at decision time (lag_1). Fallback: shift y_true (in-sample only).
    if y_prev is not None:
        y_prev = np.asarray(y_prev).ravel()[:n]
    else:
        y_prev = np.roll(y_t, 1)
        y_prev[0] = y_t[0]

    # Direction: pred vs prev. Long if pred > prev, short if pred < prev.
    diff_pred = y_p - y_prev
    position = np.where(
        diff_pred > threshold_pct * (np.abs(y_prev) + 1e-6),
        1,
        np.where(diff_pred < -threshold_pct * (np.abs(y_prev) + 1e-6), -1, 0),
    )

    # PnL: (price_change) * position. Price change = actual - prev.
    price_change = y_t - y_prev
    pnl = position * price_change
    pnl[0] = 0

    # Transaction costs: $ per MWh traded (round-trip)
    n_trades = int(np.sum(np.abs(np.diff(position, prepend=0)) > 0))
    total_pnl = float(np.sum(pnl)) - n_trades * cost_per_mwh

    wins = np.sum((pnl > 0) & (position != 0))
    active = np.sum(position != 0)
    win_rate = float(wins / active) if active > 0 else 0.0

    # Directional accuracy: sign(pred - prev) == sign(actual - prev)
    dir_pred = np.sign(diff_pred)
    dir_actual = np.sign(price_change)
    mask = (dir_pred != 0) & (dir_actual != 0)
    dir_acc = float(np.mean(dir_pred[mask] == dir_actual[mask]) * 100) if mask.any() else 0.0

    # Sharpe (annualized, hourly)
    returns = pnl / (np.abs(y_prev) + 1e-6)
    ret_std = np.std(returns)
    sharpe = float(np.mean(returns) / ret_std * np.sqrt(8760)) if ret_std > 1e-10 else 0.0

    # Max drawdown (% of peak cumulative PnL)
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.maximum(cum_pnl, 0))
    dd_pct = np.where(peak > 0, (peak - cum_pnl) / peak * 100, 0)
    max_dd = float(np.nanmax(dd_pct)) if len(dd_pct) > 0 else 0.0

    return StrategyResult(
        total_pnl=total_pnl,
        sharpe_ratio=sharpe,
        n_trades=int(np.sum(position != 0)),
        win_rate=win_rate,
        max_drawdown_pct=max_dd,
        directional_accuracy=dir_acc,
    )
