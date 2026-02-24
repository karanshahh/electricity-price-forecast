"""Forecast metrics: MAE, RMSE, MAPE, SMAPE, pinball loss."""

import numpy as np
import pandas as pd


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean absolute error."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    return float(np.mean(np.abs(y_t[mask] - y_p[mask]))) if mask.any() else float("nan")


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root mean squared error."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    return float(np.sqrt(np.mean((y_t[mask] - y_p[mask]) ** 2))) if mask.any() else float("nan")


def mape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Mean absolute percentage error. Safe handling for near-zero actuals."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p) & (np.abs(y_t) > epsilon)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / (np.abs(y_t[mask]) + epsilon))) * 100)


def smape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Symmetric MAPE. Bounded [0, 200]."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not mask.any():
        return float("nan")
    num = np.abs(y_t[mask] - y_p[mask])
    denom = (np.abs(y_t[mask]) + np.abs(y_p[mask])) / 2 + epsilon
    return float(np.mean(num / denom) * 100)


def pinball_loss(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    quantile: float,
) -> float:
    """Pinball loss for quantile forecast."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not mask.any():
        return float("nan")
    err = y_t[mask] - y_p[mask]
    return float(np.mean(np.maximum(quantile * err, (quantile - 1) * err)))
