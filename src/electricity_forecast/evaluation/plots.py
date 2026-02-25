"""Plots: actual vs forecast, error distribution, calibration."""


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_forecast_vs_actual(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    timestamps: pd.Series | None = None,
    title: str = "Forecast vs Actual",
) -> go.Figure:
    """Time series plot of actual vs forecast."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    n = min(len(y_t), len(y_p))
    y_t, y_p = y_t[:n], y_p[:n]
    if timestamps is not None:
        ts = timestamps.iloc[:n] if hasattr(timestamps, "iloc") else timestamps[:n]
    else:
        ts = pd.RangeIndex(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=y_t, name="Actual", line={"color": "blue"}))
    fig.add_trace(go.Scatter(x=ts, y=y_p, name="Forecast", line={"color": "orange", "dash": "dash"}))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price")
    return fig


def plot_error_distribution(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    title: str = "Error Distribution",
) -> go.Figure:
    """Histogram of forecast errors."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    n = min(len(y_t), len(y_p))
    err = y_pred[:n] - y_true[:n]
    fig = px.histogram(x=err, nbins=50, title=title)
    fig.update_layout(xaxis_title="Error (Pred - Actual)")
    return fig


def plot_calibration(
    y_true: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
    nominal: float = 0.9,
    title: str = "Forecast Interval Calibration",
) -> go.Figure:
    """Calibration plot: coverage of prediction intervals."""
    y_t = np.asarray(y_true).ravel()
    lo = np.asarray(lower).ravel()
    hi = np.asarray(upper).ravel()
    n = min(len(y_t), len(lo), len(hi))
    covered = np.sum((y_t[:n] >= lo[:n]) & (y_t[:n] <= hi[:n])) / n * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n), y=y_t[:n], name="Actual", line={"color": "blue"}))
    fig.add_trace(
        go.Scatter(x=np.arange(n), y=lo[:n], name="Lower", line={"color": "gray", "dash": "dot"})
    )
    fig.add_trace(
        go.Scatter(x=np.arange(n), y=hi[:n], name="Upper", line={"color": "gray", "dash": "dot"})
    )
    fig.update_layout(
        title=f"{title} (Coverage: {covered:.1f}%, Nominal: {nominal * 100:.0f}%)",
        xaxis_title="Index",
        yaxis_title="Price",
    )
    return fig
