"""Streamlit dashboard: date range, forecasts, backtest metrics, scenario what-if."""

from pathlib import Path

import pandas as pd
import streamlit as st

from electricity_forecast.config import get_config
from electricity_forecast.evaluation.plots import plot_forecast_vs_actual

st.set_page_config(page_title="Electricity Price Forecast", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Price Forecast")

cfg = get_config()
processed_dir = Path(cfg["data"]["processed_dir"])
interim_dir = Path(cfg["data"]["interim_dir"])
data_path = processed_dir / "modeling_table.parquet"

tab1, tab2, tab3, tab4 = st.tabs(["Forecasts", "Backtest", "Scenario", "Data"])

with tab1:
    st.subheader("Forecast Visualization")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        ts_col = "datetime" if "datetime" in df.columns else df.columns[0]
        date_range = st.date_input(
            "Date range", value=(df[ts_col].min().date(), df[ts_col].max().date())
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            mask = (df[ts_col].dt.date >= date_range[0]) & (df[ts_col].dt.date <= date_range[1])
            sub = df[mask]
            if len(sub) > 0 and "target" in sub.columns:
                fig = plot_forecast_vs_actual(sub["target"], sub["target"], sub[ts_col], "Price")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `make features` to build data. Run `make train` to train model.")

with tab2:
    st.subheader("Backtest Metrics")
    report_path = Path("docs") / "backtest_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text())
    else:
        st.info("Run `make backtest` to generate report.")

with tab3:
    st.subheader("Scenario: Temperature Delta")
    delta = st.slider("Temperature change (°C)", -10.0, 10.0, 0.0, 0.5)
    st.write(f"Apply +{delta}°C to weather inputs for what-if analysis.")
    st.caption("Connect to model pipeline to run scenario forecasts.")

with tab4:
    st.subheader("Data Summary")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        st.dataframe(df.head(100), use_container_width=True)
        st.metric("Rows", len(df))
    manifest = processed_dir / "feature_manifest.json"
    if manifest.exists():
        import json

        m = json.loads(manifest.read_text())
        st.json(m)
