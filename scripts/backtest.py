#!/usr/bin/env python3
"""Run rolling backtest for selected model, produce report in docs/."""

import argparse
from pathlib import Path

import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.evaluation import run_rolling_backtest
from electricity_forecast.models import get_model_class


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling backtest")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "naive", "seasonal", "lstm"], help="Model type")
    parser.add_argument("--data", type=str, default=None, help="Processed parquet path")
    parser.add_argument("--report-dir", type=str, default="docs", help="Report output dir")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    data_path = Path(args.data) if args.data else processed_dir / "modeling_table.parquet"

    if not data_path.exists():
        interim = Path(cfg["data"]["interim_dir"]) / "sample_modeling.parquet"
        if interim.exists():
            data_path = interim
        else:
            print("No data found. Run: make features")
            return

    df = pd.read_parquet(data_path)
    ts_col = "datetime" if "datetime" in df.columns else df.columns[0]
    model_cls = get_model_class(args.model)
    model = model_cls()

    summary = run_rolling_backtest(
        model, df, ts_col=ts_col, log_mlflow=not args.no_mlflow
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "backtest_report.md"

    md = [
        "# Backtest Report",
        f"**Model:** {args.model}",
        f"**Folds:** {summary['n_folds']}",
        "",
        "## Aggregate Metrics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MAE | {summary['mae_mean']:.4f} |",
        f"| RMSE | {summary['rmse_mean']:.4f} |",
        f"| MAPE | {summary['mape_mean']:.2f}% |",
        f"| SMAPE | {summary['smape_mean']:.2f}% |",
        "",
        "## Per-Fold Metrics",
        "| Fold | MAE | RMSE | MAPE | SMAPE |",
        "|------|-----|------|------|-------|",
    ]
    for f in summary["folds"]:
        md.append(f"| {f['fold']} | {f['mae']:.4f} | {f['rmse']:.4f} | {f['mape']:.2f} | {f['smape']:.2f} |")

    with open(report_path, "w") as f:
        f.write("\n".join(md))

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
