#!/usr/bin/env python3
"""Run rolling backtest for selected model, produce report in docs/."""

import argparse
from pathlib import Path

import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.evaluation import run_rolling_backtest
from electricity_forecast.evaluation.metrics import mae
from electricity_forecast.models import get_model_class
from electricity_forecast.transforms.splits import rolling_folds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling backtest")
    parser.add_argument(
        "--model",
        type=str,
        default="xgb",
        choices=["xgb", "naive", "seasonal", "lstm"],
        help="Model type",
    )
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
    ts_col = "datetime_begin" if "datetime_begin" in df.columns else "datetime" if "datetime" in df.columns else df.columns[0]
    model_cls = get_model_class(args.model)
    model = model_cls()

    summary = run_rolling_backtest(model, df, ts_col=ts_col, log_mlflow=not args.no_mlflow)

    # Baseline comparison
    baseline_maes = {"lag_1": [], "lag_24": [], "lag_168": []}
    for train, test in rolling_folds(df, ts_col=ts_col):
        y_true = test["target"].values
        for col in baseline_maes:
            if col in test.columns:
                pred = test[col].fillna(train[col].mean()).values[: len(y_true)]
                baseline_maes[col].append(mae(y_true, pred))
    baseline_means = {k: sum(v) / len(v) if v else 0 for k, v in baseline_maes.items()}

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "backtest_report.md"

    md = [
        "# Backtest Report",
        f"**Model:** {args.model}",
        f"**Folds:** {summary['n_folds']}",
        "",
        "## Forecast Metrics",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| MAE | {summary['mae_mean']:.2f} $/MWh | Avg absolute error per hour |",
        f"| RMSE | {summary['rmse_mean']:.2f} $/MWh | Penalizes large errors |",
        f"| SMAPE | {summary['smape_mean']:.2f}% | Symmetric % error (robust) |",
        f"| Directional Accuracy | {summary.get('directional_accuracy_mean', 0):.1f}% | Correct up/down calls |",
        "",
        "## Strategy Metrics (no look-ahead)",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| PnL (mean/fold) | {summary.get('strategy_pnl_mean', 0):.2f} $ | Total profit/loss |",
        f"| Sharpe (mean) | {summary.get('strategy_sharpe_mean', 0):.2f} | Risk-adjusted return |",
        f"| Win Rate | {summary.get('strategy_win_rate_mean', 0):.1%} | % profitable trades |",
        f"| Max Drawdown | {summary.get('strategy_max_dd_mean', 0):.1f}% | Peak-to-trough decline |",
        "",
        "## Baseline Comparison",
        "| Model | MAE ($/MWh) |",
        "|-------|-------------|",
        f"| {args.model} | {summary['mae_mean']:.2f} |",
    ]
    for col, m in baseline_means.items():
        if m > 0:
            md.append(f"| {col} | {m:.2f} |")
    md.extend([
        "",
        "## Per-Fold Metrics",
        "| Fold | MAE | Dir% | PnL | Sharpe | Win% | MaxDD% |",
        "|------|-----|------|-----|--------|------|--------|",
    ])
    for f in summary["folds"]:
        pnl = f.get("strategy_pnl", 0)
        sharpe = f.get("strategy_sharpe", 0)
        wr = f.get("strategy_win_rate", 0)
        dd = f.get("strategy_max_dd", 0)
        da = f.get("directional_accuracy", 0)
        md.append(
            f"| {f['fold']} | {f['mae']:.2f} | {da:.1f} | {pnl:.2f} | {sharpe:.2f} | {wr:.1%} | {dd:.1f} |"
        )

    with open(report_path, "w") as f:
        f.write("\n".join(md))

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
