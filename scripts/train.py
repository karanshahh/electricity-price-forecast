#!/usr/bin/env python3
"""Train model end-to-end, log to MLflow."""

import argparse
from pathlib import Path

import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.evaluation import run_rolling_backtest
from electricity_forecast.evaluation.metrics import mae, mape, rmse, smape
from electricity_forecast.mlops.tracking import log_run, setup_mlflow
from electricity_forecast.models import get_model_class
from electricity_forecast.transforms.splits import time_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train forecast model")
    parser.add_argument(
        "--model", type=str, default="xgb", choices=["xgb", "naive", "seasonal", "lstm"]
    )
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument(
        "--out", type=str, default="data/processed/model.pt", help="Save model path"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    data_path = Path(args.data) if args.data else processed_dir / "modeling_table.parquet"

    if not data_path.exists():
        data_path = Path(cfg["data"]["interim_dir"]) / "sample_modeling.parquet"
    if not data_path.exists():
        print("No data. Run: make features")
        return

    df = pd.read_parquet(data_path)
    ts_col = "datetime" if "datetime" in df.columns else df.columns[0]
    split = time_split(df, ts_col=ts_col)

    setup_mlflow()
    model_cls = get_model_class(args.model)
    model = model_cls()
    model.fit(split.train, val_df=split.val)

    pred = model.predict(split.test)
    if isinstance(pred, pd.DataFrame):
        pred = pred["point"] if "point" in pred.columns else pred.iloc[:, 0]
    y_true = split.test["target"].values
    y_pred = pred.values[: len(y_true)]

    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }

    artifacts = []
    manifest = Path(processed_dir) / "feature_manifest.json"
    if manifest.exists():
        artifacts.append(manifest)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(out_path)
        artifacts.append(out_path)

    log_run(
        params={"model": args.model},
        metrics=metrics,
        artifacts=artifacts,
        model=model,
        model_name=args.model,
    )

    summary = run_rolling_backtest(model, df, ts_col=ts_col, log_mlflow=True)
    print(f"Val metrics: {metrics}")
    print(f"Backtest MAE: {summary['mae_mean']:.4f}")


if __name__ == "__main__":
    main()
