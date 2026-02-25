"""Rolling-origin backtest with metrics per fold."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from electricity_forecast.config import get_config
from electricity_forecast.evaluation.metrics import directional_accuracy, mae, mape, rmse, smape
from electricity_forecast.evaluation.strategy import run_strategy_backtest
from electricity_forecast.models.base import ForecastModel
from electricity_forecast.transforms.splits import rolling_folds


def run_rolling_backtest(
    model: ForecastModel,
    df: pd.DataFrame,
    ts_col: str = "datetime",
    train_days: int | None = None,
    test_days: int | None = None,
    step_days: int | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """
    Run rolling backtest, compute metrics per fold and aggregate.
    Optionally log to MLflow.
    """
    cfg = get_config()
    bt = cfg.get("backtest", {})
    train_days = train_days or bt.get("train_days", 365)
    test_days = test_days or bt.get("test_days", 30)
    step_days = step_days or bt.get("step_days", 7)

    folds = rolling_folds(
        df, ts_col=ts_col, train_days=train_days, test_days=test_days, step_days=step_days
    )
    results = []
    for i, (train, test) in enumerate(folds):
        model.fit(train, val_df=None)
        pred = model.predict(test)
        if isinstance(pred, pd.DataFrame):
            pred = pred["point"] if "point" in pred.columns else pred.iloc[:, 0]
        y_true = test["target"].values
        y_pred = pred.values[: len(y_true)]
        y_prev = test["lag_1"].values[: len(y_true)] if "lag_1" in test.columns else None
        strat = run_strategy_backtest(y_true, y_pred, y_prev=y_prev)
        dir_acc = directional_accuracy(y_true, y_pred, y_prev=y_prev)
        results.append(
            {
                "fold": i,
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mape": mape(y_true, y_pred),
                "smape": smape(y_true, y_pred),
                "directional_accuracy": dir_acc,
                "strategy_pnl": strat.total_pnl,
                "strategy_sharpe": strat.sharpe_ratio,
                "strategy_win_rate": strat.win_rate,
                "strategy_max_dd": strat.max_drawdown_pct,
            }
        )

    agg = pd.DataFrame(results)
    summary = {
        "n_folds": len(results),
        "mae_mean": float(agg["mae"].mean()) if len(results) > 0 else float("nan"),
        "rmse_mean": float(agg["rmse"].mean()) if len(results) > 0 else float("nan"),
        "mape_mean": float(agg["mape"].mean()) if len(results) > 0 else float("nan"),
        "smape_mean": float(agg["smape"].mean()) if len(results) > 0 else float("nan"),
        "strategy_pnl_mean": float(agg["strategy_pnl"].mean())
        if "strategy_pnl" in agg.columns
        else float("nan"),
        "strategy_sharpe_mean": float(agg["strategy_sharpe"].mean())
        if "strategy_sharpe" in agg.columns
        else float("nan"),
        "strategy_win_rate_mean": float(agg["strategy_win_rate"].mean())
        if "strategy_win_rate" in agg.columns
        else float("nan"),
        "strategy_max_dd_mean": float(agg["strategy_max_dd"].mean())
        if "strategy_max_dd" in agg.columns
        else float("nan"),
        "directional_accuracy_mean": float(agg["directional_accuracy"].mean())
        if "directional_accuracy" in agg.columns
        else float("nan"),
        "folds": results,
    }

    if log_mlflow:
        try:
            import mlflow

            mlflow.log_metrics(
                {
                    "backtest_mae": summary["mae_mean"],
                    "backtest_rmse": summary["rmse_mean"],
                    "backtest_mape": summary["mape_mean"],
                    "backtest_smape": summary["smape_mean"],
                }
            )
            interim_dir = Path(cfg["data"]["interim_dir"])
            interim_dir.mkdir(parents=True, exist_ok=True)
            out_path = interim_dir / "backtest_results.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(str(out_path))
        except Exception:
            pass

    return summary
