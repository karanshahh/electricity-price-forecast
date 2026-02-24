"""MLflow experiment setup and run logging."""

import os
from pathlib import Path
from typing import Any

import mlflow


def setup_mlflow(
    experiment_name: str = "electricity-forecast",
    tracking_uri: str | None = None,
) -> str:
    """Create or get MLflow experiment. Returns experiment_id."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


def log_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path] | None = None,
    model: Any = None,
    model_name: str = "forecast_model",
) -> str | None:
    """Log params, metrics, artifacts, and optionally register model. Returns run_id."""
    artifacts = artifacts or []
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for p in artifacts:
            if p.exists():
                mlflow.log_artifact(str(p))
        if model is not None and hasattr(model, "save"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                tmp = f.name
            try:
                model.save(tmp)
                mlflow.log_artifact(tmp, artifact_path=model_name)
            finally:
                Path(tmp).unlink(missing_ok=True)
        return mlflow.active_run().info.run_id if mlflow.active_run() else None
