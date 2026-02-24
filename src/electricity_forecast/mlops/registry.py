"""Simple registry wrapper to register best model and load production model."""

from pathlib import Path
from typing import Any

import mlflow


PRODUCTION_ALIAS = "production"


def register_production_model(
    run_id: str,
    model_name: str = "electricity_forecast",
    artifact_path: str = "model",
) -> None:
    """Register model from run as production."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, model_name)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, mv.version)


def load_production_model(
    model_name: str = "electricity_forecast",
    alias: str = PRODUCTION_ALIAS,
) -> Any:
    """Load latest production model from registry."""
    model_uri = f"models:/{model_name}/{alias}"
    return mlflow.pyfunc.load_model(model_uri)
