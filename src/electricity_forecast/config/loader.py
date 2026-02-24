"""Load YAML config and merge with environment variables."""

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file into dict."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path = "configs") -> dict[str, Any]:
    """
    Load config from YAML files. Merges base config with local overrides.
    Order: config.yaml < config.local.yaml (local overrides base).
    """
    config_dir = Path(config_dir)
    base = _load_yaml(config_dir / "config.yaml")
    local = _load_yaml(config_dir / "config.local.yaml")
    return {**base, **{k: v for k, v in local.items() if v is not None}}


_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Return cached config, loading if needed."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
