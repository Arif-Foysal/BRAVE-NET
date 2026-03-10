"""
config.py
=========
Configuration loading and validation for BRAVE-Net.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "brave_net_config.yaml"
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Parameters
    ----------
    path : str or Path, optional
        Path to config YAML.  Defaults to ``configs/brave_net_config.yaml``.

    Returns
    -------
    config : dict
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Safely access nested config values.

    Example
    -------
    >>> sr = get_nested(config, "audio", "sample_rate", default=16000)
    """
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
