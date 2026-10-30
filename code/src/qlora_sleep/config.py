"""Config loading: `base.yaml` plus one experiment overlay, merged recursively."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(name: str | Path) -> dict:
    """Load `name` (a path, or a bare name resolved against `configs/`) over base.yaml."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{path.name}.yaml"
    elif not path.exists():
        path = CONFIG_DIR / path.name
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        overlay = yaml.safe_load(fh) or {}
    base_path = CONFIG_DIR / "base.yaml"
    base: dict[str, Any] = {}
    if base_path.exists() and base_path.resolve() != path.resolve():
        with base_path.open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
    return deep_merge(base, overlay)
