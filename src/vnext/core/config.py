from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a plain dictionary.

    Expected high-level structure (minimal example):

    paths:
      data_root: "data"
      out_root: "out"

    data:
      train_manifest: "manifests/train_trials.csv"

    training:
      batch_size: 32
      num_workers: 4
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {p} must be a mapping at top level")
    return cfg
