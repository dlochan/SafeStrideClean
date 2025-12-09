from __future__ import annotations
import os
import platform
from pathlib import Path
import yaml


def get_out_root(dataset_cfg: str | None = None) -> Path:
    # 1) Env override
    env = os.getenv("SAFESTRIDE_OUT_ROOT")
    if env:
        return Path(env)
    # 2) Dataset config field
    if dataset_cfg and Path(dataset_cfg).exists():
        try:
            with open(dataset_cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            key = "out_root_windows" if platform.system().lower().startswith("win") else "out_root"
            if cfg.get(key):
                return Path(cfg[key])
        except Exception:
            pass
    # 3) OS default
    if platform.system().lower().startswith("win"):
        return Path(r"E:\safestride\out_grid")
    return Path("out_grid")
