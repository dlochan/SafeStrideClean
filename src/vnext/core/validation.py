from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


def normalize_grf_axes(raw: object | None, model_type: str) -> str:
    if raw is None:
        if model_type == "fz":
            return "fz"
        if model_type == "grf3d":
            return "3d"
        raise ValueError(f"Unknown model.type '{model_type}'")

    grf_axes = str(raw).lower()
    if grf_axes in {"fxyz", "all"}:
        grf_axes = "3d"
    if grf_axes not in {"fz", "3d"}:
        raise ValueError(f"Unsupported grf_axes '{grf_axes}', expected 'fz' or '3d' (aliases: 'all', 'fxyz' -> '3d')")
    return grf_axes


def _require_mapping(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = cfg.get(key, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config '{key}' must be a mapping")
    return value


def validate_paths(paths_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    data_root = paths_cfg.get("data_root")
    out_root = paths_cfg.get("out_root")
    if not data_root or not out_root:
        raise ValueError("Config.paths must define both 'data_root' and 'out_root'")
    return Path(str(data_root)), Path(str(out_root))


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths_cfg = _require_mapping(cfg, "paths")
    _, _ = validate_paths(paths_cfg)

    model_cfg = _require_mapping(cfg, "model")
    model_type = str(model_cfg.get("type", "fz")).lower()
    if model_type not in {"fz", "grf3d"}:
        raise ValueError(f"Unsupported model.type '{model_type}', expected 'fz' or 'grf3d'")

    grf_axes = normalize_grf_axes(model_cfg.get("grf_axes"), model_type=model_type)

    target_grf_column = model_cfg.get("target_grf_column")
    if target_grf_column is not None and grf_axes != "fz":
        raise ValueError("model.target_grf_column is only supported when grf_axes='fz'")

    features_cfg = _require_mapping(cfg, "features")
    enable_kinematics = bool(features_cfg.get("enable_kinematics", False))
    if not isinstance(enable_kinematics, bool):
        raise ValueError("features.enable_kinematics must be a boolean")

    data_cfg = _require_mapping(cfg, "data")
    has_train_manifest = data_cfg.get("train_manifest") is not None
    has_combined_manifest = data_cfg.get("manifest") is not None
    if not (has_train_manifest or has_combined_manifest):
        raise ValueError("Config.data must define either 'train_manifest' (plus optional 'val_manifest') or 'manifest' (plus 'split_column')")
    if has_combined_manifest and not data_cfg.get("split_column"):
        raise ValueError("Config.data.split_column must be set when using data.manifest")

    training_cfg = _require_mapping(cfg, "training")
    for k in ("window_size", "window_stride", "batch_size"):
        if k in training_cfg:
            v = int(training_cfg[k])
            if v <= 0:
                raise ValueError(f"training.{k} must be positive")

    cfg = dict(cfg)
    cfg.setdefault("model", {})
    cfg["model"] = dict(model_cfg)
    cfg["model"]["type"] = model_type
    cfg["model"]["grf_axes"] = grf_axes
    return cfg
