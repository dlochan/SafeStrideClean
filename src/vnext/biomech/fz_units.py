from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.vnext.core.normalization import TargetNormStats

_DEFAULT_RUN_DIR_REL = Path("data/vnext_gt_real_out/vnext_fz/20260113-161742_0f9d0c7e")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_target_norm(run_dir: Path) -> Tuple[TargetNormStats, Dict[str, Any]]:
    path = run_dir / "target_norm.json"
    if not path.exists():
        raise FileNotFoundError(str(path))

    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"target_norm.json must be a dict, got {type(obj).__name__}")

    if "center" not in obj or "scale" not in obj:
        raise ValueError("target_norm.json missing required keys: center/scale")

    center = obj.get("center")
    scale = obj.get("scale")
    if not isinstance(center, list) or not isinstance(scale, list):
        raise ValueError("target_norm.json center/scale must be lists")
    if len(center) != 1 or len(scale) != 1:
        raise ValueError(f"Expected scalar center/scale for Fz, got center={len(center)} scale={len(scale)}")

    stats = TargetNormStats.from_dict(obj)

    c = float(stats.center.reshape(-1)[0].item())
    s = float(stats.scale.reshape(-1)[0].item())
    if not np.isfinite(c) or not np.isfinite(s) or s == 0.0:
        raise ValueError(f"Invalid denorm params center={c} scale={s}")

    prov: Dict[str, Any] = {
        "target_norm_path": str(path),
        "target_norm_json_sha256": _sha256_file(path),
        "target_norm": {
            "kind": str(stats.kind),
            "center": [c],
            "scale": [s],
        },
    }
    return stats, prov


def to_newtons(
    fz_model: np.ndarray,
    *,
    body_mass_kg: float,
    g: float = 9.81,
    run_dir: str | Path | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rd = Path(run_dir) if run_dir is not None else (_repo_root() / _DEFAULT_RUN_DIR_REL)

    stats, norm_prov = _load_target_norm(rd)

    y = torch.from_numpy(np.asarray(fz_model, dtype=np.float32))
    y_denorm = stats.denormalize(y)
    fz_bw = y_denorm.detach().cpu().numpy().astype(np.float32, copy=False)

    bw_n = float(body_mass_kg) * float(g)
    fz_n = (fz_bw * bw_n).astype(np.float32, copy=False)

    prov: Dict[str, Any] = {
        "units": "newtons",
        "run_dir": str(rd),
        "body_mass_kg": float(body_mass_kg),
        "g": float(g),
        "bw_n": float(bw_n),
        **norm_prov,
    }

    ckpt = rd / "model_best.pt"
    if ckpt.exists():
        prov["model_best"] = {
            "path": str(ckpt),
            "sha256": _sha256_file(ckpt),
        }

    return fz_n, prov
