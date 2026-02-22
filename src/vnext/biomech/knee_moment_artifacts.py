from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .knee_moment_2d import KneeMoment2DResult


def write_knee_moment_2d_artifacts(
    *,
    out_dir: str | Path,
    run_id: str,
    theta_shank_rad: np.ndarray,
    fz_n: np.ndarray,
    result: KneeMoment2DResult,
    overwrite: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    d = out_dir / run_id
    d.mkdir(parents=True, exist_ok=True)

    def save(name: str, arr: np.ndarray) -> None:
        p = d / name
        if p.exists() and not overwrite:
            raise FileExistsError(str(p))
        np.save(p, np.asarray(arr))

    save("theta_shank_rad.npy", theta_shank_rad)
    save("fz.npy", fz_n)
    save("knee_moment.npy", result.moment)
    save("knee_moment_filtered.npy", result.moment_filtered)
    save("omega_shank_rad_s.npy", result.omega)
    save("alpha_shank_rad_s2.npy", result.alpha)

    terms_path = d / "moment_terms.npz"
    if terms_path.exists() and not overwrite:
        raise FileExistsError(str(terms_path))
    np.savez(terms_path, **{k: np.asarray(v) for k, v in result.terms.items()})

    meta: Dict[str, Any] = dict(result.metadata)
    meta["run_id"] = str(run_id)
    meta["shapes"] = {
        "theta": [int(x) for x in np.asarray(theta_shank_rad).shape],
        "fz": [int(x) for x in np.asarray(fz_n).shape],
        "moment": [int(x) for x in np.asarray(result.moment).shape],
    }

    meta_path = d / "metadata.json"
    if meta_path.exists() and not overwrite:
        raise FileExistsError(str(meta_path))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return d
