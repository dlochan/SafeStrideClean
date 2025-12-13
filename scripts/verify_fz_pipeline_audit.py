from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import vnext  # noqa: F401
except ModuleNotFoundError as e:
    raise SystemExit(
        "Could not import 'vnext'. Install the repo in editable mode from the repo root: "
        "`python -m pip install -e .`"
    ) from e

from vnext.core.config import load_config
from vnext.core.paths import SafeStridePaths
from vnext.core.validation import normalize_grf_axes, validate_config


def _infer_units_from_col(col: Optional[str]) -> str:
    if not col:
        return "unknown"
    if col.endswith("_N"):
        return "N"
    if col.endswith("_BW"):
        return "BW"
    if col.endswith("_%BW"):
        return "%BW"
    return "unknown"


def _infer_chosen_fz_column(header: list[str], target_grf_column_config: Optional[str]) -> Optional[str]:
    if target_grf_column_config is not None and target_grf_column_config in header:
        return target_grf_column_config
    return next((c for c in ("Fz_N", "Fz_BW", "Fz_%BW") if c in header), None)


def _resolve_path(maybe_path: str | None, data_root: Path) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = Path(str(maybe_path))
    if p.exists():
        return p
    p2 = data_root / p
    if p2.exists():
        return p2
    return p


def _pick_latest_npz(run_dir: Path) -> Path:
    preds_dir = run_dir / "eval" / "preds"
    cand = sorted(preds_dir.glob("fz_windows_pred_truth*.npz"), key=lambda p: p.stat().st_mtime)
    if not cand:
        raise SystemExit(f"No eval export NPZ found under: {preds_dir}")
    return cand[-1]


def _load_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return next(csv.reader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit FZ pipeline indexing / target column selection")
    ap.add_argument("--run-dir", required=True, help="Run directory containing config.yaml + eval exports")
    ap.add_argument(
        "--config",
        default=None,
        help="Optional config.yaml override. If omitted, uses <run_dir>/config.yaml",
    )
    ap.add_argument("--seed", type=int, default=123, help="Random seed for trial/window sampling")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist or is not a directory: {run_dir}")

    cfg_path = Path(args.config) if args.config is not None else (run_dir / "config.yaml")
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = validate_config(load_config(cfg_path))
    cfg_paths: Dict[str, Any] = cfg.get("paths", {}) or {}
    paths = SafeStridePaths.from_env_or_defaults(cfg_paths)

    data_cfg: Dict[str, Any] = cfg.get("data", {}) or {}
    model_cfg: Dict[str, Any] = cfg.get("model", {}) or {}
    model_type = str(model_cfg.get("type", "fz")).lower()
    grf_axes = normalize_grf_axes(model_cfg.get("grf_axes"), model_type=model_type)
    target_grf_column_config: Optional[str] = model_cfg.get("target_grf_column")

    val_manifest_rel = data_cfg.get("val_manifest")
    train_manifest_rel = data_cfg.get("train_manifest")
    if val_manifest_rel is not None:
        eval_manifest_path = paths.data_root / str(val_manifest_rel)
        eval_manifest_name = "val_manifest"
    elif train_manifest_rel is not None:
        eval_manifest_path = paths.data_root / str(train_manifest_rel)
        eval_manifest_name = "train_manifest"
    else:
        raise SystemExit(
            "No evaluation manifest available. Configure data.val_manifest or data.train_manifest in the config."
        )
    if not eval_manifest_path.exists():
        raise SystemExit(f"Evaluation manifest not found: {eval_manifest_path}")

    npz_path = _pick_latest_npz(run_dir)
    npz = np.load(npz_path, allow_pickle=True)
    trial_ids = npz["trial_id"].astype(object)
    start_idxs = npz["start_idx"].astype(np.int64)
    y_true = npz["y_true"].astype(np.float32)
    window_len = int(npz["window_len"]) if "window_len" in npz.files else int(y_true.shape[1])

    manifest_df = pd.read_csv(eval_manifest_path)
    if "trial_id" not in manifest_df.columns or "grf_path" not in manifest_df.columns:
        raise SystemExit("Manifest must contain trial_id and grf_path columns")

    trial_to_grf: Dict[str, Path] = {}
    for _, row in manifest_df.iterrows():
        tid = str(row["trial_id"])
        raw_grf = row.get("grf_path")
        if isinstance(raw_grf, str) and raw_grf.strip():
            trial_to_grf[tid] = _resolve_path(raw_grf, paths.data_root)  # type: ignore[assignment]

    sample_grf_path: Optional[Path] = next((p for p in trial_to_grf.values() if p.exists()), None)
    chosen_grf_column: Optional[str] = None
    chosen_units = "unknown"
    if grf_axes == "fz" and sample_grf_path is not None:
        header = _load_csv_header(sample_grf_path)
        chosen_grf_column = _infer_chosen_fz_column(header, target_grf_column_config)
        chosen_units = _infer_units_from_col(chosen_grf_column)

    print(f"resolved grf_axes={grf_axes}")
    print(
        "target column selection: "
        f"target_grf_column_config={target_grf_column_config}, "
        f"chosen_grf_column={chosen_grf_column}, units={chosen_units}"
    )

    rng = random.Random(int(args.seed))

    unique_trial_ids = sorted({str(t) for t in trial_ids.tolist()})
    sampled_trial_ids = rng.sample(unique_trial_ids, k=min(3, len(unique_trial_ids)))

    trial_checks = []
    grf_cache: Dict[str, pd.Series] = {}
    for tid in sampled_trial_ids:
        grf_path = trial_to_grf.get(tid)
        exists = grf_path is not None and grf_path.exists()
        header = _load_csv_header(grf_path) if exists else []
        col_ok = bool(chosen_grf_column and chosen_grf_column in header) if exists else False
        units = _infer_units_from_col(chosen_grf_column)
        print(f"trial {tid}: grf_path={grf_path} exists={exists} col_ok={col_ok} units={units}")
        trial_checks.append(
            {
                "trial_id": tid,
                "grf_path": str(grf_path) if grf_path is not None else None,
                "grf_exists": bool(exists),
                "target_column": chosen_grf_column,
                "target_column_exists": bool(col_ok),
                "units": units,
            }
        )

    sampled_window_idxs = rng.sample(range(int(start_idxs.shape[0])), k=min(5, int(start_idxs.shape[0])))
    window_checks = []

    for wi in sampled_window_idxs:
        tid = str(trial_ids[wi])
        start = int(start_idxs[wi])
        yt = y_true[wi].reshape(-1)

        grf_path = trial_to_grf.get(tid)
        raw_first5 = None
        yt_first5 = yt[:5].astype(float).tolist()
        max_abs_diff = None
        raw_stats = None

        if grf_path is not None and grf_path.exists() and chosen_grf_column is not None:
            if tid not in grf_cache:
                df = pd.read_csv(grf_path)
                if chosen_grf_column in df.columns:
                    grf_cache[tid] = pd.to_numeric(df[chosen_grf_column], errors="coerce").fillna(0.0)
            raw = grf_cache.get(tid)
            if raw is not None:
                raw_slice = raw.iloc[start : start + window_len].to_numpy(dtype=np.float32)
                raw_first5 = raw_slice[:5].astype(float).tolist()
                if raw_slice.shape[0] == yt.shape[0]:
                    max_abs_diff = float(np.max(np.abs(raw_slice - yt)))
                    raw_stats = {
                        "median": float(np.median(raw_slice)),
                        "p95": float(np.percentile(raw_slice, 95)),
                    }

        y_true_stats = {
            "median": float(np.median(yt)),
            "p95": float(np.percentile(yt, 95)),
            "min": float(np.min(yt)),
            "max": float(np.max(yt)),
        }

        print(
            f"window: trial_id={tid} start_idx={start} "
            f"y_true_median={y_true_stats['median']:.4f} y_true_p95={y_true_stats['p95']:.4f} "
            f"y_true_first5={yt_first5} raw_first5={raw_first5} max_abs_diff={max_abs_diff}"
        )

        window_checks.append(
            {
                "trial_id": tid,
                "start_idx": start,
                "window_len": window_len,
                "y_true_stats": y_true_stats,
                "y_true_first5": yt_first5,
                "raw_first5": raw_first5,
                "raw_stats": raw_stats,
                "max_abs_diff": max_abs_diff,
                "grf_path": str(grf_path) if grf_path is not None else None,
            }
        )

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "config_path": str(cfg_path),
        "eval_manifest": {
            "path": str(eval_manifest_path),
            "source": eval_manifest_name,
        },
        "eval_export_npz": str(npz_path),
        "resolved": {
            "model_type": model_type,
            "grf_axes": grf_axes,
            "target_grf_column_config": target_grf_column_config,
            "chosen_grf_column": chosen_grf_column,
            "chosen_units": chosen_units,
        },
        "trial_checks": trial_checks,
        "window_checks": window_checks,
    }

    out_dir = run_dir / "analysis_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
