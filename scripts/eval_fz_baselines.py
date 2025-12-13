from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    import vnext  # noqa: F401
except ModuleNotFoundError as e:
    raise SystemExit(
        "Could not import 'vnext'. Install the repo in editable mode from the repo root: "
        "`python -m pip install -e .`"
    ) from e

from vnext.core.config import load_config
from vnext.core.validation import validate_config, normalize_grf_axes
from vnext.core.paths import SafeStridePaths
from vnext.core.logging_utils import get_logger
from vnext.core.metrics import GRFMetrics, compute_grf_metrics
from vnext.data.datasets import DualIMUTrialDataset, WindowedIMUDataset


def _sanitize_suffix(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s)


def _resolve_manifests(cfg: Dict[str, Any], paths: SafeStridePaths, eval_manifest_rel: str | None) -> Tuple[Path, Path]:
    data_cfg: Dict[str, Any] = cfg.get("data", {}) or {}

    train_manifest_rel = data_cfg.get("train_manifest")
    if train_manifest_rel is None:
        raise SystemExit("Config.data.train_manifest must be set for baseline training.")

    train_manifest_path = paths.data_root / str(train_manifest_rel)
    if not train_manifest_path.exists():
        raise SystemExit(f"Train manifest not found: {train_manifest_path}")

    if eval_manifest_rel is not None:
        eval_manifest_path = paths.data_root / str(eval_manifest_rel)
    else:
        val_manifest_rel = data_cfg.get("val_manifest")
        if val_manifest_rel is None:
            raise SystemExit(
                "No eval manifest provided. Provide --eval-manifest/--manifest or set data.val_manifest."
            )
        eval_manifest_path = paths.data_root / str(val_manifest_rel)

    if not eval_manifest_path.exists():
        raise SystemExit(f"Eval manifest not found: {eval_manifest_path}")

    return train_manifest_path, eval_manifest_path


def _iter_target_values(ds: WindowedIMUDataset) -> List[np.ndarray]:
    ys: List[np.ndarray] = []
    for rec in ds:
        y = rec.get("grf_v")
        if y is None:
            continue
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.size == 0:
            continue
        ys.append(y_arr)
    return ys


def _median_of_targets(ds: WindowedIMUDataset) -> float:
    ys = _iter_target_values(ds)
    if not ys:
        raise SystemExit("No GRF targets found in train windows.")
    y_all = np.concatenate(ys, axis=0)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        raise SystemExit("No finite GRF targets found in train windows.")
    return float(np.median(y_all))


def _window_features_mean_std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    return np.concatenate([mu, sd], axis=0)


def _fit_linear_baseline(train_ds: WindowedIMUDataset, l2: float) -> np.ndarray:
    feats: List[np.ndarray] = []
    targets: List[float] = []

    for rec in train_ds:
        y = rec.get("grf_v")
        x = rec.get("imu")
        if y is None or x is None:
            continue
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if y_arr.size == 0:
            continue
        y_mean = float(np.mean(y_arr))
        x_arr = np.asarray(x, dtype=np.float64)
        feats.append(_window_features_mean_std(x_arr))
        targets.append(y_mean)

    if not feats:
        raise SystemExit("No training windows with GRF/IMU available for linear baseline.")

    X = np.stack(feats, axis=0)
    y = np.asarray(targets, dtype=np.float64)

    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xb = np.concatenate([X, ones], axis=1)

    xtx = Xb.T @ Xb
    reg = l2 * np.eye(xtx.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xtx + reg, Xb.T @ y)
    return w


def _predict_linear(w: np.ndarray, x: np.ndarray) -> float:
    f = _window_features_mean_std(x)
    xb = np.concatenate([f, np.array([1.0], dtype=np.float64)], axis=0)
    return float(xb @ w)


def _run_baseline(
    name: str,
    y_pred_per_window: List[np.ndarray],
    y_true_per_window: List[np.ndarray],
    trial_ids: List[str],
    start_idxs: List[int],
    run_dir: Path,
    eval_manifest_path: Path,
    suffix: str,
    logger,
) -> None:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = eval_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    out_npz = preds_dir / f"fz_windows_pred_truth_{suffix}.npz"
    metrics_json_path = eval_dir / f"eval_metrics_baseline_{name}_{suffix}.json"
    if metrics_json_path.exists():
        logger.info(f"Baseline '{name}': metrics already exist at {metrics_json_path}; skipping")
        return

    if out_npz.exists():
        npz = np.load(out_npz, allow_pickle=True)
        y_true = np.asarray(npz["y_true"], dtype=np.float32)
        y_pred = np.asarray(npz["y_pred"], dtype=np.float32)
    else:
        y_true = np.stack(y_true_per_window, axis=0).astype(np.float32)
        y_pred = np.stack(y_pred_per_window, axis=0).astype(np.float32)
        np.savez_compressed(
            out_npz,
            trial_id=np.array(trial_ids, dtype=object),
            start_idx=np.array(start_idxs, dtype=np.int64),
            y_true=y_true,
            y_pred=y_pred,
            window_len=int(y_true.shape[1]),
        )

    y_true_3d = y_true[:, :, None]
    y_pred_3d = y_pred[:, :, None]

    metrics = compute_grf_metrics(
        y_hat=torch.from_numpy(np.asarray(y_pred_3d, dtype=np.float32)),
        y_true=torch.from_numpy(np.asarray(y_true_3d, dtype=np.float32)),
        axes="fz",
    )

    payload = {
        "baseline": name,
        "manifest": str(eval_manifest_path),
        "metrics": metrics.to_dict(),
        "preds_npz": str(out_npz),
    }
    metrics_json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    logger.info(f"Baseline '{name}': wrote {out_npz}")
    logger.info(f"Baseline '{name}': RMSE_mean={metrics.rmse_mean:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate simple FZ baselines and export NPZ for analysis")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--run-dir", required=True, help="Path to existing run directory")
    ap.add_argument(
        "--eval-manifest",
        "--manifest",
        dest="eval_manifest",
        default=None,
        help="Optional manifest path relative to paths.data_root; defaults to data.val_manifest",
    )
    ap.add_argument(
        "--baselines",
        default="naive,linear",
        help="Comma-separated baselines to run: naive,linear",
    )
    ap.add_argument(
        "--preds-suffix",
        default=None,
        help="Optional suffix prefix; baseline name will be appended.",
    )
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 regularization for linear baseline")
    args = ap.parse_args()

    logger = get_logger("eval_fz_baselines")

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist or is not a directory: {run_dir}")

    cfg = validate_config(load_config(args.config))
    cfg_paths: Dict[str, Any] = cfg.get("paths", {}) or {}
    paths = SafeStridePaths.from_env_or_defaults(cfg_paths)

    model_cfg: Dict[str, Any] = cfg.get("model", {}) or {}
    model_type = str(model_cfg.get("type", "fz")).lower()
    grf_axes = normalize_grf_axes(model_cfg.get("grf_axes"), model_type=model_type)
    if grf_axes != "fz":
        raise SystemExit("Baselines only support grf_axes='fz'.")

    train_cfg: Dict[str, Any] = cfg.get("training", {}) or {}
    window_size = int(train_cfg.get("window_size", 256))
    window_stride = int(train_cfg.get("window_stride", 128))
    require_grf = bool(train_cfg.get("require_grf", True))

    target_grf_column: str | None = model_cfg.get("target_grf_column")

    train_manifest_path, eval_manifest_path = _resolve_manifests(cfg, paths, args.eval_manifest)

    base_train = DualIMUTrialDataset(
        train_manifest_path,
        grf_axes=grf_axes,
        target_grf_column=target_grf_column,
    )
    train_ds = WindowedIMUDataset(
        base_dataset=base_train,
        window_size=window_size,
        window_stride=window_stride,
        require_grf=require_grf,
    )

    base_eval = DualIMUTrialDataset(
        eval_manifest_path,
        grf_axes=grf_axes,
        target_grf_column=target_grf_column,
    )
    eval_ds = WindowedIMUDataset(
        base_dataset=base_eval,
        window_size=window_size,
        window_stride=window_stride,
        require_grf=require_grf,
    )

    if len(train_ds) == 0:
        raise SystemExit("Train windowed dataset is empty.")
    if len(eval_ds) == 0:
        raise SystemExit("Eval windowed dataset is empty.")

    baselines = [b.strip().lower() for b in str(args.baselines).split(",") if b.strip()]
    suffix_prefix = _sanitize_suffix(args.preds_suffix) if args.preds_suffix else ""

    y_true_per_window: List[np.ndarray] = []
    trial_ids: List[str] = []
    start_idxs: List[int] = []
    eval_imu_windows: List[np.ndarray] = []

    for rec in eval_ds:
        y = rec.get("grf_v")
        x = rec.get("imu")
        if y is None or x is None:
            continue
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        x_arr = np.asarray(x, dtype=np.float32)
        y_true_per_window.append(y_arr)
        eval_imu_windows.append(x_arr)
        trial_ids.append(str(rec["trial_id"]))
        start_idxs.append(int(rec["start_idx"]))

    if not y_true_per_window:
        raise SystemExit("No eval windows with GRF found.")

    if "naive" in baselines:
        c = _median_of_targets(train_ds)
        y_pred = [np.full_like(y, fill_value=c, dtype=np.float32) for y in y_true_per_window]
        suffix = f"{suffix_prefix}_baseline_naive" if suffix_prefix else "baseline_naive"
        _run_baseline(
            name="naive",
            y_pred_per_window=y_pred,
            y_true_per_window=y_true_per_window,
            trial_ids=trial_ids,
            start_idxs=start_idxs,
            run_dir=run_dir,
            eval_manifest_path=eval_manifest_path,
            suffix=suffix,
            logger=logger,
        )

    if "linear" in baselines:
        w = _fit_linear_baseline(train_ds, l2=float(args.l2))
        y_pred_lin: List[np.ndarray] = []
        for x_arr, y_arr in zip(eval_imu_windows, y_true_per_window):
            y0 = _predict_linear(w, x_arr.astype(np.float64))
            y_pred_lin.append(np.full_like(y_arr, fill_value=float(y0), dtype=np.float32))
        suffix = f"{suffix_prefix}_baseline_linear" if suffix_prefix else "baseline_linear"
        _run_baseline(
            name="linear",
            y_pred_per_window=y_pred_lin,
            y_true_per_window=y_true_per_window,
            trial_ids=trial_ids,
            start_idxs=start_idxs,
            run_dir=run_dir,
            eval_manifest_path=eval_manifest_path,
            suffix=suffix,
            logger=logger,
        )


if __name__ == "__main__":
    main()
