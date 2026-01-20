from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import subprocess

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
from vnext.core.normalization import ChannelNormStats, TargetNormStats
from vnext.core.metrics import GRFMetrics, compute_grf_metrics
from vnext.data.datasets import DualIMUTrialDataset, WindowedIMUDataset
from vnext.data.imu_schema import get_feature_columns, get_sensor_slices
from vnext.models.vnext_fz import VNextFzModel
from vnext.models.vnext_grf3d import VNextGRF3DModel
from vnext.feats.kinematics import KinematicFeatureBuilder, KinematicFeatureConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible evaluation runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="SafeStride vNext evaluation script")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--run-dir", required=True, help="Path to existing run directory")
    ap.add_argument(
        "--eval-manifest",
        "--manifest",
        dest="eval_manifest",
        help=(
            "Optional manifest path relative to paths.data_root; if omitted, "
            "uses data.val_manifest or falls back to data.train_manifest"
        ),
    )
    ap.add_argument(
        "--checkpoint",
        choices=["last", "best"],
        default="best",
        help="Which checkpoint to evaluate: 'best' (default) or 'last'",
    )
    ap.add_argument(
        "--save-preds",
        action="store_true",
        help=(
            "If set, save per-window FZ predictions and ground truth to "
            "<run_dir>/eval/preds/fz_windows_pred_truth*.npz"
        ),
    )
    ap.add_argument(
        "--preds-suffix",
        default=None,
        help=(
            "Optional suffix for --save-preds output filename. If set, writes to "
            "<run_dir>/eval/preds/fz_windows_pred_truth_<suffix>.npz. "
            "If omitted, uses the default filename fz_windows_pred_truth.npz."
        ),
    )
    ap.add_argument("--device", default="cpu", help="Device string for torch (e.g. cpu or cuda)")
    ap.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    ap.add_argument(
        "--analyze-after-eval",
        action="store_true",
        help=(
            "If set, run scripts/analyze_fz_outputs.py after evaluation using the saved preds export. "
            "Requires --save-preds."
        ),
    )
    ap.add_argument(
        "--analysis-out-dir",
        default=None,
        help=(
            "Optional output directory for --analyze-after-eval. If omitted, uses <run_dir>/analysis_eval "
            "(and appends <preds_suffix>/ when provided)."
        ),
    )
    args = ap.parse_args()

    logger = get_logger("eval_vnext")

    if args.seed is not None:
        set_seed(args.seed)

    cfg = validate_config(load_config(args.config))
    cfg_paths: Dict[str, Any] = cfg.get("paths", {}) or {}
    paths = SafeStridePaths.from_env_or_defaults(cfg_paths)

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist or is not a directory: {run_dir}")

    logger.info(f"Evaluating run directory: {run_dir}")

    data_cfg: Dict[str, Any] = cfg.get("data", {}) or {}

    # Model configuration
    model_cfg: Dict[str, Any] = cfg.get("model", {}) or {}
    model_type = str(model_cfg.get("type", "fz")).lower()
    per_sensor_hidden = int(model_cfg.get("per_sensor_hidden", 32))
    fusion_hidden = int(model_cfg.get("fusion_hidden", 64))
    target_grf_column: str | None = model_cfg.get("target_grf_column")
    backbone = str(model_cfg.get("backbone", "baseline_mlp")).lower()
    model_dropout = float(model_cfg.get("dropout", 0.0))
    tcn_blocks = int(model_cfg.get("tcn_blocks", 5))
    transformer_layers = int(model_cfg.get("transformer_layers", 3))
    transformer_d_model = int(model_cfg.get("transformer_d_model", 96))
    transformer_heads = int(model_cfg.get("transformer_heads", 4))

    # Optional feature configuration
    features_cfg_dict: Dict[str, Any] = cfg.get("features", {}) or {}
    features_cfg = KinematicFeatureConfig(
        enable_kinematics=bool(features_cfg_dict.get("enable_kinematics", False))
    )

    # Training / window configuration (mirrors train_vnext defaults)
    train_cfg: Dict[str, Any] = cfg.get("training", {}) or {}
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))
    window_size = int(train_cfg.get("window_size", 256))
    window_stride = int(train_cfg.get("window_stride", 128))
    require_grf = bool(train_cfg.get("require_grf", True))

    grf_axes = normalize_grf_axes(model_cfg.get("grf_axes"), model_type=model_type)

    gate_path = Path("analysis") / "FZ_TO_3D_GATE.md"
    if grf_axes == "3d" and not gate_path.exists():
        logger.warning(
            "3D GRF requested (grf_axes='3d') but gate file is missing: analysis/FZ_TO_3D_GATE.md. "
            "Per repo policy, do NOT proceed to 3D until the FZ gate is generated and explicitly authorizes it."
        )

    # Resolve evaluation manifest.
    # Default behavior evaluates on data.val_manifest (or data.train_manifest as fallback).
    # When an explicit --eval-manifest/--manifest override is provided, we treat it as a
    # held-out test manifest and label outputs with a "test" suffix.
    manifest_rel_cli = args.eval_manifest
    eval_manifest_path: Path | None = None
    metrics_suffix = "val"
    if manifest_rel_cli:
        eval_manifest_path = paths.data_root / str(manifest_rel_cli)
        metrics_suffix = "test"
        logger.info(f"Using manifest from CLI: {eval_manifest_path}")
    else:
        val_manifest_rel = data_cfg.get("val_manifest")
        train_manifest_rel = data_cfg.get("train_manifest")
        if val_manifest_rel is not None:
            eval_manifest_path = paths.data_root / str(val_manifest_rel)
            logger.info(f"Using data.val_manifest for evaluation: {eval_manifest_path}")
        elif train_manifest_rel is not None:
            eval_manifest_path = paths.data_root / str(train_manifest_rel)
            logger.warning(
                "No data.val_manifest configured; falling back to data.train_manifest for evaluation."
            )
            logger.info(f"Using data.train_manifest for evaluation: {eval_manifest_path}")

    if eval_manifest_path is None:
        raise SystemExit(
            "No evaluation manifest available. Provide --manifest or configure data.val_manifest "
            "or data.train_manifest in the config."
        )

    if not eval_manifest_path.exists():
        raise SystemExit(f"Evaluation manifest not found: {eval_manifest_path}")

    logger.info(f"Eval manifest: {eval_manifest_path}")

    # Build dataset and loader
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

    if len(eval_ds) == 0:
        raise SystemExit("No evaluation windows available; check manifest and window configuration.")

    logger.info("EVAL_FULL_SIZE=%d", len(eval_ds))

    subset_indices_rel = data_cfg.get("subset_indices_path")
    subset_num_windows = int(data_cfg.get("subset_num_windows", 0) or 0)
    if subset_indices_rel:
        subset_path = paths.out_root / str(subset_indices_rel)
        if not subset_path.exists():
            logger.warning(
                "subset_indices_path is set but file does not exist: %s; proceeding without subset restriction.",
                subset_path,
            )
        else:
            try:
                subset_obj = json.loads(subset_path.read_text(encoding="utf-8"))
                requested_pairs = [
                    (str(e["trial_id"]), int(e["start_idx"])) for e in subset_obj
                ]
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Failed to load subset indices from %s: %r; proceeding without subset.", subset_path, e)
                requested_pairs = []

            if requested_pairs:
                if subset_num_windows and subset_num_windows != len(requested_pairs):
                    logger.warning(
                        "subset_num_windows=%d but subset JSON length=%d",
                        subset_num_windows,
                        len(requested_pairs),
                    )
                expected_len = subset_num_windows or len(requested_pairs)
                requested_pairs_set = set(requested_pairs)

                mapping = {}
                for idx in range(len(eval_ds)):
                    rec = eval_ds[idx]
                    if not isinstance(rec, dict):
                        continue
                    tid = str(rec.get("trial_id"))
                    try:
                        sidx = int(rec.get("start_idx"))
                    except Exception:
                        continue
                    key = (tid, sidx)
                    if key in requested_pairs_set and key not in mapping:
                        mapping[key] = idx

                indices = [mapping[p] for p in requested_pairs if p in mapping]
                if len(indices) != expected_len:
                    missing = [p for p in requested_pairs if p not in mapping]
                    raise SystemExit(
                        f"EVAL_subset produced {len(indices)} windows, expected {expected_len}; "
                        f"missing_keys={missing}"
                    )

                first5_pairs = requested_pairs[: min(5, len(indices))]
                last5_pairs = requested_pairs[max(0, len(requested_pairs) - min(5, len(indices))):]
                logger.info(
                    "EVAL_SUBSET_SIZE=%d FIRST5=%s LAST5=%s",
                    len(indices),
                    first5_pairs,
                    last5_pairs,
                )
                eval_ds = Subset(eval_ds, indices)

    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(args.device)

    # Rebuild feature builder and model
    feature_cols = get_feature_columns()
    sensor_slices = get_sensor_slices(feature_cols)
    feature_builder = KinematicFeatureBuilder(
        in_feature_names=feature_cols,
        sensor_slices=sensor_slices,
        cfg=features_cfg,
    )
    in_channels = len(feature_builder.out_feature_names)

    if model_type == "fz":
        model = VNextFzModel(
            in_channels=in_channels,
            sensor_slices=sensor_slices,
            per_sensor_hidden=per_sensor_hidden,
            fusion_hidden=fusion_hidden,
            backbone=backbone,
            dropout=model_dropout,
            tcn_blocks=tcn_blocks,
            transformer_layers=transformer_layers,
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
        ).to(device)
    else:  # "grf3d"
        model = VNextGRF3DModel(
            in_channels=in_channels,
            sensor_slices=sensor_slices,
            per_sensor_hidden=per_sensor_hidden,
            fusion_hidden=fusion_hidden,
        ).to(device)

    logger.info(f"Using model.type='{model_type}', grf_axes='{grf_axes}'")
    logger.info(f"Model in_channels={in_channels}, enable_kinematics={features_cfg.enable_kinematics}")
    logger.info(f"Model backbone='{backbone}'")

    # Load normalization stats from run_dir
    norm_stats_path = run_dir / "norm_stats.json"
    if not norm_stats_path.exists():
        raise SystemExit(f"norm_stats.json not found in run_dir: {norm_stats_path}")
    norm_stats_dict = json.loads(norm_stats_path.read_text(encoding="utf-8"))
    norm_stats = ChannelNormStats.from_dict(norm_stats_dict)

    target_norm: TargetNormStats | None = None
    target_norm_path = run_dir / "target_norm.json"
    if target_norm_path.exists():
        try:
            obj = json.loads(target_norm_path.read_text(encoding="utf-8"))
            target_norm = TargetNormStats.from_dict(obj)
            logger.info(f"Loaded target normalization from {target_norm_path}")
        except Exception:
            logger.warning("Failed to load target_norm.json; continuing without target denormalization.")

    # Decide which checkpoint to load (best vs last)
    ckpt_name = "model_best.pt" if args.checkpoint == "best" else "model_last.pt"
    ckpt_path = run_dir / ckpt_name

    if args.checkpoint == "best" and not ckpt_path.exists():
        logger.warning(
            f"Requested checkpoint='best' but {ckpt_path} not found; falling back to model_last.pt"
        )
        ckpt_name = "model_last.pt"
        ckpt_path = run_dir / ckpt_name

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Eval loop: accumulate metrics and per-window summaries
    n_batches = 0
    metrics_accum: Dict[str, float] = {}
    metrics_count = 0

    window_rows: List[Dict[str, Any]] = []
    pred_trial_ids: List[str] = []
    pred_start_idxs: List[int] = []
    pred_y_true: List[np.ndarray] = []
    pred_y_pred: List[np.ndarray] = []
    y_true_flat: List[np.ndarray] = []

    with torch.no_grad():
        for batch in eval_loader:
            imu = batch["imu"].to(device)  # (B, T, C_in)
            grf_v = batch["grf_v"]

            if grf_v is None:
                continue

            grf_v = grf_v.to(device)  # (B, T, D)

            if grf_axes == "fz":
                y_true_flat.append(grf_v[:, :, 0].detach().cpu().numpy().reshape(-1).astype(np.float64))

            imu = feature_builder.transform(imu)
            imu = norm_stats.normalize(imu)

            y_hat = model(imu)  # (B, T, D)
            if y_hat.shape != grf_v.shape:
                raise RuntimeError(
                    "Model output shape does not match GRF target shape: "
                    f"y_hat{tuple(y_hat.shape)} vs grf_v{tuple(grf_v.shape)}. "
                    f"model.type='{model_type}', grf_axes='{grf_axes}'."
                )

            # For metrics and exports we operate in the target GRF's native
            # units. If target normalization was used during training, we
            # denormalize predictions here, mirroring train_vnext behavior.
            if target_norm is not None:
                y_hat_for_metrics = target_norm.denormalize(y_hat)
            else:
                y_hat_for_metrics = y_hat

            batch_metrics = compute_grf_metrics(y_hat_for_metrics, grf_v, axes=grf_axes)

            # Accumulate metrics by summing; we'll average at the end
            for k, v in batch_metrics.to_dict().items():
                if isinstance(v, dict):
                    for ak, av in v.items():
                        key = f"{k}.{ak}"
                        metrics_accum[key] = metrics_accum.get(key, 0.0) + float(av)
                else:
                    key = k
                    metrics_accum[key] = metrics_accum.get(key, 0.0) + float(v)

            # Per-window summaries
            trial_ids: List[str] = batch["trial_id"]
            start_idxs = batch["start_idx"]

            # y_hat_for_metrics, grf_v: (B, T, D)
            # Compute mean and peak over time dimension (dim=1) in the
            # same space used for metrics.
            y_mean = y_hat_for_metrics.mean(dim=1)  # (B, D)
            y_peak, _ = y_hat_for_metrics.max(dim=1)  # (B, D)

            for i in range(y_hat.shape[0]):
                trial_id = str(trial_ids[i])
                start_idx = int(start_idxs[i])

                window_rows.append(
                    {
                        "trial_id": trial_id,
                        "start_idx": start_idx,
                    }
                )

                if args.save_preds:
                    pred_trial_ids.append(trial_id)
                    pred_start_idxs.append(start_idx)
                    if grf_axes == "fz":
                        arr_true = (
                            grf_v[i, :, 0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        arr_pred = (
                            y_hat_for_metrics[i, :, 0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    else:
                        # 3D GRF: export all axes [Fx, Fy, Fz] per window
                        arr_true = (
                            grf_v[i]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        arr_pred = (
                            y_hat_for_metrics[i]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    pred_y_true.append(arr_true)
                    pred_y_pred.append(arr_pred)

                if grf_axes == "fz":
                    # Single-axis Fz
                    window_rows[-1]["Fz_mean"] = float(y_mean[i, 0].item())
                    window_rows[-1]["Fz_peak"] = float(y_peak[i, 0].item())
                else:
                    # 3D GRF: Fx, Fy, Fz
                    window_rows[-1]["Fx_mean"] = float(y_mean[i, 0].item())
                    window_rows[-1]["Fy_mean"] = float(y_mean[i, 1].item())
                    window_rows[-1]["Fz_mean"] = float(y_mean[i, 2].item())
                    window_rows[-1]["Fx_peak"] = float(y_peak[i, 0].item())
                    window_rows[-1]["Fy_peak"] = float(y_peak[i, 1].item())
                    window_rows[-1]["Fz_peak"] = float(y_peak[i, 2].item())

            n_batches += 1

    if n_batches == 0:
        raise SystemExit("No evaluation batches with GRF were seen; cannot compute metrics.")

    if grf_axes == "fz" and y_true_flat:
        y_all = np.concatenate(y_true_flat, axis=0)
        y_med = float(np.nanmedian(y_all))
        y_p95 = float(np.nanpercentile(y_all, 95))
        logger.info(f"y_true window stats (Fz): median={y_med:.4f}, p95={y_p95:.4f}")

    # Average metrics across batches (same pattern as train_vnext)
    averaged: Dict[str, Any] = {k: v / n_batches for k, v in metrics_accum.items()}
    mse_per_axis: Dict[str, float] = {}
    rmse_per_axis: Dict[str, float] = {}
    mse_mean = float(averaged.get("mse_mean", float("nan")))
    rmse_mean = float(averaged.get("rmse_mean", float("nan")))
    for k, v in averaged.items():
        if k.startswith("mse_per_axis."):
            axis = k.split(".", 1)[1]
            mse_per_axis[axis] = float(v)
        elif k.startswith("rmse_per_axis."):
            axis = k.split(".", 1)[1]
            rmse_per_axis[axis] = float(v)

    eval_metrics = GRFMetrics(
        mse_per_axis=mse_per_axis,
        rmse_per_axis=rmse_per_axis,
        mse_mean=mse_mean,
        rmse_mean=rmse_mean,
    )

    # Prepare eval output directory under run_dir
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = eval_dir / "preds"
    if args.save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics JSON. We always write both a suffixed file
    # (eval_metrics_val.json or eval_metrics_test.json) and a legacy
    # eval_metrics.json for backward compatibility with tooling that
    # expects the unsuffixed name.
    eval_metrics_payload = {
        "model_type": model_type,
        "grf_axes": grf_axes,
        "manifest": str(eval_manifest_path),
        "checkpoint": ckpt_name,
        "metrics": eval_metrics.to_dict(),
    }
    metrics_json = json.dumps(eval_metrics_payload, indent=2)

    eval_metrics_path = eval_dir / f"eval_metrics_{metrics_suffix}.json"
    eval_metrics_path.write_text(metrics_json, encoding="utf-8")

    legacy_eval_metrics_path = eval_dir / "eval_metrics.json"
    legacy_eval_metrics_path.write_text(metrics_json, encoding="utf-8")

    # Write per-window CSV. As with metrics, we keep both suffixed and
    # legacy filenames.
    import csv

    eval_windows_path = eval_dir / f"eval_windows_{metrics_suffix}.csv"
    legacy_eval_windows_path = eval_dir / "eval_windows.csv"
    if window_rows:
        fieldnames = list(window_rows[0].keys())
        for path in (eval_windows_path, legacy_eval_windows_path):
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in window_rows:
                    writer.writerow(row)

    def _sanitize_suffix(s: str) -> str:
        s = str(s).strip()
        if not s:
            return ""
        return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s)

    if args.save_preds:
        if not pred_y_true:
            raise SystemExit(
                "--save-preds was set but no windows with GRF were collected; cannot write NPZ."
            )

        # Fz-only exports retain the original behavior and units inference.
        if grf_axes == "fz":
            inferred_target_col = None
            inferred_units = "unknown"
            try:
                import csv

                sample_grf_path = next(
                    (r.grf_path for r in base_eval.records if r.grf_path is not None),
                    None,
                )
                if sample_grf_path is not None and sample_grf_path.exists():
                    with sample_grf_path.open("r", encoding="utf-8") as f:
                        header = next(csv.reader(f))

                    if target_grf_column is not None and target_grf_column in header:
                        inferred_target_col = target_grf_column
                    else:
                        inferred_target_col = next(
                            (c for c in ("Fz_N", "Fz_BW", "Fz_%BW") if c in header),
                            None,
                        )

                    if inferred_target_col is not None:
                        if inferred_target_col.endswith("_N"):
                            inferred_units = "N"
                        elif inferred_target_col.endswith("_BW"):
                            inferred_units = "BW"
                        elif inferred_target_col.endswith("_%BW"):
                            inferred_units = "%BW"
            except Exception:
                pass

            logger.info(
                "--save-preds units note: "
                f"target_grf_column_config={target_grf_column}, "
                f"inferred_target_column={inferred_target_col}, "
                f"inferred_units={inferred_units}. "
                "GRF targets are used in native units; only IMU inputs are normalized."
            )

            suffix = _sanitize_suffix(args.preds_suffix) if args.preds_suffix is not None else ""
            fname = "fz_windows_pred_truth.npz" if not suffix else f"fz_windows_pred_truth_{suffix}.npz"
        else:
            # 3D GRF export: Fx, Fy, Fz
            suffix = _sanitize_suffix(args.preds_suffix) if args.preds_suffix is not None else ""
            fname = (
                "grf3d_windows_pred_truth.npz"
                if not suffix
                else f"grf3d_windows_pred_truth_{suffix}.npz"
            )

        out_npz = preds_dir / fname
        y_true_arr = np.stack(pred_y_true, axis=0)
        y_pred_arr = np.stack(pred_y_pred, axis=0)
        np.savez_compressed(
            out_npz,
            trial_id=np.array(pred_trial_ids, dtype=object),
            start_idx=np.array(pred_start_idxs, dtype=np.int64),
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            window_len=int(y_true_arr.shape[1]),
        )
        logger.info(f"Wrote prediction export: {out_npz}")

        if args.analyze_after_eval:
            if grf_axes == "fz":
                if args.preds_suffix is None:
                    suffix_dir = ""
                else:
                    suffix_dir = _sanitize_suffix(args.preds_suffix)
                if args.analysis_out_dir is None:
                    out_dir = run_dir / "analysis_eval"
                    if suffix_dir:
                        out_dir = out_dir / suffix_dir
                else:
                    out_dir = Path(args.analysis_out_dir)

                cmd = [
                    sys.executable,
                    "-S",
                    "scripts/analyze_fz_outputs.py",
                    "--run-dir",
                    str(run_dir),
                ]
                if args.preds_suffix is not None:
                    cmd.extend(["--preds-suffix", _sanitize_suffix(args.preds_suffix)])
                cmd.extend(["--out-dir", str(out_dir)])
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode != 0:
                    raise SystemExit(
                        "analyze_fz_outputs.py failed:\n" + (p.stdout or "") + "\n" + (p.stderr or "")
                    )
                logger.info(f"Analyzer outputs written to: {out_dir}")
            elif grf_axes == "3d":
                if args.preds_suffix is None:
                    suffix_dir = ""
                else:
                    suffix_dir = _sanitize_suffix(args.preds_suffix)
                if args.analysis_out_dir is None:
                    out_dir = run_dir / "analysis_eval"
                    if suffix_dir:
                        out_dir = out_dir / suffix_dir
                else:
                    out_dir = Path(args.analysis_out_dir)

                cmd = [
                    sys.executable,
                    "-S",
                    "scripts/analyze_3d_outputs.py",
                    "--run-dir",
                    str(run_dir),
                ]
                if args.preds_suffix is not None:
                    cmd.extend(["--preds-suffix", _sanitize_suffix(args.preds_suffix)])
                cmd.extend(["--out-dir", str(out_dir)])
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode != 0:
                    raise SystemExit(
                        "analyze_3d_outputs.py failed:\n" + (p.stdout or "") + "\n" + (p.stderr or "")
                    )
                logger.info(f"3D analyzer outputs written to: {out_dir}")
            else:
                raise SystemExit(
                    f"--analyze-after-eval is not supported for grf_axes='{grf_axes}'"
                )

    if args.analyze_after_eval and not args.save_preds:
        raise SystemExit("--analyze-after-eval requires --save-preds")

    logger.info(
        f"Evaluating run_dir={run_dir} on manifest={eval_manifest_path} "
        f"using checkpoint={ckpt_name}, model_type={model_type}, grf_axes={grf_axes}"
    )
    logger.info(
        f"Eval metrics written to: {eval_metrics_path} and {legacy_eval_metrics_path}"
    )
    logger.info(
        f"Eval window summaries written to: {eval_windows_path} and {legacy_eval_windows_path}"
    )
    logger.info(
        "Final eval RMSE_mean={:.4f}, per-axis={}".format(
            eval_metrics.rmse_mean, eval_metrics.rmse_per_axis
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
