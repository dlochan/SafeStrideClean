from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import json
import time
import uuid
import random
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

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
from vnext.data.imu_schema import (
    EXPECTED_IMU_COLUMNS,
    TIME_COL,
    get_feature_columns,
    get_sensor_slices,
)
from vnext.models.vnext_fz import VNextFzModel
from vnext.models.vnext_grf3d import VNextGRF3DModel
from vnext.feats.kinematics import KinematicFeatureBuilder, KinematicFeatureConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="SafeStride vNext stub trainer")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--device", default="cpu", help="Device string for torch (e.g. cpu or cuda)")
    ap.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    ap.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="mse",
        help="Training loss function (default: mse)",
    )
    ap.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Delta parameter for Huber loss (only used when --loss=huber)",
    )
    ap.add_argument(
        "--smooth-lambda",
        type=float,
        default=0.0,
        help="Second-difference smoothing penalty coefficient applied to predictions (default: 0.0)",
    )
    ap.add_argument(
        "--loss-window-normalize",
        action="store_true",
        help=(
            "If set, compute the data loss on y_hat and y_true after dividing each window by "
            "median(|y_true|) (per-window), to reduce scale sensitivity."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing run directory (requires --run-dir).",
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory to use when --resume is set.",
    )
    args = ap.parse_args()

    logger = get_logger("train_vnext")

    if not args.resume and args.run_dir is not None:
        logger.warning(
            "--run-dir was provided without --resume; it will be ignored and a new run directory will be created."
        )

    if args.seed is not None:
        set_seed(args.seed)

    cfg = validate_config(load_config(args.config))
    cfg_paths: Dict[str, Any] = cfg.get("paths", {}) or {}
    paths = SafeStridePaths.from_env_or_defaults(cfg_paths)

    data_cfg: Dict[str, Any] = cfg.get("data", {}) or {}

    train_manifest_rel = data_cfg.get("train_manifest")
    manifest_rel = data_cfg.get("manifest")
    split_column = data_cfg.get("split_column")

    train_manifest_path: Path | None = None
    val_manifest_path: Path | None = None

    if train_manifest_rel is not None:
        # Pattern A: explicit train/val manifests
        train_manifest_path = paths.data_root / str(train_manifest_rel)
        val_manifest_rel = data_cfg.get("val_manifest")
        if val_manifest_rel is not None:
            val_manifest_path = paths.data_root / str(val_manifest_rel)
        logger.info(f"Pattern A: train_manifest={train_manifest_path}, val_manifest={val_manifest_path}")
    elif manifest_rel is not None:
        # Pattern B: single manifest with split column
        if not split_column:
            raise SystemExit("data.split_column must be set when using data.manifest")
        import pandas as pd

        manifest_path = paths.data_root / str(manifest_rel)
        logger.info(f"Pattern B: manifest={manifest_path}, split_column={split_column}")
        if not manifest_path.exists():
            raise SystemExit(f"Combined manifest not found: {manifest_path}")
        df_all = pd.read_csv(manifest_path)
        if split_column not in df_all.columns:
            raise SystemExit(f"split_column '{split_column}' not found in manifest")

        df_train = df_all[df_all[split_column] == "train"].copy()
        df_val = df_all[df_all[split_column] == "val"].copy()

        scratch_root = paths.work_root / "vnext_splits"
        scratch_root.mkdir(parents=True, exist_ok=True)
        train_manifest_path = scratch_root / "train_manifest.csv"
        val_manifest_path = scratch_root / "val_manifest.csv"
        df_train.to_csv(train_manifest_path, index=False)
        df_val.to_csv(val_manifest_path, index=False)
        logger.info(
            f"Wrote split manifests: train={train_manifest_path} (n={len(df_train)}), "
            f"val={val_manifest_path} (n={len(df_val)})"
        )
    else:
        raise SystemExit("Config must define either data.train_manifest or data.manifest")

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

    train_cfg: Dict[str, Any] = cfg.get("training", {}) or {}
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))
    window_size = int(train_cfg.get("window_size", 256))
    window_stride = int(train_cfg.get("window_stride", 128))
    epochs = int(train_cfg.get("epochs", 3))
    lr = float(train_cfg.get("lr", 1e-3))
    require_grf = bool(train_cfg.get("require_grf", True))

    target_norm_kind = str(train_cfg.get("target_norm", "none")).lower()
    lr_scheduler_kind = str(train_cfg.get("lr_scheduler", "none")).lower()
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))

    if train_manifest_path is None:
        raise SystemExit("No train manifest path resolved")

    grf_axes = normalize_grf_axes(model_cfg.get("grf_axes"), model_type=model_type)

    gate_path = Path("analysis") / "FZ_TO_3D_GATE.md"
    if grf_axes == "3d" and not gate_path.exists():
        logger.warning(
            "3D GRF requested (grf_axes='3d') but gate file is missing: analysis/FZ_TO_3D_GATE.md. "
            "Per repo policy, do NOT proceed to 3D until the FZ gate is generated and explicitly authorizes it."
        )

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

    if len(train_ds) == 0:
        logger.warning("Train windowed dataset is empty; check window_size/stride and GRF availability.")
        raise SystemExit("No training windows available.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = None
    if val_manifest_path is not None and val_manifest_path.exists():
        base_val = DualIMUTrialDataset(
            val_manifest_path,
            grf_axes=grf_axes,
            target_grf_column=target_grf_column,
        )
        val_ds = WindowedIMUDataset(
            base_dataset=base_val,
            window_size=window_size,
            window_stride=window_stride,
            require_grf=require_grf,
        )
        if len(val_ds) == 0:
            logger.warning("Validation windowed dataset is empty; continuing without val metrics.")
        else:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(args.device)

    # Infer input channel count and per-sensor slices, then build feature transformer
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
    logger.info(f"Model: {model}")
    logger.info(f"Paths: data_root={paths.data_root}, out_root={paths.out_root}")
    logger.info(
        f"Training config: batch_size={batch_size}, window_size={window_size}, "
        f"window_stride={window_stride}, epochs={epochs}, lr={lr}, require_grf={require_grf}"
    )
    logger.info(
        f"Training loss: loss={args.loss}, huber_delta={args.huber_delta}, "
        f"smooth_lambda={args.smooth_lambda}, loss_window_normalize={args.loss_window_normalize}"
    )
    logger.info(
        f"Upgrades: backbone={backbone}, target_norm={target_norm_kind}, lr_scheduler={lr_scheduler_kind}, "
        f"grad_clip_norm={grad_clip_norm}"
    )

    # Run directory for this training job
    if args.resume:
        if args.run_dir is None:
            raise SystemExit("--resume requires --run-dir pointing to an existing run directory")
        run_dir = Path(args.run_dir)
        if not run_dir.exists() or not run_dir.is_dir():
            raise SystemExit(f"--run-dir does not exist or is not a directory: {run_dir}")
        logger.info(f"Resuming training in existing run directory: {run_dir}")
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        run_dir = paths.out_root / "vnext_fz" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Run directory: {run_dir}")

    history_csv_path = run_dir / "train_history.csv"

    # Save effective config
    try:
        import yaml

        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write config.yaml for run; continuing.")

    # Compute per-channel normalization stats on train windows (after feature transform)
    def compute_channel_norm_stats(
        dataset: WindowedIMUDataset,
        feature_builder: KinematicFeatureBuilder,
    ) -> ChannelNormStats:
        sums = None
        sq_sums = None
        count = 0
        for rec in dataset:
            x: torch.Tensor = rec["imu"]  # (T, C)
            x = x.to(torch.float32)
            x = feature_builder.transform(x)
            if sums is None:
                C = x.shape[1]
                sums = torch.zeros(C, dtype=torch.float32)
                sq_sums = torch.zeros(C, dtype=torch.float32)
            sums += x.sum(dim=0)
            sq_sums += (x * x).sum(dim=0)
            count += x.shape[0]
        if count == 0 or sums is None or sq_sums is None:
            raise RuntimeError("No samples available to compute normalization stats")
        mean = sums / count
        var = (sq_sums / count) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return ChannelNormStats(mean=mean, std=std)

    norm_stats = compute_channel_norm_stats(train_ds, feature_builder)
    # Persist normalization stats
    try:
        (run_dir / "norm_stats.json").write_text(
            json.dumps(norm_stats.to_dict(), indent=2), encoding="utf-8"
        )
    except Exception:
        logger.warning("Failed to write norm_stats.json; continuing.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if lr_scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif lr_scheduler_kind == "onecycle":
        total_steps = max(1, epochs * len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
    elif lr_scheduler_kind not in {"none", ""}:
        raise SystemExit(f"Unsupported training.lr_scheduler '{lr_scheduler_kind}'")
    if args.loss == "huber":
        criterion = torch.nn.HuberLoss(delta=float(args.huber_delta))
    else:
        criterion = torch.nn.MSELoss()

    target_norm: TargetNormStats | None = None
    if target_norm_kind not in {"none", ""}:
        if target_norm_kind == "zscore":
            mean = 0.0
            m2 = 0.0
            n = 0
            for rec in train_ds:
                y = rec.get("grf_v")
                if y is None:
                    continue
                arr = y.to(torch.float32).reshape(-1).cpu().numpy()
                for v in arr.tolist():
                    n += 1
                    delta = float(v) - mean
                    mean += delta / n
                    m2 += delta * (float(v) - mean)
            var = (m2 / max(1, n - 1)) if n > 1 else 1.0
            std = float(var) ** 0.5
            target_norm = TargetNormStats(
                kind="zscore",
                center=torch.tensor([mean], dtype=torch.float32),
                scale=torch.tensor([std], dtype=torch.float32),
            )
        elif target_norm_kind == "robust":
            rng = random.Random(args.seed if args.seed is not None else 123)
            max_points = int(train_cfg.get("target_norm_max_points", 200000))
            reservoir: List[float] = []
            seen = 0
            for rec in train_ds:
                y = rec.get("grf_v")
                if y is None:
                    continue
                arr = y.to(torch.float32).reshape(-1).cpu().numpy()
                for v in arr.tolist():
                    seen += 1
                    fv = float(v)
                    if len(reservoir) < max_points:
                        reservoir.append(fv)
                    else:
                        j = rng.randint(1, seen)
                        if j <= max_points:
                            reservoir[j - 1] = fv
            if not reservoir:
                raise SystemExit("target_norm=robust requested but no GRF samples were available")
            xs = np.asarray(reservoir, dtype=np.float32)
            center = float(np.median(xs))
            q25 = float(np.percentile(xs, 25))
            q75 = float(np.percentile(xs, 75))
            iqr = max(1e-6, q75 - q25)
            target_norm = TargetNormStats(
                kind="robust",
                center=torch.tensor([center], dtype=torch.float32),
                scale=torch.tensor([iqr], dtype=torch.float32),
            )
        else:
            raise SystemExit(f"Unsupported training.target_norm '{target_norm_kind}'")

        try:
            (run_dir / "target_norm.json").write_text(
                json.dumps(target_norm.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception:
            logger.warning("Failed to write target_norm.json; continuing.")

    if args.resume:
        ckpt_path = run_dir / "model_last.pt"
        if ckpt_path.exists():
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded model_last.pt from {ckpt_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load model_last.pt from {ckpt_path}: {e}; starting from scratch"
                )
        else:
            logger.warning(
                f"--resume was specified but no model_last.pt found in {run_dir}; starting from scratch."
            )

        opt_path = run_dir / "optimizer_last.pt"
        if opt_path.exists():
            try:
                opt_state = torch.load(opt_path, map_location=device)
                optimizer.load_state_dict(opt_state)
                logger.info(f"Loaded optimizer_last.pt from {opt_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load optimizer_last.pt from {opt_path}: {e}; continuing with fresh optimizer"
                )
        else:
            logger.warning("No optimizer_last.pt found; resuming with fresh optimizer state.")

    # History now stores one dict per epoch/split with GRFMetrics
    history: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "val": [],
    }

    # Best-validation tracking (based on lowest val RMSE_mean)
    best_val_rmse: float | None = None
    best_epoch: int | None = None

    for epoch in range(epochs):
        model.train()
        n_train_batches = 0
        train_metrics_accum: Dict[str, float] = {}
        train_loss_sum = 0.0
        train_loss_count = 0

        for batch in train_loader:
            imu = batch["imu"].to(device)  # (B, T, C_in)
            grf_v = batch["grf_v"]

            if grf_v is None:
                continue

            grf_v = grf_v.to(device)  # (B, T, D)

            # Apply feature augmentation then normalize per channel
            imu = feature_builder.transform(imu)
            imu = norm_stats.normalize(imu)

            optimizer.zero_grad()
            y_hat = model(imu)
            if y_hat.shape != grf_v.shape:
                raise RuntimeError(
                    "Model output shape does not match GRF target shape: "
                    f"y_hat{tuple(y_hat.shape)} vs grf_v{tuple(grf_v.shape)}. "
                    f"model.type='{model_type}', grf_axes='{grf_axes}'."
                )

            if target_norm is not None:
                grf_v_loss_src = target_norm.normalize(grf_v)
                y_hat_loss_src = y_hat
                y_hat_for_metrics = target_norm.denormalize(y_hat)
            else:
                grf_v_loss_src = grf_v
                y_hat_loss_src = y_hat
                y_hat_for_metrics = y_hat

            if args.loss_window_normalize:
                scale = torch.median(torch.abs(grf_v_loss_src), dim=1).values  # (B, D)
                scale = scale.unsqueeze(1)  # (B, 1, D)
                y_hat_loss = y_hat_loss_src / (scale + 1e-8)
                grf_v_loss = grf_v_loss_src / (scale + 1e-8)
            else:
                y_hat_loss = y_hat_loss_src
                grf_v_loss = grf_v_loss_src

            loss = criterion(y_hat_loss, grf_v_loss)
            if float(args.smooth_lambda) > 0.0 and y_hat_for_metrics.shape[1] >= 3:
                d2 = (
                    y_hat_for_metrics[:, 2:, :]
                    - 2.0 * y_hat_for_metrics[:, 1:-1, :]
                    + y_hat_for_metrics[:, :-2, :]
                )
                smooth_pen = torch.mean(d2 * d2)
                loss = loss + float(args.smooth_lambda) * smooth_pen
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            if scheduler is not None and lr_scheduler_kind == "onecycle":
                scheduler.step()

            train_loss_sum += float(loss.item())
            train_loss_count += 1

            # Per-axis metrics for this batch
            batch_metrics = compute_grf_metrics(y_hat_for_metrics, grf_v, axes=grf_axes)

            # Accumulate by summing; we'll average over batches later
            for k, v in batch_metrics.to_dict().items():
                # v may be dict or float; handle per-axis dicts separately
                if isinstance(v, dict):
                    for ak, av in v.items():
                        key = f"{k}.{ak}"
                        train_metrics_accum[key] = train_metrics_accum.get(key, 0.0) + float(av)
                else:
                    key = k
                    train_metrics_accum[key] = train_metrics_accum.get(key, 0.0) + float(v)

            n_train_batches += 1

        if n_train_batches == 0:
            logger.warning("No train batches with GRF were seen; check require_grf or manifests.")
            epoch_train_metrics = GRFMetrics(
                mse_per_axis={},
                rmse_per_axis={},
                mse_mean=float("nan"),
                rmse_mean=float("nan"),
            )
            epoch_train_loss = float("nan")
        else:
            # Average accumulated metrics over batches
            averaged: Dict[str, object] = {}
            for k, v in train_metrics_accum.items():
                averaged[k] = v / n_train_batches

            # Reconstruct GRFMetrics from averaged values
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
            epoch_train_metrics = GRFMetrics(
                mse_per_axis=mse_per_axis,
                rmse_per_axis=rmse_per_axis,
                mse_mean=mse_mean,
                rmse_mean=rmse_mean,
            )
            epoch_train_loss = train_loss_sum / max(train_loss_count, 1)

        history["train"].append({
            "epoch": epoch + 1,
            "split": "train",
            "metrics": epoch_train_metrics.to_dict(),
        })

        # Validation
        epoch_val_metrics: GRFMetrics | None = None
        epoch_val_loss: float | None = None
        if val_loader is not None:
            model.eval()
            n_val_batches = 0
            val_metrics_accum: Dict[str, float] = {}
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    imu = batch["imu"].to(device)
                    grf_v = batch["grf_v"]
                    if grf_v is None:
                        continue
                    grf_v = grf_v.to(device)
                    imu = feature_builder.transform(imu)
                    imu = norm_stats.normalize(imu)
                    y_hat = model(imu)
                    if y_hat.shape != grf_v.shape:
                        raise RuntimeError(
                            "Model output shape does not match GRF target shape: "
                            f"y_hat{tuple(y_hat.shape)} vs grf_v{tuple(grf_v.shape)}. "
                            f"model.type='{model_type}', grf_axes='{grf_axes}'."
                        )

                    if target_norm is not None:
                        grf_v_loss_src = target_norm.normalize(grf_v)
                        y_hat_loss_src = y_hat
                        y_hat_for_metrics = target_norm.denormalize(y_hat)
                    else:
                        grf_v_loss_src = grf_v
                        y_hat_loss_src = y_hat
                        y_hat_for_metrics = y_hat

                    if args.loss_window_normalize:
                        scale = torch.median(torch.abs(grf_v_loss_src), dim=1).values
                        scale = scale.unsqueeze(1)
                        y_hat_loss = y_hat_loss_src / (scale + 1e-8)
                        grf_v_loss = grf_v_loss_src / (scale + 1e-8)
                    else:
                        y_hat_loss = y_hat_loss_src
                        grf_v_loss = grf_v_loss_src

                    loss_v = criterion(y_hat_loss, grf_v_loss)
                    if float(args.smooth_lambda) > 0.0 and y_hat_for_metrics.shape[1] >= 3:
                        d2 = (
                            y_hat_for_metrics[:, 2:, :]
                            - 2.0 * y_hat_for_metrics[:, 1:-1, :]
                            + y_hat_for_metrics[:, :-2, :]
                        )
                        smooth_pen = torch.mean(d2 * d2)
                        loss_v = loss_v + float(args.smooth_lambda) * smooth_pen
                    val_loss_sum += float(loss_v.item())
                    val_loss_count += 1

                    batch_metrics = compute_grf_metrics(y_hat_for_metrics, grf_v, axes=grf_axes)
                    for k, v in batch_metrics.to_dict().items():
                        if isinstance(v, dict):
                            for ak, av in v.items():
                                key = f"{k}.{ak}"
                                val_metrics_accum[key] = val_metrics_accum.get(key, 0.0) + float(av)
                        else:
                            key = k
                            val_metrics_accum[key] = val_metrics_accum.get(key, 0.0) + float(v)
                    n_val_batches += 1
            if n_val_batches > 0:
                epoch_val_loss = val_loss_sum / max(val_loss_count, 1)
                averaged_val: Dict[str, object] = {}
                for k, v in val_metrics_accum.items():
                    averaged_val[k] = v / n_val_batches
                mse_per_axis_v: Dict[str, float] = {}
                rmse_per_axis_v: Dict[str, float] = {}
                mse_mean_v = float(averaged_val.get("mse_mean", float("nan")))
                rmse_mean_v = float(averaged_val.get("rmse_mean", float("nan")))
                for k, v in averaged_val.items():
                    if k.startswith("mse_per_axis."):
                        axis = k.split(".", 1)[1]
                        mse_per_axis_v[axis] = float(v)
                    elif k.startswith("rmse_per_axis."):
                        axis = k.split(".", 1)[1]
                        rmse_per_axis_v[axis] = float(v)
                epoch_val_metrics = GRFMetrics(
                    mse_per_axis=mse_per_axis_v,
                    rmse_per_axis=rmse_per_axis_v,
                    mse_mean=mse_mean_v,
                    rmse_mean=rmse_mean_v,
                )
            else:
                logger.warning("Validation loader had no usable batches; val metrics set to NaN.")

        if epoch_val_metrics is not None:
            history["val"].append({
                "epoch": epoch + 1,
                "split": "val",
                "metrics": epoch_val_metrics.to_dict(),
            })

            # Best-checkpoint logic based on lowest validation RMSE_mean
            current_rmse = float(epoch_val_metrics.rmse_mean)
            if best_val_rmse is None or current_rmse < best_val_rmse:
                best_val_rmse = current_rmse
                best_epoch = epoch + 1
                try:
                    torch.save(model.state_dict(), run_dir / "model_best.pt")
                    logger.info(
                        f"New best model at epoch {epoch+1} with val RMSE_mean={current_rmse:.4f}"
                    )
                except Exception:
                    logger.warning("Failed to save model_best.pt; continuing.")

        # Logging: summarize mean RMSE + per-axis RMSE from train (and val if present)
        def _fmt_rmse(metrics: GRFMetrics) -> str:
            parts = [f"RMSE_mean={metrics.rmse_mean:.4f}"]
            for axis, v in metrics.rmse_per_axis.items():
                parts.append(f"{axis}={v:.4f}")
            return ", ".join(parts)

        train_msg = _fmt_rmse(epoch_train_metrics)
        if epoch_val_metrics is not None:
            val_msg = _fmt_rmse(epoch_val_metrics)
            logger.info(
                f"Epoch {epoch+1}/{epochs} - train_loss={epoch_train_loss:.6f} train [{train_msg}], "
                f"val_loss={epoch_val_loss if epoch_val_loss is not None else float('nan'):.6f} val [{val_msg}]"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - train_loss={epoch_train_loss:.6f} train [{train_msg}], val [N/A]"
            )

        if scheduler is not None and lr_scheduler_kind == "cosine":
            scheduler.step()

        # Append per-epoch CSV row under the run directory
        try:
            file_exists = history_csv_path.exists()
            with history_csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "epoch",
                            "train_loss",
                            "val_loss",
                            "train_rmse_mean",
                            "train_rmse_Fx",
                            "train_rmse_Fy",
                            "train_rmse_Fz",
                            "val_rmse_mean",
                            "val_rmse_Fx",
                            "val_rmse_Fy",
                            "val_rmse_Fz",
                        ]
                    )

                def _get_axis(metrics: GRFMetrics | None, axis: str) -> float | None:
                    if metrics is None:
                        return None
                    return metrics.rmse_per_axis.get(axis)

                row = [
                    epoch + 1,
                    epoch_train_loss,
                    epoch_val_loss,
                    epoch_train_metrics.rmse_mean,
                    _get_axis(epoch_train_metrics, "Fx"),
                    _get_axis(epoch_train_metrics, "Fy"),
                    _get_axis(epoch_train_metrics, "Fz"),
                    epoch_val_metrics.rmse_mean if epoch_val_metrics is not None else None,
                    _get_axis(epoch_val_metrics, "Fx"),
                    _get_axis(epoch_val_metrics, "Fy"),
                    _get_axis(epoch_val_metrics, "Fz"),
                ]
                writer.writerow(row)
        except Exception:
            logger.warning("Failed to append to train_history.csv; continuing.")

        # Save "last" checkpoint each epoch for potential resume
        try:
            torch.save(model.state_dict(), run_dir / "model_last.pt")
            torch.save(optimizer.state_dict(), run_dir / "optimizer_last.pt")
        except Exception:
            logger.warning("Failed to save model_last/optimizer_last; continuing.")

    # Save final model and optimizer checkpoint
    try:
        torch.save(model.state_dict(), run_dir / "model_last.pt")
        torch.save(optimizer.state_dict(), run_dir / "optimizer_last.pt")
    except Exception:
        logger.warning("Failed to save model_last/optimizer_last; continuing.")

    # Save metrics history
    metrics_payload = {
        "model_type": model_type,
        "grf_axes": grf_axes,
        "epochs": epochs,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_rmse_mean": best_val_rmse,
        "best_checkpoint": "model_best.pt" if best_epoch is not None else None,
    }
    try:
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.warning("Failed to write metrics.json; continuing.")

    logger.info("vNext Fz training loop completed successfully")
    
    # Structured output for experiment runner to parse
    logger.info(f"RUN_DIR:{run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
