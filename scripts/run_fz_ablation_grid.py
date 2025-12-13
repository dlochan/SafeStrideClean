from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import shutil

import yaml

try:
    import vnext  # noqa: F401
except ModuleNotFoundError as e:
    raise SystemExit(
        "Could not import 'vnext'. Install the repo in editable mode from the repo root: "
        "`python -m pip install -e .`"
    ) from e

from vnext.core.config import load_config
from vnext.core.validation import validate_config
from vnext.core.logging_utils import get_logger


@dataclass
class Setting:
    window_size: int
    window_stride: int
    per_sensor_hidden: int
    fusion_hidden: int
    backbone: str
    loss: str
    huber_delta: float
    smooth_lambda: float
    loss_window_normalize: bool


@dataclass
class RunResult:
    setting: Setting
    run_dir: Path
    preds_suffix: str
    summary_json: Path


def _parse_run_dir(output: str) -> str | None:
    for line in output.split("\n"):
        if "RUN_DIR:" in line:
            idx = line.index("RUN_DIR:") + 8
            return line[idx:].strip()
    return None


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _write_effective_config(cfg: Dict[str, Any], out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{ts}_{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _run_cmd(cmd: List[str], logger, dry_run: bool) -> Tuple[int, str]:
    logger.info(f"Executing: {' '.join(cmd)}")
    if dry_run:
        return 0, ""
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + "\n" + p.stderr)


def _analyze_metrics(summary_json: Path) -> Tuple[float | None, float | None, float | None]:
    obj = json.loads(summary_json.read_text(encoding="utf-8"))
    wm = obj.get("window_metrics", {})
    return (
        wm.get("nrmse_bw_mean"),
        wm.get("pearson_r_mean"),
        wm.get("rmse_mean"),
    )


def _settings_grid(per_sensor_hidden: int, fusion_hidden: int) -> List[Setting]:
    out: List[Setting] = []
    for backbone in ["baseline_mlp", "tcn"]:
        for window_size in [128, 256]:
            for window_stride in [64, 128]:
                for loss in ["mse", "huber"]:
                    for smooth_lambda in [0.0, 1e-4, 1e-3]:
                        out.append(
                            Setting(
                                window_size=window_size,
                                window_stride=window_stride,
                                per_sensor_hidden=int(per_sensor_hidden),
                                fusion_hidden=int(fusion_hidden),
                                backbone=backbone,
                                loss=loss,
                                huber_delta=1.0,
                                smooth_lambda=smooth_lambda,
                                loss_window_normalize=False,
                            )
                        )
    return out


def _setting_name(s: Setting) -> str:
    return (
        f"bb{s.backbone}_ws{s.window_size}_st{s.window_stride}_ps{s.per_sensor_hidden}_fu{s.fusion_hidden}_"
        f"loss{s.loss}_sl{s.smooth_lambda:g}"
    )


def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header is not None and existing_header != fieldnames:
                ts = time.strftime("%Y%m%d_%H%M%S")
                backup = csv_path.with_suffix(csv_path.suffix + f".bak_{ts}")
                shutil.copy2(csv_path, backup)
                csv_path.write_text("", encoding="utf-8")
        except Exception:
            pass

    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _run_phase(
    phase: str,
    base_cfg: Dict[str, Any],
    settings: List[Setting],
    epochs: int,
    device: str,
    leaderboard_csv: Path,
    dry_run: bool,
    logger,
) -> List[RunResult]:
    run_results: List[RunResult] = []
    cfg_out_dir = Path("analysis") / "ablation_effective_configs"

    base_training_cfg: Dict[str, Any] = base_cfg.get("training", {}) or {}
    fixed_target_norm = str(base_training_cfg.get("target_norm", "none"))
    fixed_lr_scheduler = str(base_training_cfg.get("lr_scheduler", "none"))
    fixed_grad_clip_norm = float(base_training_cfg.get("grad_clip_norm", 0.0))

    for i, s in enumerate(settings):
        name = _setting_name(s)
        eff_cfg = validate_config(
            _deep_merge(
                base_cfg,
                {
                    "model": {
                        "per_sensor_hidden": int(s.per_sensor_hidden),
                        "fusion_hidden": int(s.fusion_hidden),
                        "backbone": str(s.backbone),
                    },
                    "training": {
                        "window_size": int(s.window_size),
                        "window_stride": int(s.window_stride),
                        "epochs": int(epochs),
                    },
                },
            )
        )
        eff_path = _write_effective_config(eff_cfg, cfg_out_dir, f"{phase}_{i:03d}_{name}")

        cmd = [
            sys.executable,
            "scripts/train_vnext.py",
            "--config",
            str(eff_path),
            "--device",
            device,
            "--loss",
            s.loss,
            "--huber-delta",
            str(s.huber_delta),
            "--smooth-lambda",
            str(s.smooth_lambda),
        ]
        if s.loss_window_normalize:
            cmd.append("--loss-window-normalize")

        rc, out = _run_cmd(cmd, logger, dry_run=dry_run)
        if rc != 0:
            logger.error(f"Training failed for {name} (rc={rc})")
            continue

        run_dir_str = _parse_run_dir(out)
        if run_dir_str is None:
            logger.error(f"Could not parse RUN_DIR for {name}")
            continue

        run_dir = Path(run_dir_str)

        preds_suffix = f"ablate_{phase}_{i:03d}"
        eval_cmd = [
            sys.executable,
            "scripts/eval_vnext.py",
            "--config",
            str(run_dir / "config.yaml"),
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best",
            "--device",
            device,
            "--save-preds",
            "--preds-suffix",
            preds_suffix,
            "--analyze-after-eval",
        ]
        eval_cmd.extend(
            [
                "--analysis-out-dir",
                str(run_dir / "analysis_eval" / preds_suffix),
            ]
        )
        rc, out = _run_cmd(eval_cmd, logger, dry_run=dry_run)
        if rc != 0:
            logger.error(f"Eval failed for {name} (rc={rc})")
            continue

        out_dir = run_dir / "analysis_eval" / preds_suffix
        summary_json = out_dir / "fz_metrics_summary.json"
        nrmse_bw_mean: float | None = None
        pearson_r_mean: float | None = None
        rmse_mean: float | None = None
        if not dry_run and summary_json.exists():
            nrmse_bw_mean, pearson_r_mean, rmse_mean = _analyze_metrics(summary_json)

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "run_dir": str(run_dir),
            "window_size": s.window_size,
            "window_stride": s.window_stride,
            "per_sensor_hidden": s.per_sensor_hidden,
            "fusion_hidden": s.fusion_hidden,
            "backbone": s.backbone,
            "loss": s.loss,
            "huber_delta": s.huber_delta,
            "smooth_lambda": s.smooth_lambda,
            "loss_window_normalize": s.loss_window_normalize,
            "target_norm": fixed_target_norm,
            "lr_scheduler": fixed_lr_scheduler,
            "grad_clip_norm": fixed_grad_clip_norm,
            "nrmse_bw_mean": nrmse_bw_mean,
            "pearson_r_mean": pearson_r_mean,
            "rmse_mean": rmse_mean,
        }
        _append_row(leaderboard_csv, row)
        run_results.append(
            RunResult(setting=s, run_dir=run_dir, preds_suffix=preds_suffix, summary_json=summary_json)
        )

    return run_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small FZ ablation grid and write a leaderboard CSV")
    ap.add_argument("--base-config", default="configs/vnext_example.yaml", help="Base config YAML")
    ap.add_argument("--device", default="cpu", help="Device string for train/eval")
    ap.add_argument("--smoke-epochs", type=int, default=1, help="Epochs for smoke phase")
    ap.add_argument("--final-epochs", type=int, default=25, help="Epochs for final phase")
    ap.add_argument("--top-k", type=int, default=3, help="Number of best smoke settings to rerun")
    ap.add_argument(
        "--leaderboard-csv",
        default="analysis/fz_ablation_leaderboard.csv",
        help="Output leaderboard CSV path",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    logger = get_logger("run_fz_ablation_grid")

    base_cfg = validate_config(load_config(args.base_config))

    model_cfg: Dict[str, Any] = base_cfg.get("model", {}) or {}
    settings = _settings_grid(
        per_sensor_hidden=int(model_cfg.get("per_sensor_hidden", 32)),
        fusion_hidden=int(model_cfg.get("fusion_hidden", 64)),
    )
    leaderboard_csv = Path(args.leaderboard_csv)

    logger.info(f"Running smoke phase ({args.smoke_epochs} epochs) with {len(settings)} settings")
    smoke_runs = _run_phase(
        phase="smoke",
        base_cfg=base_cfg,
        settings=settings,
        epochs=int(args.smoke_epochs),
        device=str(args.device),
        leaderboard_csv=leaderboard_csv,
        dry_run=bool(args.dry_run),
        logger=logger,
    )

    if args.dry_run:
        return

    scored: List[Tuple[float, float, Setting]] = []
    for rr in smoke_runs:
        if not rr.summary_json.exists():
            continue
        nrmse_bw, pearson, _ = _analyze_metrics(rr.summary_json)
        if nrmse_bw is None:
            continue
        pr = float(pearson) if pearson is not None else -1.0
        scored.append((float(nrmse_bw), -pr, rr.setting))

    scored.sort(key=lambda x: (x[0], x[1]))
    top_settings = [s for _, __, s in scored[: int(args.top_k)]]

    if not top_settings:
        logger.warning("No top settings found from smoke phase; skipping final phase")
        return

    logger.info(f"Running final phase ({args.final_epochs} epochs) with top_k={len(top_settings)}")
    _run_phase(
        phase="final",
        base_cfg=base_cfg,
        settings=top_settings,
        epochs=int(args.final_epochs),
        device=str(args.device),
        leaderboard_csv=leaderboard_csv,
        dry_run=False,
        logger=logger,
    )


if __name__ == "__main__":
    main()
