from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from vnext.core.validation import normalize_grf_axes
from vnext.core.paths import SafeStridePaths
from vnext.core.logging_utils import get_logger
from vnext.experiments.registry import (
    VNextExperimentSuite,
    load_experiment_suite,
    deep_merge_dicts,
)


def find_latest_run_dir(out_root: Path, model_dir: str = "vnext_fz") -> Path | None:
    """Find the most recently created run directory.
    
    Parameters
    ----------
    out_root : Root output directory
    model_dir : Model subdirectory name (vnext_fz)
    
    Returns
    -------
    Path | None
        Path to the latest run directory, or None if not found
    """
    base_dir = out_root / model_dir
    if not base_dir.exists():
        return None
    
    # Get all subdirectories sorted by modification time
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def parse_output_for_run_dir(output: str) -> str | None:
    """Parse output for RUN_DIR: prefix to find run directory path.
    
    Parameters
    ----------
    output : Combined stdout/stderr from train_vnext.py
    
    Returns
    -------
    str | None
        Run directory path if found, else None
    """
    for line in output.split("\n"):
        if "RUN_DIR:" in line:
            # Extract the part after RUN_DIR:
            idx = line.index("RUN_DIR:") + 8
            return line[idx:].strip()
    return None


def run_training(
    config_path: Path,
    device: str,
    dry_run: bool,
    logger,
) -> Path | None:
    """Run train_vnext.py with the given config.
    
    Parameters
    ----------
    config_path : Path to the effective config YAML
    device : Device string (cpu/cuda)
    dry_run : If True, only log the command without running
    logger : Logger instance
    
    Returns
    -------
    Path | None
        Path to the run directory if training succeeded, else None
    """
    cmd = [
        sys.executable,
        "scripts/train_vnext.py",
        "--config", str(config_path),
        "--device", device,
    ]
    
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return None
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Try to parse RUN_DIR from output (check both stdout and stderr since logger may go to stderr)
        combined_output = result.stdout + result.stderr
        run_dir_str = parse_output_for_run_dir(combined_output)
        if run_dir_str:
            return Path(run_dir_str)
        else:
            # If we can't parse it, raise an error
            logger.error("Could not parse RUN_DIR from training output")
            raise RuntimeError("Failed to extract run directory from train_vnext.py output")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return None


def run_evaluation(
    config_path: Path,
    run_dir: Path,
    device: str,
    manifest_rel: Optional[str],
    dry_run: bool,
    logger,
) -> bool:
    """Run eval_vnext.py on the given run directory.
    
    Parameters
    ----------
    config_path : Path to the effective config YAML
    run_dir : Path to the training run directory
    device : Device string (cpu/cuda)
    manifest_rel : Optional manifest path relative to data_root
    dry_run : If True, only log the command without running
    logger : Logger instance
    
    Returns
    -------
    bool
        True if evaluation succeeded, else False
    """
    cmd = [
        sys.executable,
        "scripts/eval_vnext.py",
        "--config", str(config_path),
        "--run-dir", str(run_dir),
        "--checkpoint", "best",
        "--device", device,
    ]
    
    if manifest_rel:
        cmd.extend(["--manifest", manifest_rel])
    
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


def extract_metrics_from_run(run_dir: Path, logger) -> Dict[str, Any]:
    """Extract key metrics from a run directory.
    
    Parameters
    ----------
    run_dir : Path to the run directory
    logger : Logger instance
    
    Returns
    -------
    dict
        Dictionary with training and eval metrics
    """
    metrics = {}
    
    # Read metrics.json (training metrics)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            train_metrics = json.loads(metrics_path.read_text())
            metrics["model_type"] = train_metrics.get("model_type", "unknown")
            metrics["grf_axes"] = train_metrics.get("grf_axes", "unknown")
            metrics["best_epoch"] = train_metrics.get("best_epoch")
            metrics["best_val_rmse_mean"] = train_metrics.get("best_val_rmse_mean")
        except Exception as e:
            logger.warning(f"Failed to read {metrics_path}: {e}")
    
    # Read eval_metrics.json (if it exists)
    eval_metrics_path = run_dir / "eval" / "eval_metrics.json"
    if eval_metrics_path.exists():
        try:
            eval_data = json.loads(eval_metrics_path.read_text())
            eval_metrics = eval_data.get("metrics", {})
            
            metrics["eval_rmse_mean"] = eval_metrics.get("rmse_mean")
            
            # Per-axis RMSE
            rmse_per_axis = eval_metrics.get("rmse_per_axis", {})
            for axis, value in rmse_per_axis.items():
                metrics[f"eval_rmse_{axis}"] = value
                
        except Exception as e:
            logger.warning(f"Failed to read {eval_metrics_path}: {e}")
    
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run vNext experiment suite and build leaderboard.")
    ap.add_argument("--suite", required=True, help="Path to experiment suite YAML (e.g., configs/vnext_sweep_example.yaml)")
    ap.add_argument("--device", default="cpu", help="Device string passed to train/eval scripts (e.g., cpu, cuda)")
    ap.add_argument(
        "--eval-manifest",
        default=None,
        help="Optional manifest path (relative to data_root) to use for eval; if omitted, use data.val_manifest if present.",
    )
    ap.add_argument(
        "--out-leaderboard",
        default=None,
        help="Optional path for leaderboard CSV; if omitted, use <out_root>/vnext_leaderboard.csv",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running train/eval or writing leaderboard.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    
    logger = get_logger("run_vnext_experiments")
    
    # Load experiment suite
    logger.info(f"Loading experiment suite from: {args.suite}")
    try:
        suite = load_experiment_suite(args.suite)
    except Exception as e:
        logger.error(f"Failed to load suite: {e}")
        raise SystemExit(1)
    
    logger.info(f"Base config: {suite.base_config_path}")
    logger.info(f"Found {len(suite.experiments)} experiments")
    
    # Load base config
    base_cfg = validate_config(load_config(suite.base_config_path))
    base_paths = SafeStridePaths.from_env_or_defaults(base_cfg.get("paths", {}))
    
    # Create directory for effective configs
    config_dir = base_paths.out_root / "vnext_experiment_configs"
    if not args.dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)
    
    # Leaderboard entries
    leaderboard_entries: List[Dict[str, Any]] = []
    
    # Run each experiment
    for exp_idx, experiment in enumerate(suite.experiments, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {exp_idx}/{len(suite.experiments)}: {experiment.name}")
        logger.info(f"Tags: {experiment.tags}")
        
        # Create effective config
        effective_cfg = validate_config(deep_merge_dicts(base_cfg, experiment.overrides))

        model_cfg = effective_cfg.get("model", {}) or {}
        model_type = str(model_cfg.get("type", "fz")).lower()
        grf_axes = normalize_grf_axes(model_cfg.get("grf_axes"), model_type=model_type)
        gate_path = Path("analysis") / "FZ_TO_3D_GATE.md"
        if grf_axes == "3d" and not gate_path.exists():
            logger.warning(
                "3D GRF requested (grf_axes='3d') but gate file is missing: analysis/FZ_TO_3D_GATE.md. "
                "Per repo policy, do NOT proceed to 3D until the FZ gate is generated and explicitly authorizes it."
            )
        
        # Write effective config
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        effective_config_path = config_dir / f"{experiment.name}_{timestamp}.yaml"
        
        if not args.dry_run:
            effective_config_path.write_text(
                yaml.safe_dump(effective_cfg, sort_keys=False),
                encoding="utf-8",
            )
            logger.info(f"Wrote effective config: {effective_config_path}")
        else:
            logger.info(f"[DRY RUN] Would write effective config: {effective_config_path}")
        
        # Run training
        run_dir = run_training(
            effective_config_path,
            args.device,
            args.dry_run,
            logger,
        )
        
        if args.dry_run:
            # In dry run, report whether eval would run and with which command
            data_cfg = effective_cfg.get("data", {}) or {}
            if args.eval_manifest is not None:
                do_eval = True
                manifest_arg: Optional[str] = args.eval_manifest
            elif data_cfg.get("val_manifest") is not None:
                do_eval = True
                manifest_arg = None  # eval_vnext will read from config
            else:
                do_eval = False
                manifest_arg = None

            if do_eval:
                eval_cmd = [
                    sys.executable,
                    "scripts/eval_vnext.py",
                    "--config",
                    str(effective_config_path),
                    "--run-dir",
                    "<run_dir_from_training>",
                    "--checkpoint",
                    "best",
                    "--device",
                    args.device,
                ]
                if manifest_arg is not None:
                    eval_cmd.extend(["--manifest", manifest_arg])
                logger.info(f"[DRY RUN] Would execute eval: {' '.join(eval_cmd)}")
            else:
                logger.info(
                    f"[DRY RUN] No eval manifest available for {experiment.name}; eval would be skipped."
                )

            # In dry run, create a fake entry to show what would happen
            entry: Dict[str, Any] = {
                "name": experiment.name,
                "tags": ";".join(experiment.tags) if experiment.tags else "",
                "run_dir": f"<would-be-created-for-{experiment.name}>",
                "model_type": effective_cfg.get("model", {}).get("type", "fz"),
                "grf_axes": effective_cfg.get("model", {}).get("grf_axes", "fz"),
                "enable_kinematics": bool(effective_cfg.get("features", {}).get("enable_kinematics", False)),
                "window_size": effective_cfg.get("training", {}).get("window_size", 256),
                "window_stride": effective_cfg.get("training", {}).get("window_stride", 128),
                "batch_size": effective_cfg.get("training", {}).get("batch_size", 8),
                "epochs": effective_cfg.get("training", {}).get("epochs", 3),
                "lr": effective_cfg.get("training", {}).get("lr", 1e-3),
                "best_epoch": None,
                "best_val_rmse_mean": None,
                "eval_rmse_mean": None,
            }
            leaderboard_entries.append(entry)
            continue
        
        if run_dir is None:
            logger.error(f"Training failed for {experiment.name}, skipping")
            continue
        
        # Build leaderboard entry
        entry: Dict[str, Any] = {
            "name": experiment.name,
            "tags": ";".join(experiment.tags) if experiment.tags else "",
            "run_dir": str(run_dir),
            "model_type": effective_cfg.get("model", {}).get("type", "fz"),
            "grf_axes": effective_cfg.get("model", {}).get("grf_axes", "fz"),
            "enable_kinematics": bool(effective_cfg.get("features", {}).get("enable_kinematics", False)),
            "window_size": effective_cfg.get("training", {}).get("window_size", 256),
            "window_stride": effective_cfg.get("training", {}).get("window_stride", 128),
            "batch_size": effective_cfg.get("training", {}).get("batch_size", 8),
            "epochs": effective_cfg.get("training", {}).get("epochs", 3),
            "lr": effective_cfg.get("training", {}).get("lr", 1e-3),
            "best_epoch": None,
            "best_val_rmse_mean": None,
            "eval_rmse_mean": None,
            "eval_rmse_Fx": None,
            "eval_rmse_Fy": None,
            "eval_rmse_Fz": None,
        }
        
        # Extract training metrics
        metrics = extract_metrics_from_run(run_dir, logger)
        entry.update(metrics)
        
        # Decide if we should run evaluation
        data_cfg = effective_cfg.get("data", {}) or {}
        if args.eval_manifest is not None:
            do_eval = True
            manifest_arg = args.eval_manifest
        elif data_cfg.get("val_manifest") is not None:
            do_eval = True
            manifest_arg = None  # eval_vnext will read val_manifest from config
        else:
            do_eval = False
            manifest_arg = None

        if not do_eval:
            logger.warning(
                f"No eval manifest available for {experiment.name}; skipping eval."
            )
        else:
            if manifest_arg is not None:
                logger.info(
                    f"Running evaluation with CLI manifest override: {manifest_arg}"
                )
            else:
                logger.info(
                    f"Running evaluation using data.val_manifest from config for {experiment.name}"
                )

            eval_success = run_evaluation(
                effective_config_path,
                run_dir,
                args.device,
                manifest_arg,
                args.dry_run,
                logger,
            )

            if eval_success and not args.dry_run:
                # Re-extract metrics after eval
                metrics = extract_metrics_from_run(run_dir, logger)
                entry.update(metrics)
        
        leaderboard_entries.append(entry)
    
    # Write leaderboard files
    if not args.dry_run and leaderboard_entries:
        # Determine output path
        if args.out_leaderboard:
            leaderboard_csv_path = Path(args.out_leaderboard)
        else:
            leaderboard_csv_path = base_paths.out_root / "vnext_leaderboard.csv"
        
        leaderboard_json_path = leaderboard_csv_path.with_suffix(".json")
        
        # Sort entries by eval_rmse_mean (ascending, None values at end)
        leaderboard_entries.sort(
            key=lambda x: (x.get("eval_rmse_mean") is None, x.get("eval_rmse_mean", float("inf")))
        )
        
        # Write CSV with fixed column order
        if leaderboard_entries:
            fieldnames = [
                "name", "tags", "run_dir", "model_type", "grf_axes",
                "enable_kinematics", "window_size", "window_stride",
                "batch_size", "epochs", "lr", "best_epoch",
                "best_val_rmse_mean", "eval_rmse_mean",
                "eval_rmse_Fx", "eval_rmse_Fy", "eval_rmse_Fz"
            ]
            with leaderboard_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in leaderboard_entries:
                    # Ensure all fields are present
                    row = {k: entry.get(k) for k in fieldnames}
                    writer.writerow(row)
            logger.info(f"Wrote leaderboard CSV: {leaderboard_csv_path}")
        
        # Write JSON
        leaderboard_json_path.write_text(
            json.dumps(leaderboard_entries, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Wrote leaderboard JSON: {leaderboard_json_path}")
        
        # Rank by eval RMSE (if available)
        entries_with_eval = [e for e in leaderboard_entries if "eval_rmse_mean" in e]
        if entries_with_eval:
            entries_with_eval.sort(key=lambda x: x.get("eval_rmse_mean", float("inf")))
            logger.info("\nTop experiments by eval RMSE_mean:")
            for i, entry in enumerate(entries_with_eval[:5], 1):
                logger.info(
                    f"  {i}) {entry['name']:30s} eval_rmse_mean={entry['eval_rmse_mean']:.4f}"
                )
    elif args.dry_run:
        logger.info("\n[DRY RUN] Would have created leaderboard with entries:")
        for entry in leaderboard_entries:
            logger.info(f"  - {entry['name']}: {entry.get('model_type')}, "
                       f"kinematics={entry.get('enable_kinematics')}")
    
    logger.info("\nExperiment run completed")


if __name__ == "__main__":  # pragma: no cover
    main()
