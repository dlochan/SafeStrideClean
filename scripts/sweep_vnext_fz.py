from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError as e:  # pragma: no cover - defensive
    raise SystemExit(
        "PyYAML is required to run sweep_vnext_fz.py. Install with: python -m pip install pyyaml"
    ) from e


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_base_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Expected mapping at top-level of config {path}, got {type(cfg)!r}")
    return cfg


def _build_sweep_grid(base_cfg: Dict[str, Any], max_runs: int) -> List[Dict[str, Any]]:
    training = base_cfg.get("training", {}) or {}
    model = base_cfg.get("model", {}) or {}

    base_backbone = str(model.get("backbone", "baseline_mlp")).lower()

    lr_grid = [1e-3, 3e-4]
    window_size_grid = [256, 512]
    target_norm_grid = ["zscore", "none"]  # both are supported by train_vnext

    backbones: List[str] = []
    if base_backbone:
        backbones.append(base_backbone)
    if "baseline_mlp" not in backbones:
        backbones.append("baseline_mlp")

    combos: List[Dict[str, Any]] = []
    for lr in lr_grid:
        for ws in window_size_grid:
            for tn in target_norm_grid:
                for bb in backbones:
                    combos.append(
                        {
                            "lr": float(lr),
                            "window_size": int(ws),
                            "target_norm": str(tn),
                            "backbone": str(bb),
                        }
                    )

    return combos[:max_runs]


def _resolve_out_root(base_cfg: Dict[str, Any]) -> Path:
    paths_cfg = base_cfg.get("paths", {}) or {}
    out_root = paths_cfg.get("out_root", "data/vnext_gt_real_out")
    return (REPO_ROOT / str(out_root)).resolve()


def _latest_run_dir(out_root: Path) -> Path:
    fz_root = out_root / "vnext_fz"
    if not fz_root.exists():
        raise SystemExit(f"Run root does not exist yet: {fz_root}")
    candidates = [p for p in fz_root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No run directories found under {fz_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_train_and_eval(
    idx: int,
    total: int,
    base_cfg: Dict[str, Any],
    sweep_params: Dict[str, Any],
    tmp_cfg_path: Path,
    device: str,
    out_root: Path,
) -> Tuple[str, Dict[str, Any]]:
    print(f"========== RUN {idx}/{total} ==========")
    print(f"params: lr={sweep_params['lr']}, window_size={sweep_params['window_size']}, "
          f"target_norm={sweep_params['target_norm']}, backbone={sweep_params['backbone']}")
    sys.stdout.flush()

    cfg = copy.deepcopy(base_cfg)

    train_cfg = cfg.setdefault("training", {}) or {}
    model_cfg = cfg.setdefault("model", {}) or {}

    train_cfg["lr"] = float(sweep_params["lr"])
    train_cfg["window_size"] = int(sweep_params["window_size"])
    # Keep window_stride as-is from base config; record it for the leaderboard
    train_cfg.setdefault("window_stride", train_cfg.get("window_stride", 128))
    train_cfg["target_norm"] = str(sweep_params["target_norm"])

    model_cfg["backbone"] = str(sweep_params["backbone"])

    # Write a single temporary config file that is reused across runs
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Train
    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_vnext.py"),
        "--config",
        str(tmp_cfg_path),
        "--device",
        device,
    ]
    print("[sweep] Running train:", " ".join(train_cmd))
    sys.stdout.flush()
    subprocess.run(train_cmd, check=True, cwd=str(REPO_ROOT))

    run_dir = _latest_run_dir(out_root)
    print(f"[sweep] Latest run_dir: {run_dir}")

    # Eval + analysis
    eval_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_vnext.py"),
        "--config",
        str(run_dir / "config.yaml"),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        "best",
        "--save-preds",
        "--device",
        device,
        "--analyze-after-eval",
        "--analysis-out-dir",
        str(run_dir / "analysis"),
    ]
    print("[sweep] Running eval+analyze:", " ".join(eval_cmd))
    sys.stdout.flush()
    subprocess.run(eval_cmd, check=True, cwd=str(REPO_ROOT))

    # Parse metrics
    eval_metrics_path = run_dir / "eval" / "eval_metrics_val.json"
    analysis_path = run_dir / "analysis" / "fz_metrics_summary.json"

    if not eval_metrics_path.exists():
        raise SystemExit(f"Missing eval metrics JSON: {eval_metrics_path}")
    if not analysis_path.exists():
        raise SystemExit(f"Missing analysis summary JSON: {analysis_path}")

    with eval_metrics_path.open("r", encoding="utf-8") as f:
        eval_payload = json.load(f)
    metrics = (eval_payload.get("metrics") or {})
    val_rmse_mean = float(metrics.get("rmse_mean"))

    with analysis_path.open("r", encoding="utf-8") as f:
        analysis_payload = json.load(f)
    window_metrics = analysis_payload.get("window_metrics") or {}
    nrmse_mean = window_metrics.get("nrmse_mean")
    if nrmse_mean is None:
        nrmse_mean = window_metrics.get("nrmse_bw_mean")
    nrmse_mean_val = float(nrmse_mean) if nrmse_mean is not None else None

    batch_size = int(train_cfg.get("batch_size", 0))
    window_size = int(train_cfg.get("window_size", 0))
    window_stride = int(train_cfg.get("window_stride", 0))
    target_norm = str(train_cfg.get("target_norm", ""))
    backbone = str(model_cfg.get("backbone", ""))

    row = {
        "run_dir": str(run_dir),
        "lr": float(train_cfg.get("lr", 0.0)),
        "batch_size": batch_size,
        "window_size": window_size,
        "window_stride": window_stride,
        "target_norm": target_norm,
        "backbone": backbone,
        "val_rmse_mean": val_rmse_mean,
        "nrmse_mean": nrmse_mean_val,
    }

    print(
        "[sweep] Run complete:",
        f"val_rmse_mean={val_rmse_mean:.4f}, nrmse_mean="
        + ("NA" if nrmse_mean_val is None else f"{nrmse_mean_val:.4f}"),
    )
    sys.stdout.flush()

    return str(run_dir), row


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SafeStride vNext Fz sweep runner (CPU by default)")
    ap.add_argument(
        "--config-base",
        required=True,
        help="Base YAML config to start from (e.g. configs/vnext_example.yaml)",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="Torch device string for train/eval (default: cpu)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=8,
        help="Maximum number of sweep runs to execute (default: 8)",
    )
    ap.add_argument(
        "--out-csv",
        default="data/vnext_gt_real_out/vnext_fz_sweep_results.csv",
        help="Output CSV path for sweep leaderboard (default: data/vnext_gt_real_out/vnext_fz_sweep_results.csv)",
    )
    args = ap.parse_args(argv)

    base_cfg_path = (REPO_ROOT / args.config_base).resolve()
    if not base_cfg_path.exists():
        raise SystemExit(f"Base config not found: {base_cfg_path}")

    base_cfg = _load_base_config(base_cfg_path)
    out_root = _resolve_out_root(base_cfg)

    grid = _build_sweep_grid(base_cfg, max_runs=max(1, args.runs))
    if not grid:
        raise SystemExit("Sweep grid is empty; nothing to run")

    print(f"[sweep] Using base config: {base_cfg_path}")
    print(f"[sweep] Out root: {out_root}")
    print(f"[sweep] Total planned runs: {len(grid)}")
    sys.stdout.flush()

    # Place temporary sweep config under out_root so it lives alongside
    # other experiment artifacts and remains ignored by git.
    tmp_cfg_path = out_root / "vnext_fz_sweep_tmp.yaml"

    out_csv_path = (REPO_ROOT / args.out_csv).resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for idx, params in enumerate(grid, start=1):
        run_dir_str, row = _run_train_and_eval(
            idx=idx,
            total=len(grid),
            base_cfg=base_cfg,
            sweep_params=params,
            tmp_cfg_path=tmp_cfg_path,
            device=args.device,
            out_root=out_root,
        )
        rows.append(row)

    # Write leaderboard CSV
    fieldnames = [
        "run_dir",
        "lr",
        "batch_size",
        "window_size",
        "window_stride",
        "target_norm",
        "backbone",
        "val_rmse_mean",
        "nrmse_mean",
    ]

    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Print top-3 by val_rmse_mean ascending
    sorted_rows = sorted(rows, key=lambda r: float(r["val_rmse_mean"]))
    print("========== TOP 3 BY val_rmse_mean ==========")
    for row in sorted_rows[:3]:
        nrmse = row["nrmse_mean"]
        nrmse_str = "NA" if nrmse is None else f"{float(nrmse):.4f}"
        print(
            f"run_dir={row['run_dir']} | lr={row['lr']:.1e} | "
            f"ws={row['window_size']} | stride={row['window_stride']} | "
            f"target_norm={row['target_norm']} | backbone={row['backbone']} | "
            f"val_rmse_mean={float(row['val_rmse_mean']):.4f} | nrmse_mean={nrmse_str}"
        )

    print(f"[sweep] Results written to: {out_csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
