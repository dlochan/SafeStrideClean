from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Tuple

import numpy as np
import yaml


def _fail(msg: str) -> NoReturn:
    raise SystemExit(msg)


def _run(cmd: list[str]) -> Tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out, dt


def _parse_run_dir(output: str) -> str | None:
    for line in output.splitlines():
        if "RUN_DIR:" in line:
            return line.split("RUN_DIR:", 1)[1].strip()
    return None


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _metrics_from_npz(npz_path: Path) -> Dict[str, float]:
    npz = np.load(npz_path, allow_pickle=True)
    y_true = npz["y_true"].astype(np.float64)
    y_pred = npz["y_pred"].astype(np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    scale = float(np.median(np.abs(y_true)))
    nrmse = float(rmse / (scale + 1e-8))
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "scale": scale}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FZ baseline leaderboard across 3 configs (config-based)")
    ap.add_argument("--configs", nargs="+", required=True, help="Three config YAML paths")
    ap.add_argument("--labels", nargs="+", default=None, help="Optional labels (default: small deep long)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument(
        "--save-preds",
        action="store_true",
        help="If set, uses --save-preds and computes metrics from NPZ",
    )

    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument("--window-stride", type=int, default=None)

    ap.add_argument("--out-root", default="analysis/fz_baselines", help="Root folder for per-run outputs")
    args = ap.parse_args()

    if len(args.configs) != 3:
        _fail(f"Expected exactly 3 configs; got {len(args.configs)}")

    labels = args.labels if args.labels is not None else ["small", "deep", "long"]
    if len(labels) != 3:
        _fail(f"Expected exactly 3 labels; got {len(labels)}")

    train_script = Path("scripts") / "train_vnext.py"
    eval_script = Path("scripts") / "eval_vnext.py"
    if not train_script.exists() or not eval_script.exists():
        _fail("Missing scripts/train_vnext.py or scripts/eval_vnext.py")

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    leaderboard_rows: List[Dict[str, Any]] = []
    stamp = time.strftime("%Y%m%d-%H%M%S")

    for cfg_str, label in zip(args.configs, labels):
        cfg_path = Path(cfg_str)
        if not cfg_path.exists():
            _fail(f"Config not found: {cfg_path}")

        run_folder = root / label / stamp
        run_folder.mkdir(parents=True, exist_ok=True)

        base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(base_cfg, dict):
            _fail(f"Config is not a mapping: {cfg_path}")

        overrides: Dict[str, Any] = {"training": {}}
        if args.batch_size is not None:
            overrides["training"]["batch_size"] = int(args.batch_size)
        if args.num_workers is not None:
            overrides["training"]["num_workers"] = int(args.num_workers)
        if args.window_size is not None:
            overrides["training"]["window_size"] = int(args.window_size)
        if args.window_stride is not None:
            overrides["training"]["window_stride"] = int(args.window_stride)
        if overrides["training"] == {}:
            overrides.pop("training", None)

        effective_cfg = _deep_merge(base_cfg, overrides)

        effective_cfg_path = run_folder / "effective_config.yaml"
        effective_cfg_path.write_text(yaml.safe_dump(effective_cfg, sort_keys=False), encoding="utf-8")

        train_cmd = [
            sys.executable,
            str(train_script),
            "--config",
            str(effective_cfg_path),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        rc, out, train_s = _run(train_cmd)
        (run_folder / "train_output.txt").write_text(out, encoding="utf-8")
        if rc != 0:
            _fail(f"Train failed for label={label}.\nSee: {run_folder / 'train_output.txt'}")

        run_dir = _parse_run_dir(out)
        if not run_dir:
            _fail(f"Could not parse RUN_DIR for label={label}.\nSee: {run_folder / 'train_output.txt'}")

        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--config",
            str(effective_cfg_path),
            "--run-dir",
            run_dir,
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]

        npz_path = None
        metrics = None
        if args.save_preds:
            suffix = f"leaderboard_{label}_{stamp}"
            eval_cmd += ["--save-preds", "--preds-suffix", suffix]
            npz_path = Path(run_dir) / "eval" / "preds" / f"fz_windows_pred_truth_{suffix}.npz"

        rc, out2, eval_s = _run(eval_cmd)
        (run_folder / "eval_output.txt").write_text(out2, encoding="utf-8")
        if rc != 0:
            _fail(f"Eval failed for label={label}.\nSee: {run_folder / 'eval_output.txt'}")

        eval_dir = Path(run_dir) / "eval"
        if eval_dir.exists():
            for p in [eval_dir / "eval_metrics.json", eval_dir / "eval_windows.csv"]:
                if p.exists():
                    (run_folder / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

        if npz_path is not None:
            if not npz_path.exists():
                _fail(f"--save-preds was set but NPZ not found: {npz_path}")
            (run_folder / npz_path.name).write_bytes(npz_path.read_bytes())
            metrics = _metrics_from_npz(run_folder / npz_path.name)

        row: Dict[str, Any] = {
            "label": label,
            "config": str(cfg_path),
            "effective_config": str(effective_cfg_path),
            "device": args.device,
            "run_dir": run_dir,
            "train_wall_s": train_s,
            "eval_wall_s": eval_s,
            "checkpoint": args.checkpoint,
            "pred_npz": str(run_folder / npz_path.name) if npz_path is not None else None,
        }
        if metrics is not None:
            row.update(
                {
                    "rmse_mean": metrics["rmse"],
                    "mae_mean": metrics["mae"],
                    "nrmse_mean": metrics["nrmse"],
                }
            )

        (run_folder / "result.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        leaderboard_rows.append(row)

    boards_dir = root / "leaderboards"
    boards_dir.mkdir(parents=True, exist_ok=True)
    out_csv = boards_dir / f"fz_baseline_leaderboard_{stamp}.csv"
    out_json = boards_dir / f"fz_baseline_leaderboard_{stamp}.json"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in leaderboard_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in leaderboard_rows:
            w.writerow(r)

    out_json.write_text(json.dumps(leaderboard_rows, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_csv}")
    print(f"OK: wrote {out_json}")


if __name__ == "__main__":
    main()
