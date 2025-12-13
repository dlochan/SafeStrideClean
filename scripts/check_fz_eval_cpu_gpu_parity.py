from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, NoReturn, Tuple

import numpy as np


def _fail(msg: str) -> NoReturn:
    raise SystemExit(msg)


def _run(cmd: list[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out


def _metrics_from_npz(npz_path: Path) -> Dict[str, float]:
    if not npz_path.exists():
        _fail(
            f"Missing NPZ: {npz_path}\n"
            "Next step: ensure eval_vnext.py supports --save-preds and that you passed it."
        )
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
    ap = argparse.ArgumentParser(
        description="Check CPU vs GPU eval parity for FZ on the same run-dir + checkpoint"
    )
    ap.add_argument("--run-dir", required=True, help="Existing FZ run directory")
    ap.add_argument("--config", default=None, help="Config path (default: <run_dir>/config.yaml)")
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--cpu-device", default="cpu")
    ap.add_argument("--gpu-device", default="cuda")
    ap.add_argument("--cpu-suffix", default="cpu")
    ap.add_argument("--gpu-suffix", default="gpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--tol-rmse", type=float, default=1e-2, help="Abs tolerance on RMSE")
    ap.add_argument("--tol-mae", type=float, default=1e-2, help="Abs tolerance on MAE")
    ap.add_argument("--tol-nrmse", type=float, default=1e-3, help="Abs tolerance on nRMSE")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        _fail(f"run_dir does not exist or is not a directory: {run_dir}")

    cfg_path = Path(args.config) if args.config else (run_dir / "config.yaml")
    if not cfg_path.exists():
        _fail(f"Config not found: {cfg_path}")

    eval_script = Path("scripts") / "eval_vnext.py"
    if not eval_script.exists():
        _fail(f"Missing script: {eval_script}")

    preds_dir = run_dir / "eval" / "preds"
    cpu_npz = preds_dir / f"fz_windows_pred_truth_{args.cpu_suffix}.npz"
    gpu_npz = preds_dir / f"fz_windows_pred_truth_{args.gpu_suffix}.npz"

    cpu_cmd = [
        sys.executable,
        str(eval_script),
        "--config",
        str(cfg_path),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.cpu_device,
        "--seed",
        str(args.seed),
        "--save-preds",
        "--preds-suffix",
        args.cpu_suffix,
    ]
    rc, out = _run(cpu_cmd)
    if rc != 0:
        _fail("CPU eval failed.\nCommand:\n  " + " ".join(cpu_cmd) + "\n\nOutput:\n" + out)
    if not cpu_npz.exists():
        _fail(f"CPU NPZ not found after eval: {cpu_npz}")

    gpu_cmd = [
        sys.executable,
        str(eval_script),
        "--config",
        str(cfg_path),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.gpu_device,
        "--seed",
        str(args.seed),
        "--save-preds",
        "--preds-suffix",
        args.gpu_suffix,
    ]
    rc, out = _run(gpu_cmd)
    if rc != 0:
        _fail("GPU eval failed.\nCommand:\n  " + " ".join(gpu_cmd) + "\n\nOutput:\n" + out)
    if not gpu_npz.exists():
        _fail(f"GPU NPZ not found after eval: {gpu_npz}")

    m_cpu = _metrics_from_npz(cpu_npz)
    m_gpu = _metrics_from_npz(gpu_npz)

    diffs = {
        "rmse_abs_diff": abs(m_cpu["rmse"] - m_gpu["rmse"]),
        "mae_abs_diff": abs(m_cpu["mae"] - m_gpu["mae"]),
        "nrmse_abs_diff": abs(m_cpu["nrmse"] - m_gpu["nrmse"]),
    }

    ok = (
        diffs["rmse_abs_diff"] <= args.tol_rmse
        and diffs["mae_abs_diff"] <= args.tol_mae
        and diffs["nrmse_abs_diff"] <= args.tol_nrmse
    )

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "config": str(cfg_path),
        "checkpoint": args.checkpoint,
        "cpu": {"device": args.cpu_device, "npz": str(cpu_npz), "metrics": m_cpu},
        "gpu": {"device": args.gpu_device, "npz": str(gpu_npz), "metrics": m_gpu},
        "diffs": diffs,
        "tolerances": {"rmse": args.tol_rmse, "mae": args.tol_mae, "nrmse": args.tol_nrmse},
        "status": "PASS" if ok else "FAIL",
    }

    out_path = Path("analysis") / "eval_cpu_gpu_parity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not ok:
        _fail(
            "CPU vs GPU eval parity FAILED.\n"
            f"Wrote report: {out_path}\n"
            "Next step: investigate device-dependent behavior (nondeterminism, normalization/device ops)."
        )

    print("OK: CPU vs GPU eval parity PASSED")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
