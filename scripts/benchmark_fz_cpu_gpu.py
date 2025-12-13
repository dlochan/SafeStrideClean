from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, NoReturn, Tuple

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


def _extract_rmse_mean(payload: Dict[str, Any]) -> float | None:
    candidates = [
        (payload.get("metrics") or {}).get("rmse_mean") if isinstance(payload.get("metrics"), dict) else None,
        payload.get("rmse_mean"),
        (payload.get("eval") or {}).get("rmse_mean") if isinstance(payload.get("eval"), dict) else None,
    ]
    for v in candidates:
        try:
            if v is not None:
                return float(v)
        except Exception:
            continue
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark FZ train+eval on CPU vs GPU")
    ap.add_argument("--config", required=True, help="Config YAML")
    ap.add_argument("--epochs", type=int, default=1, help="Epoch override (default: 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-device", default="cpu")
    ap.add_argument("--gpu-device", default="cuda")
    ap.add_argument(
        "--save-preds",
        action="store_true",
        help="If set, run eval with --save-preds and record NPZ path",
    )
    ap.add_argument("--out-json", default="analysis/benchmarks/fz_cpu_gpu_benchmark.json")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        _fail(f"Config not found: {cfg_path}")

    train_script = Path("scripts") / "train_vnext.py"
    eval_script = Path("scripts") / "eval_vnext.py"
    if not train_script.exists() or not eval_script.exists():
        _fail("Missing scripts/train_vnext.py or scripts/eval_vnext.py")

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(base_cfg, dict):
        _fail(f"Config is not a mapping: {cfg_path}")

    base_cfg.setdefault("training", {})
    base_cfg["training"]["epochs"] = int(args.epochs)

    eff_dir = Path("analysis") / "benchmarks" / "effective_configs"
    eff_dir.mkdir(parents=True, exist_ok=True)
    eff_cfg_path = eff_dir / f"{cfg_path.stem}__epochs{args.epochs}.yaml"
    eff_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    results: Dict[str, Any] = {
        "config": str(cfg_path),
        "effective_config": str(eff_cfg_path),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
    }

    for label, device, suffix in [
        ("cpu", args.cpu_device, "cpu"),
        ("gpu", args.gpu_device, "gpu"),
    ]:
        train_cmd = [
            sys.executable,
            str(train_script),
            "--config",
            str(eff_cfg_path),
            "--device",
            device,
            "--seed",
            str(args.seed),
        ]
        rc, out, train_s = _run(train_cmd)
        if rc != 0:
            results[label] = {"status": "FAIL", "stage": "train", "device": device, "output": out}
            continue

        run_dir = _parse_run_dir(out)
        if not run_dir:
            results[label] = {"status": "FAIL", "stage": "train", "device": device, "output": out}
            continue

        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--config",
            str(eff_cfg_path),
            "--run-dir",
            run_dir,
            "--checkpoint",
            "best",
            "--device",
            device,
            "--seed",
            str(args.seed),
        ]

        npz_path = None
        if args.save_preds:
            eval_cmd += ["--save-preds", "--preds-suffix", f"bench_{suffix}"]
            npz_path = str(
                Path(run_dir) / "eval" / "preds" / f"fz_windows_pred_truth_bench_{suffix}.npz"
            )

        rc, out2, eval_s = _run(eval_cmd)
        if rc != 0:
            results[label] = {
                "status": "FAIL",
                "stage": "eval",
                "device": device,
                "run_dir": run_dir,
                "output": out2,
                "train_wall_s": train_s,
            }
            continue

        eval_metrics_path = Path(run_dir) / "eval" / "eval_metrics.json"
        if not eval_metrics_path.exists():
            results[label] = {
                "status": "FAIL",
                "stage": "eval",
                "device": device,
                "run_dir": run_dir,
                "output": out2,
                "train_wall_s": train_s,
                "eval_wall_s": eval_s,
            }
            continue

        payload = _load_json(eval_metrics_path)
        rmse = _extract_rmse_mean(payload)
        results[label] = {
            "status": "OK",
            "device": device,
            "run_dir": run_dir,
            "train_wall_s": train_s,
            "eval_wall_s": eval_s,
            "val_rmse_mean": rmse,
            "eval_metrics_path": str(eval_metrics_path),
            "pred_npz_path": npz_path,
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
