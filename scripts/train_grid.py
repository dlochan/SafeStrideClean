# scripts/train_grid.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import pandas as pd

from src.train import train_from_arrays, _interp_grf_to_times
from src.features import rolling_features
from src.features_dual import build_dual_features


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


essential_files = ["predicted_fz.csv", str(Path("eval") / "metrics_eval.json")]


def is_completed(outdir: Path) -> bool:
    return all((outdir / p).exists() for p in essential_files)


def _subset_imu_by_tags(imu: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
    cols = ["time_s"]
    for tag in tags:
        suf = f"_{tag}"
        cols.extend([c for c in imu.columns if c.endswith(suf)])
    cols = [c for c in cols if c in imu.columns]
    return imu[cols].copy()


def _parse_models_cfg(models_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (kind, params) from either new or legacy config formats."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    # New style: top-level keys per model
    known = ["ridge", "rf", "hgb", "xgb", "stack", "cnn1d"]
    found_any = False
    for k in known:
        if k in models_cfg and isinstance(models_cfg[k], dict):
            m = models_cfg[k]
            enabled = bool(m.get("enabled", True))
            if not enabled:
                continue
            params = {kk: vv for kk, vv in m.items() if kk != "enabled"}
            out.append((k, params))
            found_any = True
    if found_any:
        return out
    # Legacy style: {models: [ {kind:..., params:{...}, enabled:bool}, ... ]}
    legacy = models_cfg.get("models", [])
    for m in legacy:
        kind = str(m.get("kind"))
        enabled = m.get("enabled", True)
        if not enabled:
            continue
        params = dict(m.get("params", {}))
        out.append((kind, params))
    return out


def main():
    ap = argparse.ArgumentParser(description="Grid trainer over window sizes and model families")
    ap.add_argument("--dataset_cfg", type=str, default="configs/dataset.yaml")
    ap.add_argument("--features_cfg", type=str, default="configs/features.yaml")
    ap.add_argument("--models_cfg", type=str, default="configs/models.yaml")
    ap.add_argument("--imu_csv", type=str, required=True, help="Path to IMU CSV")
    ap.add_argument("--grf_csv", type=str, required=True, help="Path to GRF CSV (with Fz_N)")
    ap.add_argument("--flag_csv", type=str, default=None, help="Optional activity flag CSV (unused)")
    ap.add_argument("--subject", type=str, default=None, help="Subject ID to lookup BW from dataset cfg")
    ap.add_argument("--bw_kg", type=float, default=None, help="Override body weight in kg")
    ap.add_argument("--fs", type=float, default=None, help="Override sampling frequency")
    ap.add_argument("--trial_name", type=str, default="trial", help="Name prefix for output folders")
    ap.add_argument("--out_root", type=str, default="out_grid")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--windows", type=int, nargs='*', default=None, help="Optional override for window sizes (ms)")
    args = ap.parse_args()

    dataset = read_yaml(Path(args.dataset_cfg))
    features = read_yaml(Path(args.features_cfg))
    models_cfg = read_yaml(Path(args.models_cfg))

    fs = float(args.fs) if args.fs is not None else float(dataset.get("fs_hz", 200))
    if args.bw_kg is not None:
        bw_kg = float(args.bw_kg)
    elif args.subject and "subject_masses" in dataset and args.subject in dataset["subject_masses"]:
        bw_kg = float(dataset["subject_masses"][args.subject])
    elif "default_bw_kg" in dataset:
        bw_kg = float(dataset["default_bw_kg"])
    else:
        raise SystemExit("Provide --bw_kg, or a valid --subject in dataset.yaml, or set default_bw_kg in dataset.yaml")

    imu = pd.read_csv(args.imu_csv)
    grf = pd.read_csv(args.grf_csv)

    windows = list(features.get("window_ms_grid", [200]))
    if args.windows:
        windows = [int(w) for w in args.windows]
    models_list = _parse_models_cfg(models_cfg)
    # sensor_sets can be ["lpthigh","lshank"] or [[...],[...]] for multiple sets
    ss = dataset.get("sensor_sets", ["lpthigh", "lshank"])
    if isinstance(ss, list) and ss and isinstance(ss[0], str):
        sensor_sets = [ss]  # wrap single set
    else:
        sensor_sets = ss

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    import itertools, hashlib, json as _json

    def _iter_param_combos(kind: str, params: Dict[str, Any]):
        grid = params.get("param_grid") if isinstance(params, dict) else None
        if not grid:
            yield params or {}
            return
        keys = list(grid.keys())
        vals_list = [grid[k] for k in keys]
        for combo in itertools.product(*vals_list):
            d = {k: v for k, v in zip(keys, combo)}
            # merge any base defaults outside param_grid
            base = {kk: vv for kk, vv in params.items() if kk != "param_grid"}
            base.update(d)
            yield base

    for tags in sensor_sets:
        tags = list(tags)
        imu_sub = _subset_imu_by_tags(imu, tags)
        tag_name = "-".join(tags)
        for w in windows:
            # Build features on subset (try rolling_features then dual)
            try:
                X, t = rolling_features(imu_sub, fs=fs, window_ms=int(w))
            except Exception:
                X, t = build_dual_features(imu_sub, fs=fs, window_ms=int(w))
            fz = _interp_grf_to_times(grf, t)
            y_pct = (fz.to_numpy() / (bw_kg * 9.80665)) * 100.0

            for kind, params in models_list:
                idx = 0
                for combo in _iter_param_combos(kind, params):
                    # Unique suffix by param hash
                    phash = hashlib.md5(_json.dumps(combo, sort_keys=True).encode("utf-8")).hexdigest()[:8]
                    outdir = out_root / f"{args.trial_name}_{tag_name}_{kind}_w{int(w)}_{phash}"
                    if is_completed(outdir):
                        print(f"[skip] {outdir} already complete")
                        idx += 1
                        continue
                    print(f"[run] trial={args.trial_name} tags={tag_name} kind={kind} w={w} params={combo} -> {outdir}")
                    metrics = train_from_arrays(
                        X=X,
                        y_pct=y_pct,
                        t=t,
                        bw_kg=bw_kg,
                        model_kind=kind,
                        model_params=combo,
                        outdir=outdir,
                        cv="kfold",
                        n_splits=5,
                        random_state=args.random_state,
                    )
                    print(json.dumps(metrics))
                    idx += 1


if __name__ == "__main__":
    main()
