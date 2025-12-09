import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from src.features import rolling_features, normalize_features
from src.features_dual import build_dual_features
from src.dataio import load_imu_csv, load_grf_csv, load_c3d_grf
from src.models_zoo import make_model
from src.paths import get_out_root
import yaml


def _interp_grf_to_times(grf_df: pd.DataFrame, t_feat: pd.Series) -> pd.Series:
    """
    Interpolate GRF Fz_N onto the feature time stamps t_feat (in seconds).
    Assumes grf_df has columns ['time_s','Fz_N'].
    """
    # Ensure sorted by time
    grf = grf_df.sort_values("time_s")
    # Replace NaNs in force before interpolation to avoid propagating NaNs
    fz = grf["Fz_N"].fillna(0.0).to_numpy()
    tt = grf["time_s"].to_numpy()

    # Clamp interpolation domain
    t = t_feat.to_numpy()
    t = np.clip(t, tt.min(), tt.max())

    fz_interp = np.interp(t, tt, fz)
    return pd.Series(fz_interp, index=t_feat.index, name="Fz_N")


def _read_parquet(path: Path) -> pd.DataFrame:
    last_err = None
    for engine in ("pyarrow", "fastparquet", None):
        try:
            if engine is None:
                return pd.read_parquet(path)
            return pd.read_parquet(path, engine=engine)
        except Exception as e:
            last_err = e
    raise SystemExit(f"Failed to read parquet {path}: {last_err}")


def _save_feature_importances(model, feature_names: List[str], out_path: Path):
    imp = None
    est = model
    # If Pipeline, take final step named 'model' if present
    try:
        if hasattr(model, "named_steps"):
            est = model.named_steps.get("model", model)
    except Exception:
        est = model

    if hasattr(est, "feature_importances_"):
        try:
            vals = np.asarray(est.feature_importances_).tolist()
            imp = {name: float(vals[i]) for i, name in enumerate(feature_names)}
        except Exception:
            imp = None
    elif hasattr(est, "coef_"):
        try:
            vals = np.asarray(getattr(est, "coef_")).reshape(-1)
            vals = np.abs(vals)  # magnitude as importance
            imp = {name: float(vals[i]) for i, name in enumerate(feature_names)}
        except Exception:
            imp = None

    if imp is not None:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(imp, f, indent=2)
        except Exception:
            pass


def train_from_arrays(
    X: pd.DataFrame | np.ndarray,
    y_pct: np.ndarray,
    t: Optional[pd.Series],
    bw_kg: float,
    model_kind: str,
    model_params: Optional[Dict] = None,
    outdir: Path | str = "out",
    cv: str = "kfold",
    n_splits: int = 5,
    random_state: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure array types
    X_df: Optional[pd.DataFrame] = None
    if isinstance(X, pd.DataFrame):
        X_df = X
        X_np = X.values.astype(float)
        feat_names = list(X.columns)
    else:
        X_np = np.asarray(X, dtype=float)
        feat_names = [f"f{i}" for i in range(X_np.shape[1])]
    y_pct = np.asarray(y_pct, dtype=float).reshape(-1)

    # CV evaluation
    if cv == "subject":
        if groups is None and X_df is not None and "subject" in X_df.columns:
            groups = X_df["subject"].to_numpy()
            # Drop subject from features if present
            X_df = X_df.drop(columns=["subject"])  # keep time_s handling below
            X_np = X_df.values.astype(float)
            feat_names = list(X_df.columns)
        if groups is None:
            # Fallback to KFold if no groups provided
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            splitter = GroupKFold(n_splits=n_splits)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses: List[float] = []
    cors: List[float] = []
    maesN: List[float] = []
    bwN = float(bw_kg) * 9.80665
    for split in splitter.split(X_np, y_pct, groups):
        tr, va = split
        model = make_model(model_kind, params=model_params, random_state=random_state)
        # sample weights emphasize peaks in %BW
        w = np.clip(np.abs(y_pct[tr]) ** 1.5, 1.0, None)
        fitted = False
        try:
            model.fit(X_np[tr], y_pct[tr], model__sample_weight=w)
            fitted = True
        except Exception:
            pass
        if not fitted:
            try:
                model.fit(X_np[tr], y_pct[tr], sample_weight=w)
            except Exception:
                model.fit(X_np[tr], y_pct[tr])

        yv_pred_pct = np.asarray(model.predict(X_np[va])).reshape(-1)
        yv_true_pct = y_pct[va]
        rmse_pct = float(np.sqrt(mean_squared_error(yv_true_pct, yv_pred_pct)))
        r = float(np.corrcoef(yv_true_pct, yv_pred_pct)[0, 1]) if np.std(yv_true_pct) > 1e-12 and np.std(yv_pred_pct) > 1e-12 else float("nan")
        mae_N = float(mean_absolute_error(yv_true_pct * bwN / 100.0, yv_pred_pct * bwN / 100.0))
        rmses.append(rmse_pct)
        cors.append(r)
        maesN.append(mae_N)

    # Fit on all data
    final_model = make_model(model_kind, params=model_params, random_state=random_state)
    try:
        w_all = np.clip(np.abs(y_pct) ** 1.5, 1.0, None)
        final_model.fit(X_np, y_pct, model__sample_weight=w_all)
    except Exception:
        try:
            final_model.fit(X_np, y_pct, sample_weight=w_all)
        except Exception:
            final_model.fit(X_np, y_pct)

    # Save model
    joblib.dump(final_model, outdir / "model.pkl")

    # Predict for all windows
    y_pred_pct = np.asarray(final_model.predict(X_np)).reshape(-1)
    y_pred_N = y_pred_pct * bwN / 100.0
    pred_df = pd.DataFrame({
        "Fz_%BW": y_pred_pct,
        "Fz_N": y_pred_N,
    })
    if t is not None:
        pred_df.insert(0, "time_s", t.to_numpy())
    pred_df.to_csv(outdir / "predicted_fz.csv", index=False)

    # Metrics
    metrics = {
        "cv": cv,
        "n_splits": int(n_splits),
        "rmse_pctbw_mean": float(np.nanmean(rmses) if rmses else float("nan")),
        "rmse_pctbw_std": float(np.nanstd(rmses) if rmses else float("nan")),
        "r_mean": float(np.nanmean(cors) if cors else float("nan")),
        "r_std": float(np.nanstd(cors) if cors else float("nan")),
        "mae_N_mean": float(np.nanmean(maesN) if maesN else float("nan")),
        "mae_N_std": float(np.nanstd(maesN) if maesN else float("nan")),
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Also write eval/metrics_eval.json for resume-safe grids
    (outdir / "eval").mkdir(parents=True, exist_ok=True)
    with open(outdir / "eval" / "metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Feature importances (if any)
    _save_feature_importances(final_model, feat_names, outdir / "feature_importances.json")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train GRF model from features parquet or raw CSVs.")
    # Option A: Precomputed features
    parser.add_argument("--features_X", type=str, default=None, help="Path to X parquet")
    parser.add_argument("--features_y", type=str, default=None, help="Path to y parquet (expects Fz_N column)")
    # Option B: Raw inputs -> features
    parser.add_argument("--imu_csv", type=str, default=None, help="Path to IMU CSV (single or dual-sensor)")
    parser.add_argument("--grf_path", type=str, default=None, help="Path to GRF file (CSV or C3D)")
    parser.add_argument("--grf_type", choices=["c3d", "csv"], default="csv")
    parser.add_argument("--fs", type=float, default=200.0, help="Sampling Hz for features")
    parser.add_argument("--window_ms", type=int, default=200)
    # Common
    parser.add_argument("--bw_kg", type=float, default=None, help="Body weight in kg (if omitted, will lookup from dataset config)")
    parser.add_argument("--subject", type=str, default=None, help="Subject ID for BW lookup (e.g., AB01)")
    parser.add_argument("--dataset_cfg", type=str, default="configs/dataset.yaml", help="Path to dataset YAML for BW lookup")
    parser.add_argument("--model_kind", type=str, default="ridge",
                        choices=["ridge", "rf", "hgb", "xgb", "cnn1d", "stack"]) 
    parser.add_argument("--model_params_json", type=str, default=None, help="Path to JSON of model params")
    parser.add_argument("--cv", choices=["kfold", "subject"], default="kfold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file to tee stdout/stderr")
    args = parser.parse_args()

    # Optional tee logging
    class _Tee:
        def __init__(self, file_path: str):
            self._f = open(file_path, "a", encoding="utf-8")
            self._stdout = sys.stdout
            self._stderr = sys.stderr
        def write(self, s: str):
            try:
                self._f.write(s)
            except Exception:
                pass
            return self._stdout.write(s)
        def flush(self):
            try:
                self._f.flush()
            except Exception:
                pass
            return self._stdout.flush()
        def close(self):
            try:
                self._f.close()
            except Exception:
                pass

    tee_obj = None
    if args.log_file:
        try:
            Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
            tee_obj = _Tee(args.log_file)
            sys.stdout = tee_obj  # type: ignore
            sys.stderr = tee_obj  # type: ignore
        except Exception:
            pass

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load params
    params = None
    if args.model_params_json:
        with open(args.model_params_json, "r", encoding="utf-8") as f:
            params = json.load(f)

    # Load features/labels
    t_series: Optional[pd.Series] = None
    if args.features_X and args.features_y:
        X = _read_parquet(Path(args.features_X))
        y_df = _read_parquet(Path(args.features_y))
        if "Fz_N" not in y_df.columns:
            # pick Fz_N if present else first numeric col
            raise SystemExit("features_y parquet must include 'Fz_N' column")
        y_N = y_df["Fz_N"].to_numpy()
        bwN = args.bw_kg * 9.80665
        y_pct = (y_N / bwN) * 100.0
        # If features parquet contains 'time_s', keep it as t
        if "time_s" in X.columns:
            t_series = pd.Series(X["time_s"].to_numpy(), name="time_s")
            if "time_s" in X.columns:
                X = X.drop(columns=["time_s"])  # ensure only features
    else:
        if not (args.imu_csv and args.grf_path):
            raise SystemExit("Provide --features_X/--features_y or --imu_csv/--grf_path")
        imu_df = load_imu_csv(args.imu_csv)
        if args.grf_type == "csv":
            grf_df = load_grf_csv(args.grf_path)
        else:
            grf_df = load_c3d_grf(args.grf_path)

        # Build features (single-sensor fallback)
        try:
            X, t_feat = rolling_features(imu_df, fs=args.fs, window_ms=args.window_ms)
        except Exception:
            # Try dual-sensor builder
            X, t_feat = build_dual_features(imu_df, fs=args.fs, window_ms=args.window_ms)
        t_series = t_feat
        # Align GRF to window centers
        fz_at_feat = _interp_grf_to_times(grf_df, t_feat)
        bwN = args.bw_kg * 9.80665
        y_pct = (fz_at_feat.to_numpy() / bwN) * 100.0

    # Determine BW if not given
    bw_val = args.bw_kg
    if bw_val is None:
        # Try dataset cfg
        try:
            with open(args.dataset_cfg, "r", encoding="utf-8") as f:
                dcfg = yaml.safe_load(f) or {}
            subj = (args.subject or "").strip()
            if subj and isinstance(dcfg.get("subject_masses"), dict) and subj in dcfg["subject_masses"]:
                bw_val = float(dcfg["subject_masses"][subj])
            elif dcfg.get("default_bw_kg") is not None:
                bw_val = float(dcfg.get("default_bw_kg"))
        except Exception:
            bw_val = None
        if bw_val is None:
            raise SystemExit("Provide --bw_kg or ensure dataset config has subject_masses/default_bw_kg")
    subj_info = args.subject or "unknown"
    print(f"[INFO] Using BW_KG={bw_val} for subject={subj_info}")

    metrics = train_from_arrays(
        X=X,
        y_pct=y_pct,
        t=t_series,
        bw_kg=float(bw_val),
        model_kind=args.model_kind,
        model_params=params,
        outdir=outdir,
        cv=args.cv,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))

    # Restore std streams if tee was used
    if tee_obj is not None:
        try:
            sys.stdout = tee_obj._stdout  # type: ignore
            sys.stderr = tee_obj._stderr  # type: ignore
            tee_obj.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()