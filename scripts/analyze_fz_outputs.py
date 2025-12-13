from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-8


def _fail(msg: str) -> "NoReturn":
    raise SystemExit(msg)


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        _fail(f"Missing required file: {path}\nNext step:\n  {hint}")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        return float("nan")
    if np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> Tuple[int, float]:
    # Returns (lag, corr) where positive lag means b is delayed relative to a.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return 0, float("nan")

    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: a.size + lag]
        elif lag > 0:
            aa = a[: a.size - lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        r = _safe_corr(aa, bb)
        if not np.isfinite(r):
            continue
        if r > best[1]:
            best = (lag, r)
    if best[1] == -np.inf:
        return 0, float("nan")
    return best


@dataclass
class WindowMetrics:
    trial_id: str
    start_idx: int
    rmse: float
    mae: float
    pearson_r: float
    bias: float
    stderr: float
    nrmse: float
    nrmse_bw: float | None
    y_true_median_abs: float
    y_pred_median_abs: float


def _load_units_from_config(cfg: Dict[str, Any]) -> str | None:
    # Strict: only use units if explicitly declared in config.
    # Supported keys: model.target_units or data.target_units or metrics.target_units.
    for key_path in (
        ("model", "target_units"),
        ("data", "target_units"),
        ("metrics", "target_units"),
    ):
        cur: Any = cfg
        ok = True
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()
    return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plot_time_series_overlay(diag_dir: Path, records: List[Dict[str, Any]], title: str) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    for rec in records:
        ax.plot(rec["y_true"], color="black", alpha=0.35, linewidth=1)
        ax.plot(rec["y_pred"], color="tab:blue", alpha=0.35, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("t (samples)")
    ax.set_ylabel("Fz (units unknown)")
    fig.tight_layout()
    fig.savefig(diag_dir / "pred_vs_true_overlays.png", dpi=160)
    plt.close(fig)


def _plot_hist(diag_dir: Path, errors: np.ndarray) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(errors, bins=60, color="tab:orange", alpha=0.85)
    ax.set_title("Residual distribution (y_pred - y_true)")
    ax.set_xlabel("error")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(diag_dir / "error_hist.png", dpi=160)
    plt.close(fig)


def _plot_residuals_vs_mag(diag_dir: Path, y_true: np.ndarray, errors: np.ndarray) -> None:
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.abs(y_true), errors, s=4, alpha=0.25)
    ax.set_title("Residuals vs |truth| (heteroscedasticity check)")
    ax.set_xlabel("|y_true|")
    ax.set_ylabel("error (y_pred - y_true)")
    fig.tight_layout()
    fig.savefig(diag_dir / "residuals_vs_magnitude.png", dpi=160)
    plt.close(fig)


def _plot_per_trial_rmse(diag_dir: Path, per_trial_rmse: Dict[str, float]) -> None:
    items = sorted(per_trial_rmse.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig = plt.figure(figsize=(max(8, 0.35 * len(vals)), 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(len(vals)), vals, color="tab:red", alpha=0.8)
    ax.set_title("Per-trial RMSE (windows aggregated)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(diag_dir / "per_trial_rmse.png", dpi=160)
    plt.close(fig)


def _plot_learning_sanity(diag_dir: Path, run_dir: Path) -> None:
    # Use train_history.csv if present. Keep robust parsing.
    p = run_dir / "train_history.csv"
    if not p.exists():
        return

    import csv

    rows: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        return

    def _col(name: str) -> List[float]:
        out: List[float] = []
        for row in rows:
            v = row.get(name)
            if v is None or v == "":
                out.append(float("nan"))
            else:
                out.append(float(v))
        return out

    epochs = [int(float(r["epoch"])) for r in rows]
    train_rmse = _col("train_rmse_mean")
    val_rmse = _col("val_rmse_mean")

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, train_rmse, label="train_rmse_mean")
    ax.plot(epochs, val_rmse, label="val_rmse_mean")
    ax.set_title("Learning sanity: RMSE vs epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(diag_dir / "rmse_vs_epoch.png", dpi=160)
    plt.close(fig)


def analyze_run(run_dir: Path, out_dir: Path, preds_suffix: str | None) -> Dict[str, Any]:
    _require_file(run_dir / "config.yaml", "Re-run training to produce config.yaml")
    _require_file(run_dir / "metrics.json", "Re-run training to produce metrics.json")
    _require_file(run_dir / "train_history.csv", "Re-run training to produce train_history.csv")
    _require_file(
        run_dir / "eval" / "eval_metrics.json",
        "Run eval: python scripts/eval_vnext.py --config <CONFIG> --run-dir <RUN_DIR> --checkpoint best",
    )

    preds_name = (
        "fz_windows_pred_truth.npz"
        if not preds_suffix
        else f"fz_windows_pred_truth_{preds_suffix}.npz"
    )
    preds_npz_path = run_dir / "eval" / "preds" / preds_name
    _require_file(
        preds_npz_path,
        "Run eval with prediction export: python scripts/eval_vnext.py --config <CONFIG> --run-dir <RUN_DIR> --checkpoint best --save-preds",
    )

    from vnext.core.config import load_config

    cfg = load_config(run_dir / "config.yaml")

    target_grf_column_config: str | None = (cfg.get("model", {}) or {}).get("target_grf_column")
    inferred_target_col: str | None = None
    inferred_units = "unknown"  # N | BW | %BW | unknown
    bw_denom_newtons: float | None = None
    nrmse_bw_reason: str | None = None

    units = _load_units_from_config(cfg)
    units_label = units if units is not None else "unknown"

    manifest_rel = (cfg.get("data", {}) or {}).get("val_manifest") or (cfg.get("data", {}) or {}).get(
        "train_manifest"
    )
    data_root = (cfg.get("paths", {}) or {}).get("data_root")
    subject_available = False

    subject_map: Dict[str, str] = {}
    trial_to_grf_path: Dict[str, Path] = {}
    if manifest_rel is not None and data_root is not None:
        manifest_path = Path(str(data_root)) / str(manifest_rel)
        if manifest_path.exists():
            import csv

            with manifest_path.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                if r.fieldnames and "grf_path" in r.fieldnames:
                    for row in r:
                        tid = row.get("trial_id")
                        gp = row.get("grf_path")
                        if tid and gp:
                            trial_to_grf_path[str(tid)] = Path(str(gp))
                if r.fieldnames and "subject_id" in r.fieldnames:
                    subject_available = True
                    for row in r:
                        tid = row.get("trial_id")
                        sid = row.get("subject_id")
                        if tid and sid:
                            subject_map[str(tid)] = str(sid)

    sample_grf_path = next(iter(trial_to_grf_path.values()), None) if trial_to_grf_path else None
    if sample_grf_path is not None and sample_grf_path.exists():
        try:
            import csv

            with sample_grf_path.open("r", encoding="utf-8") as f:
                header = next(csv.reader(f))

            if target_grf_column_config is not None and target_grf_column_config in header:
                inferred_target_col = target_grf_column_config
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

            if "Fz_N" in header and "Fz_BW" in header:
                ratios: List[float] = []
                with sample_grf_path.open("r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        try:
                            fz_n = float(row["Fz_N"])
                            fz_bw = float(row["Fz_BW"])
                        except Exception:
                            continue
                        if not np.isfinite(fz_n) or not np.isfinite(fz_bw):
                            continue
                        if abs(fz_bw) < 1e-6:
                            continue
                        ratios.append(fz_n / fz_bw)
                        if len(ratios) >= 2000:
                            break
                if ratios:
                    bw_denom_newtons = float(np.median(np.asarray(ratios, dtype=np.float64)))
        except Exception:
            pass

    npz = np.load(preds_npz_path, allow_pickle=True)
    trial_id = npz["trial_id"].astype(object)
    start_idx = npz["start_idx"].astype(np.int64)
    y_true = npz["y_true"].astype(np.float64)
    y_pred = npz["y_pred"].astype(np.float64)

    if y_true.shape != y_pred.shape:
        _fail(f"Shape mismatch in NPZ: y_true{y_true.shape} vs y_pred{y_pred.shape}")

    N, T = y_true.shape
    if N == 0 or T == 0:
        _fail("NPZ contains empty arrays; cannot analyze")

    # Window-level metrics
    window_metrics: List[WindowMetrics] = []
    all_errors_flat: List[float] = []
    all_true_flat: List[float] = []

    for i in range(N):
        yt = y_true[i]
        yp = y_pred[i]
        err = yp - yt

        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        pearson_r = _safe_corr(yt, yp)
        bias = float(np.mean(err))
        stderr = float(np.std(err))

        scale = float(np.median(np.abs(yt)))
        nrmse = float(rmse / (scale + EPS))

        nrmse_bw: float | None = None
        if inferred_units == "N":
            if bw_denom_newtons is not None and bw_denom_newtons > 0:
                nrmse_bw = float(rmse / bw_denom_newtons)
            else:
                nrmse_bw_reason = "Cannot compute nrmse_bw for N targets: bw_denom_newtons unavailable."
        elif inferred_units == "BW":
            nrmse_bw = float(rmse)
        elif inferred_units == "%BW":
            nrmse_bw = float(rmse / 100.0)
        else:
            nrmse_bw_reason = "Cannot compute nrmse_bw: target units could not be inferred."

        window_metrics.append(
            WindowMetrics(
                trial_id=str(trial_id[i]),
                start_idx=int(start_idx[i]),
                rmse=rmse,
                mae=mae,
                pearson_r=pearson_r,
                bias=bias,
                stderr=stderr,
                nrmse=nrmse,
                nrmse_bw=nrmse_bw,
                y_true_median_abs=scale,
                y_pred_median_abs=float(np.median(np.abs(yp))),
            )
        )

        all_errors_flat.extend(err.tolist())
        all_true_flat.extend(yt.tolist())

    errors_flat = np.asarray(all_errors_flat, dtype=np.float64)
    true_flat = np.asarray(all_true_flat, dtype=np.float64)

    # Aggregate summaries
    rmse_vals = np.array([m.rmse for m in window_metrics], dtype=np.float64)
    mae_vals = np.array([m.mae for m in window_metrics], dtype=np.float64)
    bias_vals = np.array([m.bias for m in window_metrics], dtype=np.float64)
    stderr_vals = np.array([m.stderr for m in window_metrics], dtype=np.float64)
    nrmse_vals = np.array([m.nrmse for m in window_metrics], dtype=np.float64)

    nrmse_bw_vals = np.array(
        [m.nrmse_bw if m.nrmse_bw is not None else float("nan") for m in window_metrics],
        dtype=np.float64,
    )
    nrmse_bw_finite = nrmse_bw_vals[np.isfinite(nrmse_bw_vals)]

    pearson_vals = np.array([m.pearson_r for m in window_metrics], dtype=np.float64)
    pearson_finite = pearson_vals[np.isfinite(pearson_vals)]

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "units": units_label,
        "target_grf_column_config": target_grf_column_config,
        "target_grf_column_inferred": inferred_target_col,
        "target_units_inferred": inferred_units,
        "bw_denom_newtons": bw_denom_newtons,
        "nrmse_bw_reason": nrmse_bw_reason,
        "subject_id_available": subject_available,
        "num_windows": int(N),
        "window_len": int(T),
        "scale_median_abs_y_true": float(np.median(np.abs(true_flat))),
        "window_metrics": {
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "bias_mean": float(np.mean(bias_vals)),
            "bias_std": float(np.std(bias_vals)),
            "stderr_mean": float(np.mean(stderr_vals)),
            "stderr_std": float(np.std(stderr_vals)),
            "nrmse_mean": float(np.mean(nrmse_vals)),
            "nrmse_std": float(np.std(nrmse_vals)),
            "nrmse_bw_mean": float(np.mean(nrmse_bw_finite)) if nrmse_bw_finite.size else None,
            "nrmse_bw_std": float(np.std(nrmse_bw_finite)) if nrmse_bw_finite.size else None,
            "nrmse_bw_num_finite": int(nrmse_bw_finite.size),
            "pearson_r_mean": float(np.mean(pearson_finite)) if pearson_finite.size else None,
            "pearson_r_std": float(np.std(pearson_finite)) if pearson_finite.size else None,
            "pearson_r_num_finite": int(pearson_finite.size),
        },
    }

    # Per-trial aggregates
    per_trial: Dict[str, Dict[str, Any]] = {}
    for m in window_metrics:
        per_trial.setdefault(m.trial_id, {"rmse": [], "mae": [], "nrmse": [], "bias": [], "stderr": [], "pearson_r": []})
        per_trial[m.trial_id]["rmse"].append(m.rmse)
        per_trial[m.trial_id]["mae"].append(m.mae)
        per_trial[m.trial_id]["nrmse"].append(m.nrmse)
        per_trial[m.trial_id]["bias"].append(m.bias)
        per_trial[m.trial_id]["stderr"].append(m.stderr)
        if np.isfinite(m.pearson_r):
            per_trial[m.trial_id]["pearson_r"].append(m.pearson_r)

    per_trial_out: Dict[str, Any] = {}
    per_trial_rmse: Dict[str, float] = {}
    for tid, d in per_trial.items():
        rmse_mean = float(np.mean(d["rmse"]))
        per_trial_rmse[tid] = rmse_mean
        per_trial_out[tid] = {
            "rmse_mean": rmse_mean,
            "mae_mean": float(np.mean(d["mae"])),
            "nrmse_mean": float(np.mean(d["nrmse"])),
            "bias_mean": float(np.mean(d["bias"])),
            "stderr_mean": float(np.mean(d["stderr"])),
            "pearson_r_mean": float(np.mean(d["pearson_r"])) if d["pearson_r"] else None,
            "num_windows": int(len(d["rmse"])),
            "subject_id": subject_map.get(tid) if subject_available else None,
        }

    summary["per_trial"] = per_trial_out

    # Per-subject aggregates (strict only)
    if subject_available:
        per_subject_acc: Dict[str, List[float]] = {}
        for tid, stats in per_trial_out.items():
            sid = stats.get("subject_id")
            if sid is None:
                continue
            per_subject_acc.setdefault(str(sid), []).append(float(stats["rmse_mean"]))
        per_subject = {
            sid: {
                "rmse_mean": float(np.mean(vals)),
                "rmse_std": float(np.std(vals)),
                "num_trials": int(len(vals)),
            }
            for sid, vals in per_subject_acc.items()
        }
        summary["per_subject"] = per_subject
    else:
        summary["per_subject"] = None

    # Failure mode checks (PASS/FAIL + explanation + implicated dimension)
    checks: Dict[str, Dict[str, Any]] = {}

    # Magnitude sanity: check for NaNs/Infs and extreme ratio to truth scale
    ratio = float(np.max(np.abs(y_pred)) / (np.median(np.abs(true_flat)) + EPS))
    finite_ok = bool(np.isfinite(y_pred).all() and np.isfinite(y_true).all())
    magnitude_ok = finite_ok and ratio < 50.0
    checks["magnitude_sanity"] = {
        "status": "PASS" if magnitude_ok else "FAIL",
        "details": {
            "finite": finite_ok,
            "max_abs_pred": float(np.max(np.abs(y_pred))),
            "truth_scale_median_abs": float(np.median(np.abs(true_flat))),
            "max_pred_to_truth_scale_ratio": ratio,
            "threshold_ratio": 50.0,
        },
        "implicated": None if magnitude_ok else "normalization_or_training_instability",
        "next_step": None
        if magnitude_ok
        else "Inspect normalization (norm_stats.json) and training stability; check for exploding outputs or bad scaling.",
    }

    # Constant / near-constant predictions
    pred_std = float(np.std(y_pred))
    true_std = float(np.std(y_true))
    const_ok = pred_std > 0.01 * (true_std + EPS)
    checks["constant_predictions"] = {
        "status": "PASS" if const_ok else "FAIL",
        "details": {"pred_std": pred_std, "true_std": true_std, "ratio": pred_std / (true_std + EPS)},
        "implicated": None if const_ok else "model_or_data_signal_mismatch",
        "next_step": None
        if const_ok
        else "Check whether inputs carry signal; verify target column and normalization; inspect whether model collapses to mean.",
    }

    # Temporal lag artifacts (cross-correlation lag)
    lags: List[int] = []
    for i in range(N):
        lag, _ = _best_lag(y_true[i], y_pred[i], max_lag=min(20, T // 4))
        lags.append(int(lag))
    med_lag = float(np.median(np.abs(lags)))
    lag_ok = med_lag <= 2.0
    checks["temporal_lag"] = {
        "status": "PASS" if lag_ok else "FAIL",
        "details": {"median_abs_lag_samples": med_lag, "max_lag_checked": int(min(20, T // 4))},
        "implicated": None if lag_ok else "data_alignment_or_windowing",
        "next_step": None
        if lag_ok
        else "Check IMU/GRF alignment and any resampling; verify window indexing and time synchronization.",
    }

    # Over-smoothing: compare derivative std
    d_true = np.diff(y_true, axis=1)
    d_pred = np.diff(y_pred, axis=1)
    dt = float(np.std(d_true))
    dp = float(np.std(d_pred))
    smooth_ok = dp > 0.3 * (dt + EPS)
    checks["over_smoothing"] = {
        "status": "PASS" if smooth_ok else "FAIL",
        "details": {"std_diff_true": dt, "std_diff_pred": dp, "ratio": dp / (dt + EPS)},
        "implicated": None if smooth_ok else "model_capacity_or_loss_bias",
        "next_step": None
        if smooth_ok
        else "Inspect whether model underfits peaks; compare training/val curves; consider data augmentation/normalization issues.",
    }

    # Mode collapse across windows: low variance of window means
    win_means = np.mean(y_pred, axis=1)
    collapse_ok = float(np.std(win_means)) > 0.05 * float(np.std(np.mean(y_true, axis=1)) + EPS)
    checks["mode_collapse"] = {
        "status": "PASS" if collapse_ok else "FAIL",
        "details": {
            "std_window_mean_pred": float(np.std(win_means)),
            "std_window_mean_true": float(np.std(np.mean(y_true, axis=1))),
        },
        "implicated": None if collapse_ok else "model_or_optimization_collapse",
        "next_step": None
        if collapse_ok
        else "Inspect whether training converged to a single mode; check learning rate and loss/gradients.",
    }

    summary["failure_mode_checks"] = checks

    # Plots
    diag_dir = out_dir / "fz_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    pick_n = min(6, N)
    idxs = rng.choice(N, size=pick_n, replace=False)
    overlay_records = [
        {"y_true": y_true[i], "y_pred": y_pred[i]}  # type: ignore[misc]
        for i in idxs
    ]

    _plot_time_series_overlay(diag_dir, overlay_records, title="Random window overlays: prediction vs truth")
    _plot_hist(diag_dir, errors_flat)
    _plot_residuals_vs_mag(diag_dir, true_flat, errors_flat)
    _plot_per_trial_rmse(diag_dir, per_trial_rmse)
    _plot_learning_sanity(diag_dir, run_dir)

    # Write window CSV
    import csv

    window_csv = out_dir / "fz_metrics_window.csv"
    window_csv.parent.mkdir(parents=True, exist_ok=True)
    with window_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "start_idx",
                "rmse",
                "mae",
                "pearson_r",
                "bias",
                "stderr",
                "nrmse",
                "nrmse_bw",
                "y_true_median_abs",
                "y_pred_median_abs",
            ],
        )
        writer.writeheader()
        for m in window_metrics:
            writer.writerow(
                {
                    "trial_id": m.trial_id,
                    "start_idx": m.start_idx,
                    "rmse": m.rmse,
                    "mae": m.mae,
                    "pearson_r": m.pearson_r,
                    "bias": m.bias,
                    "stderr": m.stderr,
                    "nrmse": m.nrmse,
                    "nrmse_bw": m.nrmse_bw,
                    "y_true_median_abs": m.y_true_median_abs,
                    "y_pred_median_abs": m.y_pred_median_abs,
                }
            )

    # Write summary JSON
    _write_json(out_dir / "fz_metrics_summary.json", summary)

    # Baseline report and gate
    pass_all = all(v["status"] == "PASS" for v in checks.values())
    nrmse_mean = float(summary["window_metrics"]["nrmse_mean"])
    pearson_mean = summary["window_metrics"]["pearson_r_mean"]

    baseline_lines: List[str] = []
    baseline_lines.append("# FZ Baseline Report")
    baseline_lines.append("")
    baseline_lines.append(f"Run dir: `{run_dir}`")
    baseline_lines.append(f"Units: `{units_label}`")
    baseline_lines.append(f"Windows: N={N}, T={T}")
    baseline_lines.append("")
    baseline_lines.append("## Core metrics (window-level aggregates)")
    baseline_lines.append("")
    wm = summary["window_metrics"]
    baseline_lines.append(f"- RMSE: {wm['rmse_mean']:.4f} ± {wm['rmse_std']:.4f}")
    baseline_lines.append(f"- MAE: {wm['mae_mean']:.4f} ± {wm['mae_std']:.4f}")
    baseline_lines.append(f"- Bias: {wm['bias_mean']:.4f} ± {wm['bias_std']:.4f}")
    baseline_lines.append(f"- Std(error): {wm['stderr_mean']:.4f} ± {wm['stderr_std']:.4f}")
    baseline_lines.append(f"- nRMSE (RMSE/median(|y_true|)): {wm['nrmse_mean']:.4f} ± {wm['nrmse_std']:.4f}")
    if pearson_mean is None:
        baseline_lines.append("- Pearson r: unavailable (insufficient variance in one or more windows)")
    else:
        baseline_lines.append(
            f"- Pearson r: {wm['pearson_r_mean']:.4f} ± {wm['pearson_r_std']:.4f} (finite windows={wm['pearson_r_num_finite']})"
        )
    baseline_lines.append("")

    baseline_lines.append("## Failure mode checks")
    baseline_lines.append("")
    for name, res in checks.items():
        baseline_lines.append(f"- {name}: **{res['status']}**")
        if res["status"] != "PASS":
            baseline_lines.append(f"  - implicated: {res['implicated']}")
            baseline_lines.append(f"  - next_step: {res['next_step']}")
    baseline_lines.append("")

    baseline_lines.append("## Per-subject availability")
    baseline_lines.append("")
    baseline_lines.append(
        "- subject_id: available" if subject_available else "- subject_id: unavailable (not present in manifest)"
    )

    _write_text(out_dir / "FZ_BASELINE.md", "\n".join(baseline_lines) + "\n")

    gate_lines: List[str] = []
    gate_lines.append("# FZ to 3D GRF Gate")
    gate_lines.append("")
    gate_lines.append("This file authorizes (or blocks) work on 3D GRF.")
    gate_lines.append("")

    # Optional acceptance thresholds loaded from analysis/gates.yaml.
    # If thresholds are undefined, we MUST block (Outcome C) per policy.
    gates_path = Path("analysis") / "gates.yaml"
    thresholds: Dict[str, Any] | None = None
    if gates_path.exists():
        try:
            import yaml  # type: ignore
        except Exception:
            thresholds = None
        else:
            try:
                thresholds = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
            except Exception:
                thresholds = None

    def _get_threshold(section: str, name: str) -> float | None:
        if not isinstance(thresholds, dict):
            return None
        sec = thresholds.get(section)
        if not isinstance(sec, dict):
            return None
        if name not in sec:
            return None
        try:
            return float(sec[name])
        except Exception:
            return None

    def _evaluate(section: str) -> Tuple[bool | None, List[str]]:
        nrmse_median_max = _get_threshold(section, "nrmse_median_max")
        nrmse_bw_max = _get_threshold(section, "nrmse_bw_max")
        pearson_min = _get_threshold(section, "pearson_r_mean_min")

        if nrmse_median_max is None and nrmse_bw_max is None and pearson_min is None:
            return None, []

        fail_reasons: List[str] = []
        if nrmse_median_max is not None and nrmse_mean > nrmse_median_max:
            fail_reasons.append(
                f"nrmse_mean={nrmse_mean:.4f} > nrmse_median_max={nrmse_median_max:.4f}"
            )
        if nrmse_bw_max is not None:
            nrmse_bw_mean = summary["window_metrics"].get("nrmse_bw_mean")
            if nrmse_bw_mean is None:
                fail_reasons.append("nrmse_bw_mean is None (cannot evaluate nrmse_bw_max)")
            elif float(nrmse_bw_mean) > nrmse_bw_max:
                fail_reasons.append(
                    f"nrmse_bw_mean={float(nrmse_bw_mean):.4f} > nrmse_bw_max={nrmse_bw_max:.4f}"
                )
        if pearson_min is not None:
            if pearson_mean is None:
                fail_reasons.append("pearson_r_mean is None (undefined)")
            elif float(pearson_mean) < pearson_min:
                fail_reasons.append(
                    f"pearson_r_mean={float(pearson_mean):.4f} < pearson_r_mean_min={pearson_min:.4f}"
                )

        return (len(fail_reasons) == 0), fail_reasons

    progress_pass, progress_reasons = _evaluate("fz_acceptance_progress")
    literature_pass, literature_reasons = _evaluate("fz_acceptance_literature")
    legacy_pass, legacy_reasons = _evaluate("fz_acceptance")

    if literature_pass is None and legacy_pass is not None:
        literature_pass = legacy_pass
        literature_reasons = legacy_reasons

    any_defined = (progress_pass is not None) or (literature_pass is not None)
    if not any_defined:
        gate_lines.append("## Outcome C — BLOCKED")
        gate_lines.append("")
        gate_lines.append("Blocked because acceptance thresholds are undefined.")
        gate_lines.append("")
        gate_lines.append("Next step:")
        gate_lines.append("- Define thresholds in analysis/gates.yaml (see template) and re-run analysis.")
    else:
        gate_lines.append("## Gate results")
        gate_lines.append("")

        if progress_pass is None:
            gate_lines.append("- progress gate: UNDEFINED")
        elif progress_pass:
            gate_lines.append("- progress gate: PASS")
        else:
            gate_lines.append("- progress gate: FAIL")
            for r in progress_reasons:
                gate_lines.append(f"  - {r}")

        if literature_pass is None:
            gate_lines.append("- literature gate: UNDEFINED")
        elif literature_pass:
            gate_lines.append("- literature gate: PASS")
        else:
            gate_lines.append("- literature gate: FAIL")
            for r in literature_reasons:
                gate_lines.append(f"  - {r}")

        gate_lines.append("")

        if literature_pass is True:
            gate_lines.append("## Outcome A — 3D NOT JUSTIFIED")
            gate_lines.append("")
            gate_lines.append(
                "Evidence: literature-level FZ thresholds were met. "
                "No evidence-based requirement for dimensional expansion was established."
            )
        elif progress_pass is True:
            gate_lines.append("## Outcome B — PROGRESS")
            gate_lines.append("")
            gate_lines.append(
                "Progress-level FZ thresholds were met, but literature-level thresholds were not. "
                "Continue iterating on the FZ pipeline."
            )
        else:
            gate_lines.append("## Outcome C — BLOCKED")
            gate_lines.append("")
            gate_lines.append("Blocked because progress-level FZ acceptance thresholds were not met.")

    _write_text(out_dir / "FZ_TO_3D_GATE.md", "\n".join(gate_lines) + "\n")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SafeStride vNext FZ outputs")
    ap.add_argument("--run-dir", required=True, help="Path to completed FZ run directory")
    ap.add_argument(
        "--preds-suffix",
        type=str,
        default=None,
        help="Optional suffix used during eval --save-preds (loads fz_windows_pred_truth_<suffix>.npz)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="analysis",
        help="Directory to write analysis artifacts (default: ./analysis)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        _fail(f"run_dir does not exist or is not a directory: {run_dir}")

    out_dir = Path(args.out_dir)
    summary = analyze_run(run_dir, out_dir=out_dir, preds_suffix=args.preds_suffix)
    print(f"OK: wrote analysis outputs to {out_dir}")
    print(f"nRMSE_mean={summary['window_metrics']['nrmse_mean']:.4f}")


if __name__ == "__main__":
    main()
