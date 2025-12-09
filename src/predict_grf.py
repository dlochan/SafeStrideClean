import argparse
import pandas as pd
import numpy as np
from src.dataio import load_imu_csv
from src.features import rolling_features, _find_sensor_sets
from src.models import load_model


def main():
    parser = argparse.ArgumentParser(description='Predict GRF using a trained model and IMU data.')
    parser.add_argument('--imu_csv', required=True, help='Path to the IMU CSV file')
    parser.add_argument('--model_pkl', required=True, help='Path to the trained model file (pkl)')
    parser.add_argument('--fs_hint', type=float, default=100.0, help='Sampling frequency hint')
    parser.add_argument('--window_ms', type=int, default=200, help='Window size in milliseconds')
    parser.add_argument('--bw_kg', type=float, required=True, help='Body weight in kilograms')
    parser.add_argument('--out_csv', default='out/predicted_grf.csv', help='Output CSV file path')

    args = parser.parse_args()

    # Load IMU data and normalize time_s
    imu_df = load_imu_csv(args.imu_csv)
    imu_df['time_s'] = pd.to_numeric(imu_df['time_s'], errors='coerce')
    base = ["ax","ay","az","gx","gy","gz"]
    # Discover sensor sets (single or multi)
    sets = _find_sensor_sets(list(imu_df.columns))

    # Load model early to determine expected feature dimensionality
    model = load_model(args.model_pkl)
    n_expected = None
    try:
        scaler = None
        if hasattr(model, 'named_steps') and isinstance(model.named_steps, dict):
            scaler = model.named_steps.get('scaler', None)
        if scaler is not None:
            if hasattr(scaler, 'mean_'):
                n_expected = int(len(scaler.mean_))
            elif hasattr(scaler, 'n_features_in_'):
                n_expected = int(getattr(scaler, 'n_features_in_'))
        if n_expected is None and hasattr(model, 'n_features_in_'):
            n_expected = int(getattr(model, 'n_features_in_'))
    except Exception:
        n_expected = None

    # Choose sensor tags to approximate model expectation (prefer thighs)
    tags_available = list(sets.keys())
    pref = []
    for cand in ["lpthigh", "rpthigh"]:
        if cand in sets:
            pref.append(cand)
    thighs = [k for k in tags_available if ("thigh" in k) and (k not in pref)]
    others = [k for k in tags_available if (k not in pref) and (k not in thighs)]
    ordered_tags = pref + thighs + others
    if len(ordered_tags) == 0:
        ordered_tags = tags_available
    # target number of sensors based on 28 features/sensor
    if n_expected and n_expected > 0:
        k_sensors = max(1, min(len(ordered_tags), int(round(n_expected / 28.0))))
    else:
        k_sensors = max(1, min(len(ordered_tags), 2))
    chosen_tags = ordered_tags[:k_sensors]

    # Subset IMU to only the chosen sensor sets
    keep_cols = ['time_s']
    for tag in chosen_tags:
        mapping = sets[tag]
        for k in base:
            keep_cols.append(mapping[k])
    keep_cols = [c for c in keep_cols if c in imu_df.columns]
    imu_df = imu_df[keep_cols].copy()

    # Build features (single-sensor) with fs derived from time_s, but guard against bad scales
    fs_use = float(args.fs_hint)
    try:
        tt = pd.to_numeric(imu_df['time_s'], errors='coerce').to_numpy()
        diffs = np.diff(tt)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            dt_med = float(np.median(diffs))
            if dt_med > 0:
                fs_est = 1.0 / dt_med
                # accept only plausible IMU rates and reasonably close to hint
                plausible = 20.0 <= fs_est <= 2000.0
                close_to_hint = (abs(fs_est - fs_use) / max(fs_use, 1e-6)) <= 0.5
                if plausible and (close_to_hint or fs_use <= 0):
                    fs_use = float(fs_est)
    except Exception:
        pass

    # Ensure window length yields at least one window; if too large, clamp effective fs
    n = len(imu_df)
    if n >= 3:
        max_fs_for_windows = ((n - 2) * 1000.0) / max(1, args.window_ms)
        if fs_use > max_fs_for_windows:
            fs_use = float(max_fs_for_windows)

    X, t = rolling_features(imu_df, fs_use, args.window_ms)
    if X.shape[0] == 0:
        raise ValueError(f"No windows for features (n={len(imu_df)}, fs={fs_use}, window_ms={args.window_ms})")

    # Align features to expected names/dimension
    expected_names = None
    try:
        if hasattr(model, 'named_steps') and isinstance(model.named_steps, dict):
            sc = model.named_steps.get('scaler', None)
            if sc is not None and hasattr(sc, 'feature_names_in_'):
                expected_names = [str(x) for x in sc.feature_names_in_]
        if expected_names is None:
            # some sklearn versions put names on the estimator or pipeline
            if hasattr(model, 'feature_names_in_'):
                expected_names = [str(x) for x in getattr(model, 'feature_names_in_')]
    except Exception:
        expected_names = None

    # ensure X has string column names
    X.columns = [str(c) for c in X.columns]
    if expected_names:
        # build ordered frame matching expected names, filling missing with zeros
        data = {}
        for name in expected_names:
            if name in X.columns:
                data[name] = pd.to_numeric(X[name], errors='coerce').fillna(0.0).to_numpy()
            else:
                data[name] = np.zeros(X.shape[0], dtype=float)
        X = pd.DataFrame(data)
    elif n_expected and n_expected > 0:
        # fallback: pad/truncate to length without relying on names
        if X.shape[1] > n_expected:
            X = X.iloc[:, :n_expected]
        elif X.shape[1] < n_expected:
            pad = n_expected - X.shape[1]
            pad_cols = {f"pad_{i}": np.zeros(X.shape[0], dtype=float) for i in range(pad)}
            X = pd.concat([X, pd.DataFrame(pad_cols)], axis=1)

    # ensure numeric dtype
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Predict (model may output Newtons, %BW, or BW ratio)
    y_pred = model.predict(X)
    try:
        p95 = float(np.nanpercentile(np.abs(y_pred), 95))
    except Exception:
        p95 = 0.0
    # Heuristics:
    # - If clearly in Newtons (very large absolute values), use directly
    # - Else if values look like %BW (>10), convert from %BW
    # - Else treat as BW ratio
    if np.isfinite(p95) and p95 > 1000.0:
        Fz_N = pd.to_numeric(pd.Series(y_pred), errors='coerce').to_numpy()
    elif np.isfinite(p95) and p95 > 10.0:
        y_ratio = y_pred / 100.0
        Fz_N = y_ratio * args.bw_kg * 9.81
    else:
        Fz_N = y_pred * args.bw_kg * 9.81
    grf_df = pd.DataFrame({'time_s': t, 'Fx_N': np.zeros_like(Fz_N), 'Fy_N': np.zeros_like(Fz_N), 'Fz_N': Fz_N})

    # Save to CSV and print head
    grf_df.to_csv(args.out_csv, index=False)
    print(grf_df.head())


if __name__ == "__main__":
    main()
