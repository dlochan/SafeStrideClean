from pathlib import Path


try:
    import vnext  # noqa: F401
except ModuleNotFoundError as e:
    raise SystemExit(
        "Could not import 'vnext'. Install the repo in editable mode from the repo root: "
        "`python -m pip install -e .`"
    ) from e


def main() -> None:
    import argparse
    import pandas as pd

    from vnext.core.config import load_config
    from vnext.core.validation import validate_config
    from vnext.core.paths import SafeStridePaths
    from vnext.data.imu_schema import EXPECTED_IMU_COLUMNS, validate_canonical_imu_df
    from vnext.data.datasets import DualIMUTrialDataset, WindowedIMUDataset

    ap = argparse.ArgumentParser(description="Validate vNext setup (imports, config, data/manifests, schema)")
    ap.add_argument(
        "--config",
        default="configs/vnext_example.yaml",
        help="Config to validate (default: configs/vnext_example.yaml)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(
            f"Config not found: {cfg_path}\n"
            "Next step: run\n"
            "  python scripts/check_vnext_setup.py --config configs/vnext_example.yaml"
        )

    cfg = validate_config(load_config(cfg_path))
    paths = SafeStridePaths.from_env_or_defaults(cfg.get("paths", {}) or {})
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    grf_axes = str(model_cfg.get("grf_axes", "fz"))

    print(f"config={cfg_path}")
    print(f"data_root={paths.data_root}")
    print(f"out_root={paths.out_root}")
    print(f"grf_axes={grf_axes}")

    if not paths.data_root.exists():
        raise SystemExit(
            f"data_root does not exist: {paths.data_root}\n"
            "If you are using GT-small, build canonical data with:\n"
            "  python scripts/make_vnext_gt_real.py\n"
            "Or for a tiny smoke dataset:\n"
            "  python scripts/make_vnext_gt_smoke.py"
        )

    train_manifest_rel = data_cfg.get("train_manifest")
    val_manifest_rel = data_cfg.get("val_manifest")
    if train_manifest_rel is None:
        raise SystemExit(
            "Config is missing data.train_manifest\n"
            "Next step: use configs/vnext_example.yaml or set data.train_manifest in your config."
        )

    train_manifest_path = paths.data_root / str(train_manifest_rel)
    val_manifest_path = (paths.data_root / str(val_manifest_rel)) if val_manifest_rel else None

    if not train_manifest_path.exists():
        raise SystemExit(
            f"Train manifest not found: {train_manifest_path}\n"
            "Next step (GT-small):\n"
            "  python scripts/make_vnext_gt_real.py\n"
            f"Expected to create: {train_manifest_path}"
        )
    if val_manifest_path is not None and not val_manifest_path.exists():
        raise SystemExit(
            f"Val manifest not found: {val_manifest_path}\n"
            "Next step (GT-small):\n"
            "  python scripts/make_vnext_gt_real.py\n"
            f"Expected to create: {val_manifest_path}"
        )

    df = pd.read_csv(train_manifest_path)
    for col in ("trial_id", "imu_path"):
        if col not in df.columns:
            raise SystemExit(f"Manifest schema invalid; missing column '{col}' in {train_manifest_path}")

    if len(df) == 0:
        raise SystemExit(f"Train manifest is empty: {train_manifest_path}")

    first = df.iloc[0]
    imu_path = Path(str(first["imu_path"]))
    grf_path = None
    if "grf_path" in df.columns:
        raw = first.get("grf_path")
        if isinstance(raw, str) and raw.strip():
            grf_path = Path(raw)

    if not imu_path.exists():
        raise SystemExit(
            f"Sample IMU file from manifest not found: {imu_path}\n"
            f"Manifest: {train_manifest_path}"
        )
    if grf_path is not None and not grf_path.exists():
        raise SystemExit(
            f"Sample GRF file from manifest not found: {grf_path}\n"
            f"Manifest: {train_manifest_path}"
        )

    imu_df = pd.read_csv(imu_path)
    validate_canonical_imu_df(imu_df)
    for c in EXPECTED_IMU_COLUMNS:
        if c not in imu_df.columns:
            raise SystemExit(f"IMU schema invalid; missing column '{c}' in {imu_path}")

    if grf_path is not None:
        grf_df = pd.read_csv(grf_path)
        if grf_axes == "fz":
            if not any(c in grf_df.columns for c in ("Fz_N", "Fz_BW", "Fz_%BW")):
                raise SystemExit(
                    f"GRF schema invalid for fz; expected one of Fz_N/Fz_BW/Fz_%BW in {grf_path}"
                )
        else:
            missing = [c for c in ("Fx_N", "Fy_N", "Fz_N") if c not in grf_df.columns]
            if missing:
                raise SystemExit(
                    f"GRF schema invalid for 3d; missing columns {missing} in {grf_path}"
                )

    ds = DualIMUTrialDataset(train_manifest_path, grf_axes=grf_axes)
    print(f"Base dataset length={len(ds)}")
    wds = WindowedIMUDataset(base_dataset=ds, window_size=256, window_stride=128, require_grf=True)
    print(f"Windowed dataset length={len(wds)}")
    print("OK: vNext setup looks good")


if __name__ == "__main__":
    main()
