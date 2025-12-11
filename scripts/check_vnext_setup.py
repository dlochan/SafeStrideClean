from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))


def main() -> None:
    from vnext.data.datasets import DualIMUTrialDataset, WindowedIMUDataset
    from vnext.data.imu_schema import TIME_COL, get_feature_columns, get_sensor_slices
    from vnext.core.paths import SafeStridePaths
    from vnext.core.config import load_config

    cfg = load_config("configs/vnext_example.yaml")
    paths = SafeStridePaths.from_env_or_defaults(cfg.get("paths", {}) or {})
    print(f"data_root={paths.data_root}")
    print(f"out_root={paths.out_root}")

    feature_cols = get_feature_columns()
    print(f"feature_cols (len={len(feature_cols)}): {feature_cols[:10]}...")

    data_cfg = cfg.get("data", {}) or {}
    train_manifest_rel = data_cfg.get("train_manifest")
    if train_manifest_rel:
        train_manifest_path = paths.data_root / str(train_manifest_rel)
        print(f"train_manifest_path={train_manifest_path} exists={train_manifest_path.exists()}")
        if train_manifest_path.exists():
            ds = DualIMUTrialDataset(train_manifest_path, grf_axes="fz")
            print(f"Base dataset length={len(ds)}")
            wds = WindowedIMUDataset(base_dataset=ds, window_size=256, window_stride=128, require_grf=True)
            print(f"Windowed dataset length={len(wds)}")


if __name__ == "__main__":
    main()
