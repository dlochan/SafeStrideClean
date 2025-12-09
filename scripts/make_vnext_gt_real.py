from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
_data_root_env = os.getenv("SAFESTRIDE_DATA_ROOT")
if _data_root_env:
    DATA_ROOT = Path(_data_root_env)
else:
    DATA_ROOT = REPO_ROOT / "data"

BASE_PROCESSED = DATA_ROOT / "datasets" / "ProcessedData"

# Expanded set of GT trials for the small real dataset.
# Each entry is a (subject_id, trial_name) pair corresponding to a folder
# BASE_PROCESSED / subject / trial_name containing
#   <subject>_<trial_name>_imu_real.csv and <subject>_<trial_name>_grf.csv.
TRIALS: list[dict[str, str]] = [
    # Train subjects
    {"subject": "AB01", "name": "normal_walk_1_0-6"},
    {"subject": "AB01", "name": "normal_walk_1_2-0"},
    {"subject": "AB01", "name": "weighted_walk_1_25lbs"},

    {"subject": "AB02", "name": "normal_walk_1_0-6"},
    {"subject": "AB02", "name": "normal_walk_1_2-0"},
    {"subject": "AB02", "name": "weighted_walk_1_25lbs"},

    {"subject": "AB03", "name": "normal_walk_1_0-6"},
    {"subject": "AB03", "name": "normal_walk_1_2-0"},
    {"subject": "AB03", "name": "weighted_walk_1_25lbs"},

    {"subject": "AB05", "name": "normal_walk_1_0-6"},
    {"subject": "AB05", "name": "normal_walk_1_2-0"},
    {"subject": "AB05", "name": "stairs_1_5_up"},

    {"subject": "AB06", "name": "normal_walk_1_0-6"},
    {"subject": "AB06", "name": "normal_walk_1_2-0"},
    {"subject": "AB06", "name": "weighted_walk_1_25lbs"},

    # Val subjects
    {"subject": "AB08", "name": "normal_walk_1_0-6"},
    {"subject": "AB08", "name": "normal_walk_1_2-0"},
    {"subject": "AB08", "name": "weighted_walk_1_25lbs"},

    {"subject": "AB09", "name": "normal_walk_1_0-6"},
    {"subject": "AB09", "name": "normal_walk_1_2-0"},
    {"subject": "AB09", "name": "weighted_walk_1_25lbs"},

    # Test subjects
    {"subject": "AB10", "name": "normal_walk_1_0-6"},
    {"subject": "AB10", "name": "normal_walk_1_2-0"},
    {"subject": "AB10", "name": "weighted_walk_1_25lbs"},

    {"subject": "AB11", "name": "normal_walk_1_0-6"},
    {"subject": "AB11", "name": "normal_walk_1_2-0"},
    {"subject": "AB11", "name": "weighted_walk_1_25lbs"},
]


def _load_mass_kg(subject_id: str) -> float:
    """Return subject mass in kg, falling back to a default if unavailable."""

    masses_path = BASE_PROCESSED / "Subject_masses.csv"
    default_mass = 75.0
    if not masses_path.exists():
        return default_mass

    df_masses = pd.read_csv(masses_path)
    subject_cols = ["SubjectID", "Subject", "ID"]
    mass_cols = ["Mass_kg", "Mass (kg)", "MassKg"]

    subj_col = next((c for c in subject_cols if c in df_masses.columns), None)
    mass_col = next((c for c in mass_cols if c in df_masses.columns), None)
    if subj_col is None or mass_col is None:
        raise SystemExit(
            f"Could not find suitable subject/mass columns in {masses_path}; "
            f"available columns: {list(df_masses.columns)}"
        )

    row = df_masses.loc[df_masses[subj_col] == subject_id]
    if row.empty:
        # Fall back to default if this specific subject is missing.
        return default_mass
    return float(row[mass_col].iloc[0])


def main() -> None:
    """Build canonical vNext IMU/GRF CSVs for a small multi-subject GT dataset.

    Outputs under repo-relative data/vnext_gt_real/:
      - imu_gt_<k>.csv, grf_gt_<k>.csv per trial
      - manifests/vnext_train_real.csv
      - manifests/vnext_val_real.csv
      - manifests/vnext_test_real.csv

    IMU mapping (canonical -> GT columns):
      time_s      <- time
      axx_thigh   <- RAThigh_ACCX
      axy_thigh   <- RAThigh_ACCY
      axz_thigh   <- RAThigh_ACCZ
      gxx_thigh   <- RAThigh_GYROX
      gxy_thigh   <- RAThigh_GYROY
      gxz_thigh   <- RAThigh_GYROZ

      axx_shank   <- RShank_ACCX
      axy_shank   <- RShank_ACCY
      axz_shank   <- RShank_ACCZ
      gxx_shank   <- RShank_GYROX
      gxy_shank   <- RShank_GYROY
      gxz_shank   <- RShank_GYROZ

    GRF mapping (3D total forces):
      Fx_N  <- RForceX + LForceX
      Fy_N  <- RForceZ + LForceZ   (fore-aft / horizontal, per GT axes)
      Fz_N  <- RForceY_Vertical + LForceY_Vertical

      Fx_BW <- Fx_N / (mass_kg * 9.81)
      Fy_BW <- Fy_N / (mass_kg * 9.81)
      Fz_BW <- Fz_N / (mass_kg * 9.81)

    Length alignment: truncate both IMU and GRF to the same min length per trial.
    """

    out_root = Path("data/vnext_gt_real")
    manifests_dir = out_root / "manifests"
    out_root.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    required_imu_cols = [
        "time",
        "RAThigh_ACCX",
        "RAThigh_ACCY",
        "RAThigh_ACCZ",
        "RAThigh_GYROX",
        "RAThigh_GYROY",
        "RAThigh_GYROZ",
        "RShank_ACCX",
        "RShank_ACCY",
        "RShank_ACCZ",
        "RShank_GYROX",
        "RShank_GYROY",
        "RShank_GYROZ",
    ]

    for idx, trial in enumerate(TRIALS, start=1):
        subject = trial["subject"]
        name = trial["name"]
        base = BASE_PROCESSED / subject / name
        src_imu = base / f"{subject}_{name}_imu_real.csv"
        src_grf = base / f"{subject}_{name}_grf.csv"

        if not src_imu.exists() or not src_grf.exists():
            print(
                f"Skipping {subject}_{name} due to missing files: "
                f"imu_exists={src_imu.exists()}, grf_exists={src_grf.exists()}"
            )
            continue

        df_imu = pd.read_csv(src_imu)
        missing_imu = [c for c in required_imu_cols if c not in df_imu.columns]
        if missing_imu:
            raise SystemExit(f"IMU source missing required columns: {missing_imu}")

        df_imu_new = pd.DataFrame(
            {
                "time_s": df_imu["time"],
                "axx_thigh": df_imu["RAThigh_ACCX"],
                "axy_thigh": df_imu["RAThigh_ACCY"],
                "axz_thigh": df_imu["RAThigh_ACCZ"],
                "gxx_thigh": df_imu["RAThigh_GYROX"],
                "gxy_thigh": df_imu["RAThigh_GYROY"],
                "gxz_thigh": df_imu["RAThigh_GYROZ"],
                "axx_shank": df_imu["RShank_ACCX"],
                "axy_shank": df_imu["RShank_ACCY"],
                "axz_shank": df_imu["RShank_ACCZ"],
                "gxx_shank": df_imu["RShank_GYROX"],
                "gxy_shank": df_imu["RShank_GYROY"],
                "gxz_shank": df_imu["RShank_GYROZ"],
            }
        )

        df_grf = pd.read_csv(src_grf)
        required_grf_cols = [
            "RForceX",
            "LForceX",
            "RForceZ",
            "LForceZ",
            "RForceY_Vertical",
            "LForceY_Vertical",
        ]
        missing_grf = [c for c in required_grf_cols if c not in df_grf.columns]
        if missing_grf:
            raise SystemExit(f"GRF source missing required columns: {missing_grf}")

        mass_kg = _load_mass_kg(subject)

        # Total GRFs in N from GT force plate channels
        rx = pd.to_numeric(df_grf["RForceX"], errors="coerce").fillna(0.0)
        lx = pd.to_numeric(df_grf["LForceX"], errors="coerce").fillna(0.0)
        rz = pd.to_numeric(df_grf["RForceZ"], errors="coerce").fillna(0.0)
        lz = pd.to_numeric(df_grf["LForceZ"], errors="coerce").fillna(0.0)
        rfy = pd.to_numeric(df_grf["RForceY_Vertical"], errors="coerce").fillna(0.0)
        lfy = pd.to_numeric(df_grf["LForceY_Vertical"], errors="coerce").fillna(0.0)

        fx_n = rx + lx
        fy_n = rz + lz
        fz_n = rfy + lfy

        # Convert to body weights
        g = 9.81
        denom = mass_kg * g
        fx_bw = fx_n / denom
        fy_bw = fy_n / denom
        fz_bw = fz_n / denom

        df_grf_new = pd.DataFrame(
            {
                "Fx_N": fx_n,
                "Fy_N": fy_n,
                "Fz_N": fz_n,
                "Fx_BW": fx_bw,
                "Fy_BW": fy_bw,
                "Fz_BW": fz_bw,
            }
        )

        # Align lengths (truncate to min length)
        n = int(min(len(df_imu_new), len(df_grf_new)))
        if n == 0:
            raise SystemExit("No overlapping IMU/GRF samples after alignment")

        df_imu_new = df_imu_new.iloc[:n].reset_index(drop=True)
        df_grf_new = df_grf_new.iloc[:n].reset_index(drop=True)

        imu_out = out_root / f"imu_gt_{idx}.csv"
        grf_out = out_root / f"grf_gt_{idx}.csv"

        df_imu_new.to_csv(imu_out, index=False)
        df_grf_new.to_csv(grf_out, index=False)

        if idx == 1:
            # Log canonical GRF columns for a representative trial
            print(f"Example canonical GRF columns: {list(df_grf_new.columns)}")

        manifest_rows.append(
            {
                "trial_id": f"gt_small_{idx:03d}",
                "imu_path": f"data/vnext_gt_real/imu_gt_{idx}.csv",
                "grf_path": f"data/vnext_gt_real/grf_gt_{idx}.csv",
                "subject": subject,
                "name": name,
            }
        )

        print(f"Wrote {imu_out} and {grf_out} for {subject}_{name}")

    if not manifest_rows:
        print("No trials were converted; nothing to write to manifests.")
        return

    # Subject-aware split: train/val/test
    train_subjects = {"AB01", "AB02", "AB03", "AB05", "AB06"}
    val_subjects = {"AB08", "AB09"}
    test_subjects = {"AB10", "AB11"}

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for r in manifest_rows:
        subj = r["subject"]
        if subj in train_subjects:
            train_rows.append(r)
        elif subj in val_subjects:
            val_rows.append(r)
        elif subj in test_subjects:
            test_rows.append(r)
        else:
            # Fallback: if a subject somehow isn't categorized, send to train.
            train_rows.append(r)

    def _write_manifest(rows: list[dict], path: Path) -> None:
        df = pd.DataFrame(
            [
                {
                    "trial_id": r["trial_id"],
                    "imu_path": r["imu_path"],
                    "grf_path": r["grf_path"],
                }
                for r in rows
            ]
        )
        df.to_csv(path, index=False)

    train_manifest = manifests_dir / "vnext_train_real.csv"
    val_manifest = manifests_dir / "vnext_val_real.csv"
    test_manifest = manifests_dir / "vnext_test_real.csv"

    _write_manifest(train_rows, train_manifest)
    _write_manifest(val_rows, val_manifest)
    _write_manifest(test_rows, test_manifest)

    # Optional tiny overfit manifests (1–2 trials per split) for debugging.
    tiny_train_rows = train_rows[:2]
    tiny_val_rows = val_rows[:2]
    if tiny_train_rows and tiny_val_rows:
        tiny_train_manifest = manifests_dir / "vnext_tiny_overfit_train.csv"
        tiny_val_manifest = manifests_dir / "vnext_tiny_overfit_val.csv"
        _write_manifest(tiny_train_rows, tiny_train_manifest)
        _write_manifest(tiny_val_rows, tiny_val_manifest)

    print("Trials used (subject, name, split):")
    for r in manifest_rows:
        subj = r["subject"]
        if subj in train_subjects:
            split = "train"
        elif subj in val_subjects:
            split = "val"
        elif subj in test_subjects:
            split = "test"
        else:
            split = "train?"
        print(f"  {r['subject']}, {r['name']} -> {split}")

    print(
        f"Wrote train manifest: {train_manifest} ({len(train_rows)} trials)\n"
        f"Wrote val manifest:   {val_manifest} ({len(val_rows)} trials)\n"
        f"Wrote test manifest:  {test_manifest} ({len(test_rows)} trials)"
    )


if __name__ == "__main__":
    main()
