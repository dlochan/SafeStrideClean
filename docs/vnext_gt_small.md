# vNext GT Small Real Dataset

## Overview

This is a small multi-subject Georgia Tech real dataset in canonical vNext format, intended for quick local CPU experiments and as a template for cloud / GPU runs.

- **Canonical data root**: `data/vnext_gt_real`
- **Sweep config (Fz only)**: `configs/vnext_sweep_gt_small.yaml`
- **3D GRF sweep config (Fx/Fy/Fz)**: `configs/vnext_sweep_gt_small_fxyz.yaml`
- **Build script**: `scripts/make_vnext_gt_real.py`

Regenerate everything with:

```bash
python scripts/make_vnext_gt_real.py
python scripts/run_vnext_experiments.py --suite configs/vnext_sweep_gt_small.yaml --device cpu --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard.csv
```

## Environment and data roots

To run the GT-small vNext experiments from a clean clone, create and activate the conda env:

```bash
conda env create -f environment.yml
conda activate safestride
```

In a cloud or container environment, point the build script at your mounted GT ProcessedData root using `SAFESTRIDE_DATA_ROOT` (the script expects `datasets/ProcessedData` beneath this root):

```bash
export SAFESTRIDE_DATA_ROOT=/workspace/data  # contains datasets/ProcessedData
python scripts/make_vnext_gt_real.py
```

This produces canonical IMU/GRF CSVs and manifests under the repo-relative root `data/vnext_gt_real`.

To run the **20-epoch long sweep** on CPU:

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_small_long.yaml \
  --device cpu \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_long.csv
```

You can swap `--device cpu` for `--device cuda` (or `cuda:0`, etc.) on a GPU box.

## Subjects and Trials

Trials are drawn from `${SAFESTRIDE_DATA_ROOT:-./data}/datasets/ProcessedData/<Subject>/<trial_name>` and require both `*_imu_real.csv` and `*_grf.csv`.

### Train subjects (15 trials)

- **AB01**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`
- **AB02**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`
- **AB03**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`
- **AB05**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `stairs_1_5_up`
- **AB06**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`

### Val subjects (6 trials)

- **AB08**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`
- **AB09**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`

### Test subjects (6 trials)

- **AB10**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`
- **AB11**
  - `normal_walk_1_0-6`
  - `normal_walk_1_2-0`
  - `weighted_walk_1_25lbs`

## Manifests and Trial IDs

Manifests live under `data/vnext_gt_real/manifests`:

- **Train**: `vnext_train_real.csv` (15 trials)
- **Val**: `vnext_val_real.csv` (6 trials)
- **Test**: `vnext_test_real.csv` (6 trials)

Schema:

```csv
trial_id,imu_path,grf_path
```

- **`trial_id`**: `gt_small_XXX`, where `XXX` is `001`–`027` in conversion order.
- **`imu_path`**: relative path, e.g. `data/vnext_gt_real/imu_gt_1.csv`.
- **`grf_path`**: relative path, e.g. `data/vnext_gt_real/grf_gt_1.csv`.

## Canonical Schemas

### IMU CSV (`imu_gt_k.csv`)

Columns (dual-IMU, right thigh + shank):

```csv
time_s,axx_thigh,axy_thigh,axz_thigh,gxx_thigh,gxy_thigh,gxz_thigh,axx_shank,axy_shank,axz_shank,gxx_shank,gxy_shank,gxz_shank
```

Mappings from GT raw IMU (`*_imu_real.csv`):

- `time_s      <- time`
- `axx_thigh   <- RAThigh_ACCX`
- `axy_thigh   <- RAThigh_ACCY`
- `axz_thigh   <- RAThigh_ACCZ`
- `gxx_thigh   <- RAThigh_GYROX`
- `gxy_thigh   <- RAThigh_GYROY`
- `gxz_thigh   <- RAThigh_GYROZ`
- `axx_shank   <- RShank_ACCX`
- `axy_shank   <- RShank_ACCY`
- `axz_shank   <- RShank_ACCZ`
- `gxx_shank   <- RShank_GYROX`
- `gxy_shank   <- RShank_GYROY`
- `gxz_shank   <- RShank_GYROZ`

### GRF CSV (`grf_gt_k.csv`)

Columns:

```csv
Fx_N,Fy_N,Fz_N,Fx_BW,Fy_BW,Fz_BW
```

Where for each trial (total right + left force plate channels):

- `Fx_N  = RForceX + LForceX` (Newtons)
- `Fy_N  = RForceZ + LForceZ` (Newtons; GT fore-aft axis)
- `Fz_N  = RForceY_Vertical + LForceY_Vertical` (Newtons, vertical)
- `Fx_BW = Fx_N / (mass_kg * 9.81)` (body weights)
- `Fy_BW = Fy_N / (mass_kg * 9.81)` (body weights)
- `Fz_BW = Fz_N / (mass_kg * 9.81)` (body weights)

Subject mass `mass_kg` is read from `Subject_masses.csv` under `${SAFESTRIDE_DATA_ROOT:-./data}/datasets/ProcessedData` with a sensible default fallback if the subject is missing.

Existing Fz-only configs continue to use just `Fz_N` or `Fz_BW` as the regression target; the additional Fx/Fy axes are used by the 3D GRF models.

## Sweep Config: `configs/vnext_sweep_gt_small.yaml`

This suite defines three experiments, all using the manifests above and windowed Fz prediction:

- **`gt_small_fz_raw_n`**
  - Tags: `["gt_small", "fz", "raw", "cpu", "N"]`
  - Inputs: raw dual-IMU only (no kinematics).
  - Target: default vertical GRF column (Fz_N in Newtons).

- **`gt_small_fz_raw_bw`**
  - Tags: `["gt_small", "fz", "raw", "cpu", "bw"]`
  - Inputs: raw dual-IMU.
  - Target: `Fz_BW` via `model.target_grf_column: "Fz_BW"`.

- **`gt_small_fz_kinematics_bw`**
  - Tags: `["gt_small", "fz", "kinematics", "cpu", "bw"]`
  - Inputs: raw dual-IMU + derived kinematics (from base config).
  - Target: `Fz_BW`.

Common training hyperparameters:

- `batch_size = 2`
- `window_size = 64`
- `window_stride = 32`
- `epochs = 1`
- `lr = 1e-3`
- `num_workers = 0`

## Metrics (example runs)

### 1-epoch "smoke" sweep

The leaderboard for a short, 1-epoch CPU run is written to:

- `data/vnext_gt_real_out/vnext_leaderboard.csv`
- `data/vnext_gt_real_out/vnext_leaderboard.json`

Sorted by `eval_rmse_mean`, a typical run produced approximately:

- **`gt_small_fz_kinematics_bw`**: `eval_rmse_mean ≈ 0.42 BW`
- **`gt_small_fz_raw_bw`**: `eval_rmse_mean ≈ 0.42–0.43 BW`
- **`gt_small_fz_raw_n`**: `eval_rmse_mean ≈ 3.1e2 N` (raw Newton scale)

So on this small dataset:

- **BW models** achieve roughly **0.42–0.43 BW RMSE** on the val set.
- **Adding kinematics** provides a small but consistent improvement over raw IMU for the BW target.

### 20-epoch "long" sweep

A longer-training sweep uses `configs/vnext_sweep_gt_small_long.yaml` and writes its leaderboard to:

- `data/vnext_gt_real_out/vnext_leaderboard_gt_small_long.csv`

For a representative 20-epoch CPU run with `window_size=128`, `batch_size=32`, and `epochs=20`:

- **`gt_small_fz_kinematics_bw`**
  - Val: `eval_rmse_mean ≈ 0.405 BW`
  - Test: `rmse_mean ≈ 0.532 BW`
- **`gt_small_fz_raw_bw`**
  - Val: `eval_rmse_mean ≈ 0.409 BW`
  - Test: `rmse_mean ≈ 0.565 BW`

On this GT-small dataset, the BW models stabilize around **0.40–0.42 BW RMSE on val** and **~0.53–0.57 BW on test**, with kinematics giving a modest but consistent gain over raw IMU.

## GPU training

The training and evaluation scripts both accept a `--device` argument and move models/tensors to that device via `torch.device(args.device)`. There are no hard-coded `.to("cpu")` or `.cuda()` calls in the vNext training/eval loop; everything follows the `--device` flag.

### Run on one GPU

Assuming you have already built the GT-small canonical dataset with:

```bash
export SAFESTRIDE_DATA_ROOT=/workspace/data  # contains datasets/ProcessedData
python scripts/make_vnext_gt_real.py
```

you can run the standard suites on a single GPU (e.g. `cuda` or `cuda:0`) as follows:

**Fz BW 1-epoch smoke sweep (sanity / quick check):**

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_small.yaml \
  --device cuda \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_smoke_gpu.csv
```

**Fz BW 20-epoch long sweep (main Fz baseline):**

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_small_long.yaml \
  --device cuda \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_long_gpu.csv
```

**3D GRF (Fx/Fy/Fz) sweep:**

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_small_fxyz.yaml \
  --device cuda \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_fxyz_gpu.csv
```

You can swap `cuda` for `cuda:0`, `cuda:1`, etc., depending on your setup.

### Tiny overfit suite (debugging tool)

For low-level debugging, there is an additional tiny overfit suite:

- Config: `configs/vnext_sweep_gt_tiny_overfit.yaml`
- Data: uses the tiny manifests written by `scripts/make_vnext_gt_real.py`:
  - `data/vnext_gt_real/manifests/vnext_tiny_overfit_train.csv`
  - `data/vnext_gt_real/manifests/vnext_tiny_overfit_val.csv`
- Purpose: train on just 1–2 GT-small trials in both train and val for many epochs so that train/val RMSE can be driven very low. This is useful to verify that the model and data pipeline are wired correctly and can overfit when given an easy task.

Example run on one GPU:

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_tiny_overfit.yaml \
  --device cuda \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_tiny_overfit_gpu.csv
```

On a healthy setup, you should see train/val `rmse_mean` drop roughly monotonically and reach very low values compared to the standard GT-small sweeps.

### Resuming long runs

For longer single-config runs (outside of the sweep driver), `scripts/train_vnext.py` supports resuming from a run directory that already contains checkpoints.

Example (resuming on GPU from an existing `run_dir`):

```bash
python scripts/train_vnext.py \
  --config path/to/effective_config.yaml \
  --device cuda \
  --run-dir data/vnext_gt_real_out/vnext_fz/<run_id> \
  --resume
```

When `--resume` is set:

- `train_vnext.py` looks for `model_last.pt` (and `optimizer_last.pt`) in `--run-dir` and loads them if present, otherwise it logs a warning and starts from scratch.
- A per-epoch CSV log `train_history.csv` is written under the same `run_dir` with columns like `epoch`, `train_loss`, `train_rmse_mean`, `val_rmse_mean`, and per-axis RMSE (e.g., `val_rmse_Fz`). This is useful for plotting or inspecting epoch-by-epoch behavior for long runs.

## 3D GRF (fxyz) on GT-small

### Sweep config: `configs/vnext_sweep_gt_small_fxyz.yaml`

This suite trains simple 3-axis GRF models (Fx, Fy, Fz in Newtons) on the same GT-small manifests:

- **`gt_small_fxyz_raw_n`**
  - Tags: `["gt_small", "fxyz", "raw", "cpu", "N"]`
  - Inputs: raw dual-IMU only (no kinematics).
  - Model: `model.type: "grf3d"`, `model.grf_axes: "fxyz"` (treated as 3D Fx/Fy/Fz).
  - Target: 3D GRF in Newtons, using canonical `Fx_N`, `Fy_N`, `Fz_N`.

- **`gt_small_fxyz_kinematics_n`**
  - Tags: `["gt_small", "fxyz", "kinematics", "cpu", "N"]`
  - Inputs: raw dual-IMU + kinematic features (from the base config).
  - Model: `model.type: "grf3d"`, `model.grf_axes: "fxyz"`.
  - Target: same 3D GRF in Newtons.

Both experiments reuse the same manifests as the Fz-only sweeps:

- Train: `manifests/vnext_train_real.csv`
- Val:   `manifests/vnext_val_real.csv`

Training hyperparameters mirror the long Fz sweep (windowed, 5 epochs here for a quick run):

- `batch_size = 32`
- `window_size = 128`
- `window_stride = 32`
- `epochs = 5`
- `lr = 1e-3`

### 3D GRF metrics (example 5-epoch CPU run)

From `data/vnext_gt_real_out/vnext_leaderboard_gt_small_fxyz.csv` and a held-out test evaluation using `--eval-manifest manifests/vnext_test_real.csv`:

- **`gt_small_fxyz_raw_n`** (raw IMU, no kinematics)
  - **Val (vnext_val_real)**
    - `rmse_mean ≈ 145.4 N`
    - Per-axis RMSE: `Fx ≈ 52.4 N`, `Fy ≈ 62.6 N`, `Fz ≈ 321.1 N`.
  - **Test (vnext_test_real)**
    - `rmse_mean ≈ 135.4 N`
    - Per-axis RMSE: `Fx ≈ 48.6 N`, `Fy ≈ 46.1 N`, `Fz ≈ 311.6 N`.

- **`gt_small_fxyz_kinematics_n`** (IMU + kinematics)
  - **Val (vnext_val_real)**
    - `rmse_mean ≈ 149.9 N`
    - Per-axis RMSE: `Fx ≈ 52.5 N`, `Fy ≈ 62.2 N`, `Fz ≈ 335.1 N`.

On this small dataset and short run, the 3D models primarily reflect vertical-force error (Fz dominates the scale), and kinematics do not outperform the raw IMU baseline for 3D GRF.

### How to run the 3D GRF experiments

1. **Run the 3D GRF sweep (Fx/Fy/Fz in N)**

   ```bash
   python scripts/run_vnext_experiments.py \
     --suite configs/vnext_sweep_gt_small_fxyz.yaml \
     --device cpu \
     --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_fxyz.csv
   ```

2. **Evaluate a 3D run on the held-out test manifest**

   After selecting a 3D experiment (for example, `gt_small_fxyz_raw_n`) from `vnext_leaderboard_gt_small_fxyz.csv`, use its `run_dir` and corresponding effective config path, then run:

   ```bash
   python scripts/eval_vnext.py \
     --config <effective_config_for_3d_experiment>.yaml \
     --run-dir <run_dir_from_3d_leaderboard> \
     --checkpoint best \
     --device cpu \
     --eval-manifest manifests/vnext_test_real.csv
   ```

   This writes `eval/eval_metrics_test.json` and `eval/eval_windows_test.csv` for the chosen 3D model, alongside the corresponding `*_val.*` files when you evaluate on the val manifest.
