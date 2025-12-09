# Running vNext GT-small on Vast.ai

This guide shows how to spin up a Vast.ai instance, mount your Georgia Tech `datasets/ProcessedData` folder, and run the vNext GT-small smoke and long sweeps on a GPU.

It assumes:

- Your GT `ProcessedData` is mounted inside the container at:
  - `/workspace/data/datasets/ProcessedData`
- You will set:
  - `SAFESTRIDE_DATA_ROOT=/workspace/data`

The vNext code will then build a small canonical dataset under the **repo-local** `data/vnext_gt_real` directory and write outputs under `data/vnext_gt_real_out`, both of which are git-ignored.

## 1. Clone the repo on Vast

Inside your Vast container shell:

```bash
cd /workspace
git clone <YOUR_SAFESTRIDE_REPO_URL>.git safestride
cd safestride
```

> The `data/`, `out/`, `work/`, `release/`, and checkpoint files (`*.pt`, `*.pth`) are all listed in `.gitignore`, so runs on Vast will not pollute git history.

If you have already committed large data/checkpoint files in history **before** adding `.gitignore`, consider cleaning history locally using tools like `git filter-repo` or BFG before pushing to a shared remote.

## 2. Create and activate the conda environment

Assuming `conda` (or `mamba`) is available in your image:

```bash
conda env create -f environment.yml
conda activate safestride
```

If the environment already exists, you can just run:

```bash
conda activate safestride
```

## 3. Point to GT ProcessedData via `SAFESTRIDE_DATA_ROOT`

`make_vnext_gt_real.py` expects to find the GT ProcessedData under:

```text
${SAFESTRIDE_DATA_ROOT}/datasets/ProcessedData
```

On Vast, with your dataset mounted at `/workspace/data/datasets/ProcessedData`, set:

```bash
export SAFESTRIDE_DATA_ROOT=/workspace/data  # contains datasets/ProcessedData
```

This affects:

- `scripts/make_vnext_gt_real.py`, which reads raw GT IMU/GRF files from:
  - `/workspace/data/datasets/ProcessedData/<Subject>/<trial_name>`
- Path utilities (`SafeStridePaths`) when configs choose to use `SAFESTRIDE_DATA_ROOT`.

## 4. Build the GT-small canonical dataset

From the repo root (`/workspace/safestride`):

```bash
python scripts/make_vnext_gt_real.py
```

This will create canonical IMU/GRF CSVs and manifests under:

- `data/vnext_gt_real/`
  - `imu_gt_<k>.csv`, `grf_gt_<k>.csv`
  - `manifests/vnext_train_real.csv`
  - `manifests/vnext_val_real.csv`
  - `manifests/vnext_test_real.csv`

The **sweep configs** consume these manifests via `paths.data_root: "data/vnext_gt_real"`, independent of the original GT ProcessedData location.

## 5. Run the 1-epoch smoke sweep on GPU (Fz BW)

To run the GT-small 1-epoch smoke suite on CUDA (quick sanity check):

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_small.yaml \
  --device cuda \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_smoke_gpu.csv
```

Notes:

- You can replace `cuda` with `cuda:0`, `cuda:1`, etc., depending on the device string for your Vast instance.
- Outputs (run directories, metrics, leaderboard) are written under `data/vnext_gt_real_out`, which is git-ignored.

## 6. Run the 20-epoch long sweep on GPU (Fz BW)

For the longer 20-epoch Fz BW suite:

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_small_long.yaml \
  --device cuda \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_long_gpu.csv
```

This uses the same canonical `data/vnext_gt_real` manifests but with:

- `batch_size = 32`
- `window_size = 128`
- `epochs = 20`

and writes its leaderboard CSV to `data/vnext_gt_real_out/vnext_leaderboard_gt_small_long_gpu.csv`.

## 7. Optional: 3D GRF sweep on GPU

If you also want to run the 3D GRF (Fx/Fy/Fz) sweep on CUDA:

```bash
python scripts/run_vnext_experiments.py \
  --suite configs/vnext_sweep_gt_small_fxyz.yaml \
  --device cuda \
  --out-leaderboard data/vnext_gt_real_out/vnext_leaderboard_gt_small_fxyz_gpu.csv
```

This uses the same GT-small manifests but configures the model as 3D GRF (`model.type: "grf3d"`, `model.grf_axes: "fxyz"`).

---

For more details on the dataset, manifests, and metrics, see `docs/vnext_gt_small.md`.
