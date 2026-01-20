# MP-CONVERGE-3D Vast.ai Launch Notes

This note sketches how to run a full 3D vNext training job for MP-CONVERGE-3D
on Vast.ai, using the existing SafeStride repo layout.

## 1. Files to Sync to Vast

From your local machine, rsync the repo root and (optionally) recent outputs:

```bash
# From host
rsync -av --exclude '.venv' --exclude '.git' \
  /Volumes/Extreme\ SSD/safestride_clean/ \
  <vast_ssh_alias>:/workspace/safestride_clean/
```

If you want to reuse the tiny subset and converge logs for context, also sync:

- `data/vnext_gt_real_out/mp_converge_subset.json`
- `data/vnext_gt_real_out/mp_converge_3d.log`

## 2. Environment Setup on Vast

On the Vast instance (inside the container):

```bash
cd /workspace/safestride_clean
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# Optional: install extra tooling used in analyses
pip install numpy pyyaml torch
```

If you want the MP-CONVERGE-3D runners to use this venv explicitly, create:

```bash
echo ".venv/bin/python" > data/vnext_gt_real_out/mp_converge_python.txt
```

## 3. Dataset Path Assumptions

- Training configs generally assume:
  - `SAFESTRIDE_DATA_ROOT` points at a directory containing manifests and raw data.
  - `SAFESTRIDE_OUT_ROOT` points at the output root (defaults to `data/vnext_gt_real_out`).
- On Vast, a common pattern is:

```bash
export SAFESTRIDE_DATA_ROOT=/data
export SAFESTRIDE_OUT_ROOT=/workspace/safestride_clean/data/vnext_gt_real_out
```

Make sure any config you use for MP-CONVERGE-3D is compatible with these paths.

## 4. Launching a Full 3D Training Job

Use the stub script created for this purpose:

```bash
cd /workspace/safestride_clean
chmod +x scripts/run_mp_converge_3d_full_train_stub.sh

# Example (you must provide a real config path):
./scripts/run_mp_converge_3d_full_train_stub.sh \
  --config data/vnext_gt_real_out/tmp_converge_overfit_3d.yaml \
  --device cuda
```

This stub does **not** run training by default in this workflow; it logs the
suggested `train_vnext.py` command to:

- `data/vnext_gt_real_out/mp_converge_3d_full_train.log`

You can then copy/paste and adapt that command for the actual full-train run
once hyperparameters and manifests are finalized.

## 5. Using the Converge Runner in Retrain / Autopatch Mode

Once you have at least one successful 3D run under `data/vnext_gt_real_out/vnext_fz/`,
you can drive short overfit and autopatch experiments from the converge script:

```bash
cd /workspace/safestride_clean

# Fast reuse (no training, reuses latest metrics if present)
bash scripts/run_mp_converge_3d.sh

# Short 60-epoch overfit retrain using latest config
bash scripts/run_mp_converge_3d.sh --retrain-overfit

# Bounded autopatch loop (<=3 attempts: target_norm, axis_weights, loss)
bash scripts/run_mp_converge_3d.sh --retrain-overfit --auto-patch
```

Each invocation appends to `data/vnext_gt_real_out/mp_converge_3d.log` and prints
exactly six summary lines to stdout.

## 6. Artifacts to Copy Back

After a full training job, copy back at least:

- The new `data/vnext_gt_real_out/vnext_fz/<run_id>/` directory
- The associated `analysis_converge/3d_metrics_summary.json`
- Updated converge logs:
  - `data/vnext_gt_real_out/mp_converge_3d.log`
  - `data/vnext_gt_real_out/mp_converge_3d_full_train.log`

Example rsync from Vast to local:

```bash
rsync -av \
  /workspace/safestride_clean/data/vnext_gt_real_out/ \
  <local_alias>:/Volumes/Extreme\ SSD/safestride_clean/data/vnext_gt_real_out/
```

These artifacts allow the local `run_mp_converge_3d.sh` runner to reuse the
latest 3D run, recompute metrics if needed, and drive subsequent audits.

## 7. Common Failure Modes and Quick Fixes

- **ImportError: vnext not found**
  - Ensure `pip install -e .` was run from `/workspace/safestride_clean` inside
    the active virtualenv.
- **Config file not found**
  - Double-check the `--config` path when using the full-train stub or
    `train_vnext.py`. Paths are relative to repo root unless absolute.
- **Missing GRF data or zero evaluation windows**
  - Verify manifests under `SAFESTRIDE_DATA_ROOT` and that the config's
    `data.train_manifest` / `data.manifest` are correct.
- **OOM or very slow training**
  - Reduce `training.batch_size`, increase `training.window_stride`, or ensure
    you are running on a GPU device (`--device cuda`).
- **Converge runner reports high Fz RMSE but high corr**
  - Use `--retrain-overfit` and then `--retrain-overfit --auto-patch` to try
    `tweak_target_norm`, `tweak_axis_weights`, or `tweak_loss` automatically.
