# SafeStride vNext: Fresh GPU Instance Guide

## 0) Clone

```bash
git clone <YOUR_REPO_URL>.git safestride
cd safestride
```

## 1) Create environment

### Option A: conda (recommended)

```bash
conda env create -f environment.yml
conda activate safestride
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r <YOUR_REQUIREMENTS_IF_ANY>
```

## 2) Install editable (no PYTHONPATH)

```bash
python -m pip install -e .
```

## 3) Import smoke test

```bash
python scripts/import_smoke_test.py
python -c "import vnext; import vnext.data"
```

## 4) Put raw GT ProcessedData in place (or mount it)

Expected raw location:

```text
${SAFESTRIDE_DATA_ROOT}/datasets/ProcessedData
```

Example on Vast:

```bash
export SAFESTRIDE_DATA_ROOT=/workspace/data
```

## 5) Build canonical GT-small dataset

```bash
python scripts/make_vnext_gt_real.py
```

This should create:

- data/vnext_gt_real/manifests/vnext_train_real.csv
- data/vnext_gt_real/manifests/vnext_val_real.csv

## 6) Check setup (fails loudly with next steps)

```bash
python scripts/check_vnext_setup.py --config configs/vnext_example.yaml
```

## 7) Run a smoke sweep

```bash
python scripts/run_vnext_experiments.py --suite configs/vnext_sweep_gt_small_fz.yaml --device cuda
python scripts/run_vnext_experiments.py --suite configs/vnext_sweep_gt_small_3d.yaml --device cuda
```

## 8) Run a single training job

```bash
python scripts/train_vnext.py --config configs/vnext_example.yaml --device cuda
```

## 9) Evaluate a run

```bash
python scripts/eval_vnext.py --config configs/vnext_example.yaml --run-dir <RUN_DIR> --device cuda
```

## 10) Zip outputs for download

```bash
tar -czf vnext_outputs.tar.gz data/vnext_gt_real_out
```
