# Running SafeStride locally vs in the cloud

This document explains how the new path abstraction (`SafeStridePaths`) lets you run the
same scripts on your current `E:` layout and on a POSIX-style cloud mount by changing
only configuration or environment variables.

## Defaults: local `E:` behavior (no env overrides)

By default, when you do **not** set any `SAFESTRIDE_*_ROOT` environment variables,
`tools.path_config` resolves roots as:

- `DATASET_ROOT`: `E:/safestride/datasets/ProcessedData` (from `configs/dataset.yaml`)
- `DATA_ROOT`: `E:/safestride/data/raw`
- `WORK_ROOT`: `E:/safestride/data/working`
- `OUT_ROOT`: `E:/safestride/out`
- `RELEASE_ROOT`: `E:/safestride/release`
- `LOG_ROOT`: `E:/safestride/logs`
- `DOC_ROOT`: `E:/safestride/docs`

If `E:` is unavailable, these fall back to repo-local mirrors under `c:/Users/.../Documents/safestride`
(e.g. `data/raw`, `data/working`, `out`, `release`, `logs`, `docs`).

Scripts like `scripts/run_safestride_mvp.py` that use `SafeStridePaths.from_env_or_config()`
will therefore behave exactly as they do today on your Windows machine when you do **not**
set any overrides.

## Cloud / POSIX usage via environment variables

In a cloud or containerized environment, you typically mount storage into POSIX paths such
as `/data` or `/mnt/safestride_out`. To point SafeStride at these locations without touching
code, set the following environment variables before running any scripts:

- `SAFESTRIDE_DATA_ROOT`   → e.g. `/mnt/safestride_data`
- `SAFESTRIDE_WORK_ROOT`   → e.g. `/mnt/safestride_work`
- `SAFESTRIDE_OUT_ROOT`    → e.g. `/mnt/safestride_out`
- `SAFESTRIDE_RELEASE_ROOT`→ e.g. `/mnt/safestride_release`
- `SAFESTRIDE_LOG_ROOT`    → e.g. `/mnt/safestride_logs`
- `SAFESTRIDE_DOC_ROOT`    → e.g. `/mnt/safestride_docs`

The `SafeStridePaths.from_env_or_config()` helper in `tools.path_config` reads these
variables and will route all downstream path usages accordingly.

### Example: MVP runner

**Local E: run (no env overrides):**

```bash
python scripts/run_safestride_mvp.py \
  --in_root E:/safestride/new_trials \
  --run_kin --run_risk
```

**Cloud / container run (POSIX mounts):**

```bash
export SAFESTRIDE_DATA_ROOT=/data
export SAFESTRIDE_OUT_ROOT=/out
export SAFESTRIDE_RELEASE_ROOT=/release
export SAFESTRIDE_LOG_ROOT=/logs
export SAFESTRIDE_DOC_ROOT=/docs

python scripts/run_safestride_mvp.py \
  --in_root /data/new_trials \
  --run_kin --run_risk
```

In both cases, the MVP runner will discover the frozen Clinical_v1 model and write metrics
and outputs under the appropriate `out_root` and `doc_root` locations for that environment.

## Config-driven overrides (for vNext)

For new vNext training entrypoints (e.g. `scripts/train_vnext.py`), configs may include a
`paths:` block such as:

```yaml
paths:
  data_root: /mnt/safestride_data
  out_root: /mnt/safestride_out
```

`SafeStridePaths.from_env_or_config(cfg.get("paths"))` will then resolve roots using the
following priority:

1. `paths:` values from the config (if provided),
2. `SAFESTRIDE_*_ROOT` environment variables,
3. Existing `E:` / repo-local defaults from `tools.path_config`.

This keeps existing Clinical_v1 + MVP behavior intact while making the same codebase
ready for GPU/cloud training with vNext models.
