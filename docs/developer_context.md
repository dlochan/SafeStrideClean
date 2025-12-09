# Developer Context (SafeStride)

## Language & Style
- Python 3.10; PEP8; clear docstrings (Args/Returns/Raises).
- Small functions; prefer readable code over clever code.
- Print simple, helpful error messages.

## Data Conventions
- IMU DataFrame columns: time_s, ax, ay, az, gx, gy, gz
- GRF DataFrame columns: time_s, Fx_N, Fy_N, Fz_N
- Units: seconds, Newtons, degrees (unless noted). Monotonic time.

## File Structure
safestride/
  dataio.py            # load IMU .csv, GRF .c3d, resample to IMU time
  filters.py           # bandpass IMU, lowpass GRF
  plotting.py          # quick sanity plots
  opensim_wrap.py      # OpenSense IK + Inverse Dynamics wrappers
  model_baseline.py    # baseline IMU->vertical GRF model
  eval_compare.py      # compare true vs predicted GRF & knee moments
  tests/
  docs/

## Acceptance for v0
- dataio.py: imports; loads IMU CSV; (GRF C3D optional for now).
- plotting.py: saves a PNG of accel magnitude + (if available) Fz.
- filters.py: filters without producing NaNs/Infs.
- opensim_wrap.py: callable stubs; IK/ID wired later after sample works.
- model_baseline.py: trains a simple ridge baseline on sample (later).
- eval_compare.py: runs ID with true vs predicted GRF (later).

## Prompting Rules
- When asked to "create file X", return only that file’s code.
- If an error occurs, propose a minimal fix to the exact function, not a rewrite.
