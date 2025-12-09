# Project: SafeStride Core Pipeline

## Epic
Turn field IMU into:
1) joint kinematics (OpenSense),
2) estimated GRF (baseline ML),
3) knee load surrogates (OpenSim ID),
4) simple risk/quality indicators.

## Stories & Tasks (first pass)
S1. Data ingestion
  T1: dataio.py - load_imu_csv, (optional) load_c3d_grf, resample_grf_to_imu_time
  T2: plotting.py - plot accel magnitude (+ Fz if present)
S2. Preprocessing
  T3: filters.py - bandpass_imu, lowpass_grf
S3. OpenSim integration
  T4: Manual OpenSense in GUI (one sample)
  T5: opensim_wrap.py - run_opensense_ik, run_inverse_dynamics
S4. Baseline ML for GRF
  T6: model_baseline.py - ridge regression IMU->Fz (%BW)
S5. Evaluation
  T7: eval_compare.py - ID with true vs predicted GRF; peak knee moment % error

## Status
✅ (done)
🔄 (in progress)
❌ (blocked)
