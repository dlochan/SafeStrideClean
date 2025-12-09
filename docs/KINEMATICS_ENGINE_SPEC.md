# Kinematics Engine v1 (IMU-derived surrogates)

Scope: minimal, robust, literature-backed surrogates from knee-pair IMUs (thigh & shank), optionally pelvis/trunk when present. No OpenSim. Heuristics only.

Features per trial
- knee_flex_ic_deg
- knee_flex_min_deg
- hip_flex_ic_deg
- trunk_lean_ic_deg
- frontal_sur_idx (ML accel/roll gyro surrogate during stance)
- stance_time_s (from vGRF > 5% BW if available, else IMU contact surrogate)
- contact_class (walk vs run from stance time)

Derivations
- Orientation surrogate: low-pass accelerometer to estimate gravity vector; pitch tilt = atan2(ax, sqrt(ay^2+az^2)) [deg]; knee_flex ≈ pitch_thigh − pitch_shank.
- IC: first sample with vGRF > 5% BW; if not available, detect IMU contact via vertical accel zero-cross + peak heuristic.
- Trunk/pelvis: when a trunk/pelvis IMU column is present, compute tilt; else NaN with a logged assumption.
- Frontal surrogate: RMS of mediolateral accel and roll gyro within stance; index = normed RMS vs vertical accel RMS.
- Contact class: walk if stance_time_s ≥ 0.25; run if < 0.25.

Evidence links (study_registry keys)
- PubMed20303276_KneeFlexGRF: greater knee flexion reduces GRF/moments.
- PMC5763249_LandingTasks: flexed landing reduces impact loads.
- OARSI2020_WalkingImpact: loading rate associated with knee pain risk.
- PubMed23899938_TTP_Landing: time-to-peak / RFD characteristics in repeated landings.

Outputs
- docs/vNext_multisignal/kin_surrogates.csv (one row per trial with fields above)
- logs/kinematics_engine.log (assumptions and resume decisions)

Constraints
- Resume-safe: do not recompute rows when present unless --force.
- No model retraining. Evidence linked via configs/clinical_thresholds.yaml and docs/evidence/study_registry.csv.
