# SafeStride MVP IMU Input Schema

- Required canonical accelerometer columns:
  - ax, ay, az
- Optional canonical gyroscope columns:
  - gx, gy, gz
- Column normalization rules (case-insensitive):
  - Ax, Ay, Az -> ax, ay, az
  - Wx, Wy, Wz -> gx, gy, gz
  - acc_x, acc_y, acc_z -> ax, ay, az
  - accel_x, acceleration_x, linearacc_x -> ax (same for y, z)
  - gyro_x, gyr_x -> gx (same for y, z)
  - Spaces and dashes converted to underscores
  - Sensor tags preserved, e.g. ax_shank remains ax_shank
- Multi-sensor files are accepted if any single tag group has a full set of ax_<tag>, ay_<tag>, az_<tag>.

- Time handling (time_s):
  - If a numeric, monotonic time_s exists, it is used (sorted if necessary).
  - Else, if fs_hz is known from configs/dataset.yaml, synthesize time_s = arange(N)/fs_hz.
  - Else, the file is rejected with reason: "no_time_s and no fs_hz".

- Rejection reasons logged to logs/validation_mvp_risk.log and summarized in docs/validation/VALIDATION_REPORT_MVP_RISK_v1.md:
  - bad_columns (missing accelerometer axes)
  - no_time_s (no time_s and no fs_hz)
  - too_short_for_window (insufficient samples for model window)
  - other_format_issue
