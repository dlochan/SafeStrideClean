# vNext PI Apply (Sandbox) Report

Date: 2025-10-24 20:46

Global baseline (post-cal) coverage: 0.7907887135796781
Per-task apply (sandbox) coverage: 0.7532099035817565 (n_trials=144)

| task | coverage | band_width | n | >=0.75? |
|---|---:|---:|---:|:---:|
| curb | 1.000 | 287.4 | 18 | yes |
| cutting | 0.762 | 238.9 | 27 | yes |
| dynamic | 0.606 | 213.4 | 9 | no |
| normal | 0.568 | 175.3 | 27 | no |
| obstacle | 0.787 | 150.0 | 9 | yes |
| side | 0.770 | 164.3 | 9 | yes |
| squats | 0.720 | 68.1 | 9 | no |
| step | 0.752 | 100.5 | 18 | yes |
| tire | 1.000 | 458.7 | 9 | yes |
| weighted | 0.671 | 77.1 | 9 | no |

Reference (analysis-only pre vs. post-est):
| task | pre_cov | post_cov_est |
|---|---:|---:|
| curb | 0.946 | 0.946 |
| cutting | 0.641 | 0.795 |
| dynamic | 0.557 | 0.792 |
| normal | 0.490 | 0.792 |
| obstacle | 0.703 | 0.786 |
| side | 0.598 | 0.802 |
| squats | 0.525 | 0.796 |
| step | 0.637 | 0.795 |
| tire | 0.868 | 0.868 |
| turn | nan | nan |
| weighted | 0.583 | 0.786 |
