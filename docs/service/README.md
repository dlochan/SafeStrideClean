# SafeStride FastAPI Service

This service exposes a CPU-only inference API for predicting vertical GRF (Fz) from dual-IMU streams using the same windowed feature pipeline as training (`src/features_dual.py`).

- Path: `src/service/app.py`
- Auth: Bearer token via `SERVICE_TOKEN` environment variable (see `src/service/security.py`).
- Models: Load from `models_registry/<subject>/<trial>/model.pkl` or via `?model_path=...`.

## Endpoints

- `GET /health` -> `{ "ok": true }`
- `POST /predict` -> `[{ time_s, Fz_%BW, Fz_N }, ...]`

Request body shape:
```json
{
  "subject": "AB01",
  "trial": "cutting_leftfast",
  "bw_kg": 75,
  "fs": 200,
  "imu": [
    {
      "time_s": 0.0,
      "ax_thigh": 0, "ay_thigh": 0, "az_thigh": 0,
      "gx_thigh": 0, "gy_thigh": 0, "gz_thigh": 0,
      "ax_shank": 0, "ay_shank": 0, "az_shank": 0,
      "gx_shank": 0, "gy_shank": 0, "gz_shank": 0
    }
  ]
}
```

## Run locally

- Set a token: on Windows PowerShell
```powershell
$env:SERVICE_TOKEN = "devtoken"
```
- Start server (auto-reload):
```powershell
uvicorn src.service.app:app --reload
```

## curl example

```bash
curl -X POST \
  -H "Authorization: Bearer devtoken" \
  -H "Content-Type: application/json" \
  "http://localhost:8000/predict?model_path=results/out_gt_AB01_cutting_leftfast/model.pkl" \
  -d @request.json
```

Where `request.json` matches the request schema above.

## Python SDK example

Using `sdk/safestride_client.py`:

```python
import pandas as pd
from sdk.safestride_client import SafeStrideClient

client = SafeStrideClient(base_url="http://localhost:8000", token="devtoken")

# imu_df must include the 13 columns: time_s and 12 sensor channels
imu_df = pd.DataFrame([...])

pred_df = client.predict(
    subject="AB01",
    trial="cutting_leftfast",
    bw_kg=75.0,
    fs=200.0,
    imu_df=imu_df,
)
print(pred_df.head())
```

## Dev tests

Run API tests:
```powershell
pytest -q tests/test_service.py
```

## Notes

- Feature builder: `build_dual_features()` aligns outputs to window centers `time_s`.
- Predictions are % body-weight (`Fz_%BW`), converted to Newtons via `Fz_N = Fz_%BW * (bw_kg * 9.80665) / 100`.
