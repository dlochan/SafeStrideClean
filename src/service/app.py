# src/service/app.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.features_dual import build_dual_features
from .security import verify_bearer_token


app = FastAPI(title="SafeStride Service", version="1.0.0")


class IMUSample(BaseModel):
    time_s: float
    ax_thigh: float
    ay_thigh: float
    az_thigh: float
    gx_thigh: float
    gy_thigh: float
    gz_thigh: float
    ax_shank: float
    ay_shank: float
    az_shank: float
    gx_shank: float
    gy_shank: float
    gz_shank: float


class PredictRequest(BaseModel):
    subject: str = Field(..., example="AB01")
    trial: str = Field(..., example="cutting_leftfast")
    bw_kg: float = Field(..., gt=0)
    fs: float = Field(200.0, gt=0)
    imu: List[IMUSample]


class PredictPoint(BaseModel):
    time_s: float
    Fz_pctBW: float = Field(..., alias="Fz_%BW")
    Fz_N: float

    class Config:
        allow_population_by_field_name = True


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/predict", dependencies=[Depends(verify_bearer_token)], response_model=List[PredictPoint])
def predict(req: PredictRequest, model_path: Optional[str] = Query(default=None, description="Optional explicit path to model.pkl")):
    # Build DataFrame from IMU samples
    if not req.imu:
        raise HTTPException(status_code=400, detail="imu list is empty")

    imu_df = pd.DataFrame([s.dict() for s in req.imu])
    # Ensure sorted by time
    imu_df = imu_df.sort_values("time_s").reset_index(drop=True)

    # Compute windowed features using dual-sensor builder
    try:
        X, t_feat = build_dual_features(imu_df, fs=float(req.fs), window_ms=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"feature_build_failed: {e}")

    # Resolve model path
    model_file: Path
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = Path("models_registry") / req.subject / req.trial / "model.pkl"

    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"model_not_found: {model_file}")

    # Load model (CPU-only scikit-learn style)
    try:
        model = joblib.load(model_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model_load_failed: {e}")

    # Predict %BW and convert to Newtons
    try:
        X_np = X.values.astype(float)
        y_pct = np.asarray(model.predict(X_np)).reshape(-1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model_predict_failed: {e}")

    bwN = float(req.bw_kg) * 9.80665
    y_N = (y_pct * bwN) / 100.0

    out = [
        {
            "time_s": float(t_feat.iloc[i]),
            "Fz_%BW": float(y_pct[i]),
            "Fz_N": float(y_N[i]),
        }
        for i in range(len(y_pct))
    ]
    return out
