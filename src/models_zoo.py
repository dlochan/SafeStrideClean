# src/models_zoo.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor

# Optional XGBoost
try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

# Optional PyTorch for 1D-CNN
try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.utils.data as torchdata
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    torchdata = None  # type: ignore


# ---- Classic models ----

def make_ridge(params: Dict[str, Any] | None = None) -> Pipeline:
    p = {"alpha": 1.0}
    if params:
        p.update(params)
    model = Ridge(alpha=float(p.get("alpha", 1.0)))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def make_random_forest(params: Dict[str, Any] | None = None, random_state: int = 42) -> RandomForestRegressor:
    p = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "n_jobs": -1,
    }
    if params:
        p.update(params)
    return RandomForestRegressor(
        n_estimators=int(p.get("n_estimators", 300)),
        max_depth=None if p.get("max_depth") in (None, "None") else int(p.get("max_depth")),
        min_samples_leaf=int(p.get("min_samples_leaf", 1)),
        n_jobs=int(p.get("n_jobs", -1)),
        random_state=int(p.get("random_state", random_state)),
    )


def make_hgb(params: Dict[str, Any] | None = None, random_state: int = 42) -> Pipeline:
    p = {
        "max_depth": None,
        "max_iter": 500,
        "learning_rate": 0.05,
        "l2_regularization": 1e-3,
    }
    if params:
        p.update(params)
    hgb = HistGradientBoostingRegressor(
        max_depth=None if p.get("max_depth") in (None, "None") else int(p.get("max_depth")),
        max_iter=int(p.get("max_iter", 500)),
        learning_rate=float(p.get("learning_rate", 0.05)),
        l2_regularization=float(p.get("l2_regularization", 1e-3)),
        random_state=int(p.get("random_state", random_state)),
    )
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", hgb),
    ])


def make_xgb(params: Dict[str, Any] | None = None, random_state: int = 42):  # pragma: no cover
    if XGBRegressor is None:
        raise ImportError("XGBoost not installed. Please install xgboost to use 'xgb'.")
    p = {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    }
    if params:
        p.update(params)
    return XGBRegressor(
        n_estimators=int(p.get("n_estimators", 600)),
        max_depth=int(p.get("max_depth", 6)),
        learning_rate=float(p.get("learning_rate", 0.05)),
        subsample=float(p.get("subsample", 0.8)),
        colsample_bytree=float(p.get("colsample_bytree", 0.8)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        n_jobs=int(p.get("n_jobs", -1)),
        random_state=int(p.get("random_state", random_state)),
    )


# ---- Lightweight PyTorch 1D-CNN ----

class Torch1DCNNRegressor:  # pragma: no cover
    """Minimal 1D-CNN regressor for sequence windows (N, L, C) -> y.
    CPU-safe, deterministic via manual seed. Not used for tabular features.
    """

    def __init__(self, input_channels: int, seq_len: int, hidden: int = 16, lr: float = 1e-3,
                 epochs: int = 10, batch_size: int = 64, random_state: int = 42):
        if torch is None:
            raise ImportError("PyTorch not installed. Install torch to use 'cnn1d'.")
        self.input_channels = int(input_channels)
        self.seq_len = int(seq_len)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self._model: Optional[nn.Module] = None
        self._fitted = False

    def _build(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(self.input_channels, self.hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError("cnn1d expects X with shape (n_samples, seq_len, n_channels)")
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        N, L, C = X.shape
        if C != self.input_channels or L != self.seq_len:
            raise ValueError(f"Expected (*,{self.seq_len},{self.input_channels}), got {X.shape}")
        torch.manual_seed(self.random_state)
        model = self._build()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        tensor_X = torch.from_numpy(X.astype(np.float32)).permute(0, 2, 1)
        tensor_y = torch.from_numpy(y.astype(np.float32))
        ds = torchdata.TensorDataset(tensor_X, tensor_y)
        dl = torchdata.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        self._model = model.eval()
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted")
        if X.ndim != 3:
            raise ValueError("cnn1d expects X with shape (n_samples, seq_len, n_channels)")
        tensor_X = torch.from_numpy(X.astype(np.float32)).permute(0, 2, 1)
        with torch.no_grad():
            pred = self._model(tensor_X).cpu().numpy().reshape(-1)
        return pred


def make_cnn1d(params: Dict[str, Any] | None = None, random_state: int = 42):  # pragma: no cover
    p = {
        "input_channels": 6,
        "seq_len": 100,
        "hidden": 16,
        "lr": 1e-3,
        "epochs": 10,
        "batch_size": 64,
    }
    if params:
        p.update(params)
    return Torch1DCNNRegressor(
        input_channels=int(p["input_channels"]),
        seq_len=int(p["seq_len"]),
        hidden=int(p.get("hidden", 16)),
        lr=float(p.get("lr", 1e-3)),
        epochs=int(p.get("epochs", 10)),
        batch_size=int(p.get("batch_size", 64)),
        random_state=int(p.get("random_state", random_state)),
    )


# ---- Stacking (HGB + RF + Ridge) ----

def make_stacking(params: Dict[str, Any] | None = None, random_state: int = 42) -> StackingRegressor:
    p = {
        "ridge_alpha": 1.0,
        "rf_n_estimators": 300,
        "rf_max_depth": None,
        "hgb_max_iter": 300,
        "hgb_learning_rate": 0.05,
        "final_alpha": 1.0,
        "n_jobs": -1,
    }
    if params:
        p.update(params)

    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=float(p.get("ridge_alpha", 1.0)))),
    ])
    rf = RandomForestRegressor(
        n_estimators=int(p.get("rf_n_estimators", 300)),
        max_depth=None if p.get("rf_max_depth") in (None, "None") else int(p.get("rf_max_depth")),
        n_jobs=int(p.get("n_jobs", -1)),
        random_state=int(p.get("random_state", random_state)),
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=int(p.get("hgb_max_iter", 300)),
        learning_rate=float(p.get("hgb_learning_rate", 0.05)),
        random_state=int(p.get("random_state", random_state)),
    )

    final_estimator = Ridge(alpha=float(p.get("final_alpha", 1.0)))

    ests = [
        ("hgb", hgb),
        ("rf", rf),
        ("ridge", ridge_pipe),
    ]
    return StackingRegressor(
        estimators=ests,
        final_estimator=final_estimator,
        n_jobs=int(p.get("n_jobs", -1)),
        passthrough=False,
    )


def make_model(kind: str, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
    kind = kind.lower()
    if kind in ("ridge", "rdg"):
        return make_ridge(params)
    if kind in ("rf", "random_forest"):
        return make_random_forest(params, random_state=random_state)
    if kind in ("hgb", "histgb", "hist_gradient_boosting"):
        return make_hgb(params, random_state=random_state)
    if kind in ("xgb", "xgboost"):
        return make_xgb(params, random_state=random_state)
    if kind in ("cnn1d", "1dcnn"):
        return make_cnn1d(params, random_state=random_state)
    if kind in ("stack", "stacking"):
        return make_stacking(params, random_state=random_state)
    raise ValueError(f"Unknown model kind: {kind}")
