from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .imu_schema import validate_canonical_imu_df, TIME_COL, EXPECTED_IMU_COLUMNS


@dataclass
class TrialRecord:
    trial_id: str
    imu_path: Path
    grf_path: Optional[Path]


class DualIMUTrialDataset(Dataset):
    """Minimal dual-IMU + GRF dataset for vNext scaffolding.

    This dataset:
    - Reads a manifest CSV listing trial IDs and file paths.
    - Loads canonical IMU CSVs and optional GRF CSVs.
    - Returns tensors ready for sequence modeling (no windowing yet).

    Manifest CSV columns:
    - trial_id
    - imu_path
    - grf_path (optional; may be empty or NaN if no GRF is available)

    Each item returns a dict with keys:
    - "trial_id": str
    - "imu": torch.FloatTensor of shape (T, C)
    - "grf_v": Optional[torch.FloatTensor] of shape (T, 1) or None
    """

    def __init__(
        self,
        manifest_path: str | Path,
        grf_axes: str = "fz",
        target_grf_column: str | None = None,
    ) -> None:
        logger = logging.getLogger(__name__)
        grf_axes = str(grf_axes).lower()
        if grf_axes == "all":
            grf_axes = "3d"
        if grf_axes not in {"fz", "3d"}:
            raise ValueError(
                f"Unsupported grf_axes '{grf_axes}', expected 'fz' or '3d' (alias: 'all' -> '3d')"
            )

        if grf_axes != "fz" and target_grf_column is not None:
            raise ValueError(
                "target_grf_column is only supported when grf_axes='fz'; "
                "for multi-axis GRF, use the canonical Fx/Fy/Fz columns instead."
            )

        self.manifest_path = Path(manifest_path)
        self.grf_axes = grf_axes
        # Optional explicit GRF column to use as the regression target (e.g. "Fz_N" or "Fz_BW").
        # If None, we fall back to the default candidate-based selection logic in _load_grf.
        self.target_grf_column = target_grf_column
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        required_cols = ["trial_id", "imu_path"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Manifest is missing required column: {col}")

        self.records: list[TrialRecord] = []
        for _, row in df.iterrows():
            trial_id = str(row["trial_id"])
            imu_path = Path(str(row["imu_path"]))
            raw_grf = row.get("grf_path") if "grf_path" in df.columns else None
            grf_path: Optional[Path]
            if isinstance(raw_grf, str) and raw_grf.strip():
                grf_path = Path(raw_grf)
            else:
                grf_path = None
            self.records.append(TrialRecord(trial_id=trial_id, imu_path=imu_path, grf_path=grf_path))

        chosen_col: str | None = None
        chosen_units = "unknown"
        if self.grf_axes == "fz":
            try:
                import csv

                sample_grf_path = next(
                    (r.grf_path for r in self.records if r.grf_path is not None and r.grf_path.exists()),
                    None,
                )
                if sample_grf_path is not None:
                    with sample_grf_path.open("r", encoding="utf-8") as f:
                        header = next(csv.reader(f))

                    if self.target_grf_column is not None and self.target_grf_column in header:
                        chosen_col = self.target_grf_column
                    else:
                        chosen_col = next((c for c in ("Fz_N", "Fz_BW", "Fz_%BW") if c in header), None)

                    if chosen_col is not None:
                        if chosen_col.endswith("_N"):
                            chosen_units = "N"
                        elif chosen_col.endswith("_BW"):
                            chosen_units = "BW"
                        elif chosen_col.endswith("_%BW"):
                            chosen_units = "%BW"
            except Exception:
                pass

        logger.info(
            "DualIMUTrialDataset: grf_axes=%s, target_grf_column_config=%s, chosen_grf_column=%s, units=%s",
            self.grf_axes,
            self.target_grf_column,
            chosen_col,
            chosen_units,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def _load_imu(self, path: Path) -> torch.Tensor:
        df = pd.read_csv(path)
        validate_canonical_imu_df(df)
        # Ensure deterministic column order: expected schema minus time column
        feature_cols = [c for c in EXPECTED_IMU_COLUMNS if c != TIME_COL]
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return torch.from_numpy(X)  # (T, C)

    def _load_grf(self, path: Optional[Path]) -> Optional[torch.Tensor]:
        if path is None or not path.exists():
            return None
        df = pd.read_csv(path)
        # Fz-only or 3D GRF depending on configuration.
        # NOTE: For fz mode we currently take the first available vertical GRF
        # column in fz_candidates (typically Fz_N for GT canonical files). The
        # regression target is used in its native units; only IMU inputs are
        # normalized via ChannelNormStats, not the GRF target.
        fz_candidates = ["Fz_N", "Fz_BW", "Fz_%BW"]
        fx_candidates = ["Fx_N", "Fx_BW", "Fx_%BW"]
        fy_candidates = ["Fy_N", "Fy_BW", "Fy_%BW"]

        if self.grf_axes == "fz":
            # If an explicit target GRF column was configured and is present,
            # prefer it. Otherwise fall back to the first available candidate
            # (typically Fz_N for GT canonical files).
            col = None
            if self.target_grf_column is not None and self.target_grf_column in df.columns:
                col = self.target_grf_column
            else:
                col = next((c for c in fz_candidates if c in df.columns), None)
            if col is None:
                return None
            y = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            y = y.reshape(-1, 1)
            return torch.from_numpy(y)  # (T, 1)

        # 3D GRF: Fx, Fy, Fz (in that order)
        fx_col = next((c for c in fx_candidates if c in df.columns), None)
        fy_col = next((c for c in fy_candidates if c in df.columns), None)
        fz_col = next((c for c in fz_candidates if c in df.columns), None)
        if not (fx_col and fy_col and fz_col):
            # Not all axes available; treat as missing GRF for this trial
            return None
        fx = pd.to_numeric(df[fx_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        fy = pd.to_numeric(df[fy_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        fz = pd.to_numeric(df[fz_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        Y = np.stack([fx, fy, fz], axis=-1)  # (T, 3)
        return torch.from_numpy(Y)

    def __getitem__(self, idx: int) -> Dict[str, object]:  # type: ignore[override]
        rec = self.records[idx]
        imu = self._load_imu(rec.imu_path)
        grf_v = self._load_grf(rec.grf_path)
        return {
            "trial_id": rec.trial_id,
            "imu": imu,
            "grf_v": grf_v,
        }


class WindowedIMUDataset(Dataset):
    """Windowed dataset built on top of DualIMUTrialDataset.

    This wraps a per-trial dataset and returns fixed-length windows for
    training temporal models.

    Each item contains:
    - "trial_id": str
    - "imu": (T_win, C) tensor
    - "grf_v": (T_win, 1) tensor or None
    - "has_grf": bool
    - "start_idx": int
    """

    def __init__(
        self,
        base_dataset: DualIMUTrialDataset,
        window_size: int,
        window_stride: int,
        require_grf: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")

        self.base = base_dataset
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.require_grf = bool(require_grf)

        # Precompute (trial_idx, start_idx, has_grf) records
        self._windows: List[Tuple[int, int, bool]] = []
        for trial_idx in range(len(self.base)):
            rec = self.base[trial_idx]
            imu: torch.Tensor = rec["imu"]  # (T, C)
            grf_v = rec["grf_v"]            # Optional[Tensor]
            T = int(imu.shape[0])
            has_grf = grf_v is not None and grf_v.numel() > 0

            if self.require_grf and not has_grf:
                continue

            start = 0
            while start + self.window_size <= T:
                self._windows.append((trial_idx, start, has_grf))
                start += self.window_stride

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, object]:  # type: ignore[override]
        trial_idx, start, has_grf = self._windows[idx]
        rec = self.base[trial_idx]
        imu: torch.Tensor = rec["imu"]
        grf_v = rec["grf_v"]

        end = start + self.window_size
        X = imu[start:end, :]
        y = None
        if has_grf and grf_v is not None:
            y = grf_v[start:end, :]

        return {
            "trial_id": rec["trial_id"],
            "imu": X,
            "grf_v": y,
            "has_grf": bool(has_grf and y is not None),
            "start_idx": int(start),
        }
