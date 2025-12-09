from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import os


@dataclass
class SafeStridePaths:
    """Centralized root paths for SafeStride vNext.

    Resolution priority for each root:
    1. Explicit value from ``cfg_paths`` (if provided).
    2. Environment variable (e.g. ``SAFESTRIDE_DATA_ROOT``).
    3. Repo-local default (e.g. ``data/``, ``work/``, ``out/``, ``release/``).
    """

    data_root: Path
    work_root: Path
    out_root: Path
    release_root: Path

    @classmethod
    def from_env_or_defaults(cls, cfg_paths: Optional[Dict[str, str]] = None) -> "SafeStridePaths":
        """Construct SafeStridePaths from config and environment.

        Parameters
        ----------
        cfg_paths:
            Optional mapping with keys like "data_root", "work_root",
            "out_root", "release_root". If present, these override
            environment variables and defaults.
        """

        def _pick(name: str, default: str) -> Path:
            # 1) explicit config value
            if cfg_paths is not None:
                v = cfg_paths.get(name)
                if isinstance(v, str) and v:
                    return Path(v)

            # 2) environment variable
            env_name = f"SAFESTRIDE_{name.upper()}"
            v_env = os.getenv(env_name)
            if v_env:
                return Path(v_env)

            # 3) repo-local default path
            return Path(default)

        data_default = "data"
        work_default = "work"
        out_default = "out"
        release_default = "release"

        return cls(
            data_root=_pick("data_root", data_default),
            work_root=_pick("work_root", work_default),
            out_root=_pick("out_root", out_default),
            release_root=_pick("release_root", release_default),
        )
