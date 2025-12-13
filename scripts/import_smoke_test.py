from __future__ import annotations

import importlib
import sys


MODULES = [
    "vnext",
    "vnext.core",
    "vnext.core.config",
    "vnext.core.metrics",
    "vnext.core.normalization",
    "vnext.core.paths",
    "vnext.data",
    "vnext.data.datasets",
    "vnext.data.imu_schema",
    "vnext.models",
    "vnext.models.vnext_fz",
    "vnext.models.vnext_grf3d",
    "vnext.experiments",
    "vnext.experiments.registry",
]


def main() -> None:
    failed: list[str] = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {type(e).__name__}: {e}")

    if failed:
        msg = "\n".join(failed)
        raise SystemExit(
            "Import smoke test failed. Ensure you installed the repo in editable mode:\n"
            "  python -m pip install -e .\n\n"
            "Failures:\n" + msg
        )

    print("OK: import smoke test passed")


if __name__ == "__main__":
    main()
