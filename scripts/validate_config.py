from __future__ import annotations

import argparse
from pathlib import Path

from vnext.core.config import load_config
from vnext.core.validation import validate_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a vNext YAML config and fail loudly if invalid")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    validate_config(cfg)
    print("OK: config validated")


if __name__ == "__main__":
    main()
