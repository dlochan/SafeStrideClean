from pathlib import Path

import torch

try:
    import vnext  # noqa: F401
except ModuleNotFoundError as e:
    raise SystemExit(
        "Could not import 'vnext'. Install the repo in editable mode from the repo root: "
        "`python -m pip install -e .`"
    ) from e

from vnext.core.normalization import ChannelNormStats


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")

    # Dummy input on target device, matching vNext shape (B, T, C) with C=12.
    x = torch.randn(4, 64, 12, device=device)

    # Norm stats intentionally created on CPU to simulate loaded-from-disk
    mean = torch.zeros(12, device="cpu")
    std = torch.ones(12, device="cpu")
    stats = ChannelNormStats(mean=mean, std=std)

    y = stats.normalize(x)

    print(f"x.device={x.device}, y.device={y.device}")
    print(f"y mean={y.mean().item():.4f}, std={y.std().item():.4f}")


if __name__ == "__main__":
    main()
