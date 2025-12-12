import sys
from pathlib import Path

import torch


repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from vnext.core.normalization import ChannelNormStats


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")

    # Dummy input on target device
    x = torch.randn(4, 12, 64, device=device)

    # Norm stats intentionally created on CPU to simulate loaded-from-disk
    mean = torch.zeros(12, device="cpu")
    std = torch.ones(12, device="cpu")
    stats = ChannelNormStats(mean=mean, std=std)

    y = stats.normalize(x)

    print(f"x.device={x.device}, y.device={y.device}")
    print(f"y mean={y.mean().item():.4f}, std={y.std().item():.4f}")


if __name__ == "__main__":
    main()
