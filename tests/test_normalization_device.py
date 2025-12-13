import torch

from vnext.core.normalization import ChannelNormStats


def test_channel_norm_stats_cpu_device_safe():
    x = torch.randn(2, 8, 12, device="cpu")
    stats = ChannelNormStats(mean=torch.zeros(12, device="cpu"), std=torch.ones(12, device="cpu"))
    y = stats.normalize(x)
    assert y.device.type == "cpu"
    assert y.shape == x.shape


def test_channel_norm_stats_cuda_device_safe_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.randn(2, 8, 12, device="cuda")
    stats = ChannelNormStats(mean=torch.zeros(12, device="cpu"), std=torch.ones(12, device="cpu"))
    y = stats.normalize(x)
    assert y.device.type == "cuda"
    assert y.shape == x.shape
