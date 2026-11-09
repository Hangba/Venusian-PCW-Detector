"""Production dual residual CNN, preserving the frozen dual24084 state layout.

The original experiment remains frozen in experiments/architecture_ablation.
Time averaging here matches its MeanTimeAdapter used for mission inference.
"""

import torch
import torch.nn.functional as F
from torch import nn

from venus_express.icw_model import ICWNet


class MultiKernel(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        assert cout % 3 == 0
        self.branches = nn.ModuleList(
            [nn.Conv1d(cin, cout // 3, k, padding=k // 2) for k in (3, 5, 7)]
        )

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.branches], 1)


class Residual(ICWNet):
    def __init__(self, multiscale=False, **kwargs):
        super().__init__(**kwargs)
        if multiscale:
            self.freq_conv2 = MultiKernel(self.freq_conv2.in_channels, self.freq_conv2.out_channels)
        for conv, norm in ((self.freq_conv2, self.freq_gn2), (self.freq_conv3, self.freq_gn3)):
            cache = {}

            def capture(module, args, cache=cache):
                cache["x"] = args[0]

            def skip(module, args, output, cache=cache):
                x = cache.pop("x")
                return output + F.pad(x, (0, 0, 0, output.shape[1] - x.shape[1]))

            conv.register_forward_pre_hook(capture)
            norm.register_forward_hook(skip)


class DualBranch(nn.Module):
    def __init__(self, bounded=False, **kwargs):
        super().__init__()
        if kwargs.get("time_pooling", "mean") != "mean":
            raise ValueError("dual_residual_v1 requires mean time pooling")
        if kwargs.get("channel_dropout_exempt"):
            raise ValueError("dual_residual_v1 does not support channel dropout exemptions")
        self.bounded = bounded
        self.channel_dropout = kwargs["channel_dropout"]
        branch = dict(kwargs, freq_cnn_channels=(20, 36, 48), hidden_dim=48, channel_dropout=0.0)
        self.local = Residual(multiscale=True, **branch)
        self.context = Residual(multiscale=True, **branch)
        self.register_buffer("local_ratio", torch.linspace(0.6, 1.4, 54))

    def forward(self, x, fcp_curve=None, freq_pol=None):
        if x.ndim == 4:
            x = x.mean(dim=-1)
        assert x.ndim == 3 and fcp_curve is not None and freq_pol is not None
        if self.training and self.channel_dropout:
            keep = 1 - self.channel_dropout
            mask = torch.empty(x.shape[0], 4, 1, device=x.device).bernoulli_(keep) / keep
            x = x * mask
        cp = fcp_curve.median(1).values.clamp(min=1e-6)
        target = cp[:, None] * self.local_ratio[None, :]
        # Piecewise-linear interpolation of frozen frequency profiles. No new
        # frequency resolution is claimed. Clamp missing support to edge bins.
        pos = torch.searchsorted(freq_pol.contiguous(), target.contiguous()).clamp(
            1, len(freq_pol) - 1
        )
        lo = pos - 1
        hi = pos
        alpha = ((target - freq_pol[lo]) / (freq_pol[hi] - freq_pol[lo])).clamp(0, 1)
        low = x.gather(2, lo[:, None, :].expand(-1, 4, -1))
        high = x.gather(2, hi[:, None, :].expand(-1, 4, -1))
        local = low + (high - low) * alpha[:, None, :]
        local_logits = self.local(
            local, fcp_curve=torch.ones_like(fcp_curve), freq_pol=self.local_ratio
        )
        context_logits = self.context(x, fcp_curve=fcp_curve, freq_pol=freq_pol)
        return local_logits + (0.5 * torch.tanh(context_logits) if self.bounded else context_logits)


LEGACY_ARCHITECTURE = "icwnet_v2"
CURRENT_ARCHITECTURE = "dual_residual_v1"


def build_production_model(architecture=LEGACY_ARCHITECTURE, **kwargs):
    """Missing historical metadata always selects the original architecture."""
    if architecture == LEGACY_ARCHITECTURE:
        return ICWNet(**kwargs)
    if architecture == CURRENT_ARCHITECTURE:
        kwargs.setdefault("channel_dropout", 0.08)
        return DualBranch(**kwargs)
    raise ValueError(f"Unknown model architecture: {architecture!r}")
