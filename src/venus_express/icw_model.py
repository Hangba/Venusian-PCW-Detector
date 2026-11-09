"""
ICWNet v2 - Reduced-capacity frequency CNN

Applies both architectural simplification and stronger regularization:
- Reduced channels: (24, 48, 72)
- Reduced hidden: 96
- Dropout: 0.45
- Channel dropout: 0.08

These defaults match the values used by the production training path and the
event-level ablation experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_pool_mps_safe(x, output_size, pool_type='avg', dim=1):
    """MPS-safe adaptive pooling."""
    if x.device.type == 'mps':
        x_cpu = x.cpu()
        if dim == 1:
            pooled = (F.adaptive_avg_pool1d(x_cpu, output_size) if pool_type == 'avg'
                      else F.adaptive_max_pool1d(x_cpu, output_size))
        else:
            pooled = (F.adaptive_avg_pool2d(x_cpu, output_size) if pool_type == 'avg'
                      else F.adaptive_max_pool2d(x_cpu, output_size))
        return pooled.to(x.device)
    if dim == 1:
        return (F.adaptive_avg_pool1d(x, output_size) if pool_type == 'avg'
                else F.adaptive_max_pool1d(x, output_size))
    return (F.adaptive_avg_pool2d(x, output_size) if pool_type == 'avg'
            else F.adaptive_max_pool2d(x, output_size))


class ICWNet(nn.Module):
    """Combined: reduced capacity + stronger regularization."""

    def __init__(
        self,
        in_channels: int = 4,
        freq_cnn_channels: tuple = (24, 48, 72),  # Reduced
        hidden_dim: int = 96,  # Reduced
        dropout: float = 0.45,
        channel_dropout: float = 0.08,
        fcp_attn_band: tuple = (0.8, 1.2),
        fcp_attn_sigma: float = 0.15,
        use_fcp_attention_gate: bool = True,
        channel_dropout_exempt: tuple = (),
        time_pooling: str = "mean",  # "mean" | "mean_max"
        num_classes: int = 1,  # 1 = binary logit [B]; >1 = multiclass logits [B,C]
        ablate_channels: tuple = (),   # zero these channels in forward (ablation)
        normalize_channels: tuple = (),  # per-sample standardize these channels
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = in_channels
        self.ablate_channels = tuple(int(c) for c in (ablate_channels or ()))
        self.normalize_channels = tuple(int(c) for c in (normalize_channels or ()))
        # Fail loudly on out-of-range indices. The channel layout shrank 8 -> 7
        # -> 4, so an index carried over from an older layout would either point
        # at a different physical quantity or blow up with a bare IndexError
        # deep inside forward(). Callers that record these in model notes (see
        # webapp/training_jobs.py) must be re-checked whenever the layout moves.
        for name, channels in (("ablate_channels", self.ablate_channels),
                               ("normalize_channels", self.normalize_channels)):
            bad = [c for c in channels if not 0 <= c < in_channels]
            if bad:
                raise ValueError(
                    f"{name}={bad} out of range for in_channels={in_channels}; "
                    f"the joint_profile layout is now "
                    f"(0 psd, 1 ellip, 2 wna, 3 fcp_dist)"
                )
        self.time_pooling = str(time_pooling)
        if self.time_pooling not in ("mean", "mean_max"):
            raise ValueError(f"Unknown time_pooling: {time_pooling}")
        # mean_max concatenates a max-over-time profile next to the mean one,
        # so an intermittent wave packet (e.g. 20s burst in the 80s window)
        # is not diluted away by the time average.
        conv_in = in_channels * (2 if self.time_pooling == "mean_max" else 1)
        self.fcp_attn_band = fcp_attn_band
        self.fcp_attn_sigma = fcp_attn_sigma
        self.use_fcp_attention_gate = bool(use_fcp_attention_gate)
        self.channel_dropout = float(channel_dropout)
        # Channels that are never dropped by channel_dropout. v14: the frequency
        # anchor signed (f−f_cp)/f_cp is now ch3 (ch7 -> ch6 -> ch3 as
        # wna_reliability then coh/dop/trans_ratio were removed). Exempting it
        # sharpens out-of-band suppression but was measured to cost in-band
        # recall (0.94 -> 0.84), so it is opt-in.
        self.channel_dropout_exempt = tuple(
            int(c) for c in (channel_dropout_exempt or ())
            if 0 <= int(c) < in_channels
        )

        # Frequency-axis 1D CNN (reduced capacity)
        self.freq_conv1 = nn.Conv1d(conv_in, freq_cnn_channels[0], kernel_size=5, padding=2)
        self.freq_gn1 = nn.GroupNorm(4, freq_cnn_channels[0])
        self.freq_conv2 = nn.Conv1d(
            freq_cnn_channels[0], freq_cnn_channels[1], kernel_size=5, padding=2
        )
        self.freq_gn2 = nn.GroupNorm(6, freq_cnn_channels[1])
        self.freq_conv3 = nn.Conv1d(
            freq_cnn_channels[1], freq_cnn_channels[2], kernel_size=3, padding=1
        )
        self.freq_gn3 = nn.GroupNorm(6, freq_cnn_channels[2])

        # f_cp attention
        self.attn_evidence = nn.Conv1d(freq_cnn_channels[2], 1, kernel_size=1)

        # Classifier head (reduced capacity)
        self.fc1 = nn.Linear(freq_cnn_channels[2], hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, self.num_classes)

    def forward(
        self,
        joint_profile: torch.Tensor,
        fcp_curve: torch.Tensor | None = None,
        freq_pol: torch.Tensor | None = None,
        return_localization: bool = False,
    ) -> torch.Tensor | dict:
        if joint_profile.ndim == 4:
            B, C, _, _ = joint_profile.shape
            is_time_pooled = False
        elif joint_profile.ndim == 3:
            B, C, _ = joint_profile.shape
            is_time_pooled = True
        else:
            raise ValueError(
                f"joint_profile must be [B,C,F,T] or pre-pooled [B,C,F], "
                f"got {tuple(joint_profile.shape)}"
            )
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        # Per-sample standardize selected channels (zero-mean unit-std over F',T).
        # NOTE this is NOT a pure rescale: it erases each channel's absolute
        # level, which for these inputs is physical (ch0's in-band peak sits at
        # 0 by construction; ch1 = -1 means left-hand; ch2 = 0 means parallel).
        # For ch3 = (f - f_cp)/f_cp it is degenerate -- standardizing gives
        # (f - mean f)/std f, identical for every window, so the f_cp anchor is
        # gone (measured: cross-window spread 0.930 -> 9e-6).
        # Measured on the 565/class balanced pool (5-fold x 5 repeats, paired):
        # standardizing ch0 alone -0.012 (Holm p 0.20), all four -0.052
        # (p 0.0001). A pure global rescale of ch0 to unit std is +0.003
        # (p 1.00) -- the channel scale imbalance itself costs nothing, because
        # GroupNorm follows conv1 and Adam's step is per-parameter.
        # See scripts/dev/channel_normalization_cv.py.
        if self.normalize_channels:
            joint_profile = joint_profile.clone()
            for ch in self.normalize_channels:
                x_ch = joint_profile[:, ch]
                reduce_dims = (-2, -1) if not is_time_pooled else (-1,)
                mu = x_ch.mean(dim=reduce_dims, keepdim=True)
                sd = x_ch.std(dim=reduce_dims, keepdim=True).clamp(min=1e-6)
                joint_profile[:, ch] = (x_ch - mu) / sd

        # Ablation: zero selected channels so the model gets no info from them.
        if self.ablate_channels:
            joint_profile = joint_profile.clone()
            for ch in self.ablate_channels:
                joint_profile[:, ch] = 0.0

        # Channel dropout (stronger)
        if self.training and self.channel_dropout > 0.0:
            keep_prob = max(1e-6, 1.0 - self.channel_dropout)
            mask_shape = (B, C, 1) if is_time_pooled else (B, C, 1, 1)
            mask = torch.empty(*mask_shape, device=joint_profile.device, dtype=joint_profile.dtype)
            mask.bernoulli_(keep_prob).div_(keep_prob)
            for ch in self.channel_dropout_exempt:
                mask[:, ch] = 1.0
            joint_profile = joint_profile * mask

        # Pool time dimension
        if is_time_pooled:
            if self.time_pooling != "mean":
                raise ValueError("Pre-pooled profiles are only valid for time_pooling='mean'")
            x = joint_profile
        elif self.time_pooling == "mean_max":
            x = torch.cat(
                [joint_profile.mean(dim=-1), joint_profile.amax(dim=-1)], dim=1
            )  # [B, 8, F']  (4 channels x {mean, max})
        else:
            x = joint_profile.mean(dim=-1)  # [B, 4, F']

        # Frequency CNN (3 layers, reduced channels)
        x = F.relu(self.freq_gn1(self.freq_conv1(x)))  # [B, 24, F']
        x = F.relu(self.freq_gn2(self.freq_conv2(x)))  # [B, 48, F']
        x = F.relu(self.freq_gn3(self.freq_conv3(x)))  # [B, 72, F']

        # f_cp neighborhood attention pooling
        evidence = self.attn_evidence(x).squeeze(1)  # [B, F']

        # Build soft gate
        if self.use_fcp_attention_gate and fcp_curve is not None and freq_pol is not None:
            fcp_median = fcp_curve.median(dim=1, keepdim=True).values  # [B, 1]
            freq_grid = freq_pol.view(1, -1)  # [1, F']
            rel_freq = freq_grid / fcp_median.clamp(min=1e-6)  # [B, F']
            lo, hi = self.fcp_attn_band
            center = (lo + hi) / 2.0
            gate = torch.exp(-((rel_freq - center) ** 2) / (2 * self.fcp_attn_sigma ** 2))
            in_band = ((rel_freq >= lo) & (rel_freq <= hi)).float()
            gate = gate * (0.05 + 0.95 * in_band)
        else:
            gate = torch.ones_like(evidence)

        # Attention weights
        attn_weights = F.softmax(evidence + torch.log(gate + 1e-12), dim=1)  # [B, F']

        # Weighted sum over frequency → [B, 72]
        feat = (x * attn_weights.unsqueeze(1)).sum(dim=2)

        # Classifier (reduced hidden dim, stronger dropout)
        feat = self.dropout(F.relu(self.ln1(self.fc1(feat))))  # [B, 96]
        logits = self.fc2(feat)  # [B, num_classes]
        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # [B] — preserve binary behavior

        if return_localization:
            return {'logits': logits, 'freq_localization': attn_weights}
        return logits

    def predict_proba(self, joint_profile: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convenience method: forward + sigmoid."""
        logits = self.forward(joint_profile, **kwargs)
        if isinstance(logits, dict):
            logits = logits['logits']
        return torch.sigmoid(logits)
