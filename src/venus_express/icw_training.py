"""Weighted training objective used by the paper."""

import torch
import torch.nn.functional as F


def weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    use_sample_weight: bool = True,
) -> torch.Tensor:
    """
    Compute weighted BCE-with-logits using per-sample weights.

    Args:
        logits: model raw logits, shape [B]
        target: target confidence in [0,1], shape [B]
        sample_weight: per-sample weight, shape [B]
        pos_weight: positive-class weight (scalar tensor) to handle positive/negative imbalance
    """
    per_sample_loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction='none',
        pos_weight=pos_weight,
    )
    if use_sample_weight:
        weighted_loss = per_sample_loss * sample_weight
        return weighted_loss.sum() / sample_weight.sum().clamp_min(1e-8)
    return per_sample_loss.mean()
