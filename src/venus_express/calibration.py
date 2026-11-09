"""Probability calibration utilities for ICW model logits.

The classifier is trained with class/sample weighting, so sigmoid(logit) is a
ranking score rather than a calibrated probability.  We fit monotonic Platt
scaling on the held-out validation fold and optionally correct its class prior
for a deployment prevalence supplied by the user.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _logit(probability: float) -> float:
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def calibration_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(targets, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    ece = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"brier": brier, "log_loss": log_loss, "ece_10": float(ece)}


def fit_platt_calibration(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    target_prevalence: float | None = None,
) -> dict[str, Any]:
    """Fit monotonic ``sigmoid(scale * logit + bias)`` calibration."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = (np.asarray(targets, dtype=np.float64).reshape(-1) > 0.0).astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    source_prevalence = float(y.mean()) if y.size else float("nan")
    if y.size < 10 or np.unique(y).size < 2:
        return {
            "method": "identity",
            "scale": 1.0,
            "bias": 0.0,
            "n_samples": int(y.size),
            "n_positive": int(y.sum()),
            "source_prevalence": source_prevalence,
            "target_prevalence": target_prevalence,
            "reason": "validation fold is too small or contains one class",
        }

    def objective(params: np.ndarray) -> float:
        log_scale, bias = float(params[0]), float(params[1])
        scale = np.exp(np.clip(log_scale, -5.0, 5.0))
        z = scale * x + bias
        nll = np.mean(np.logaddexp(0.0, z) - y * z)
        return float(nll + 1e-4 * (log_scale * log_scale + bias * bias))

    result = minimize(
        objective,
        np.asarray([0.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (-30.0, 30.0)],
    )
    scale = float(np.exp(np.clip(result.x[0], -5.0, 5.0)))
    fit_bias = float(result.x[1])
    prior_adjustment = 0.0
    if target_prevalence is not None:
        target_prevalence = float(target_prevalence)
        if not 0.0 < target_prevalence < 1.0:
            raise ValueError("target_prevalence must be between 0 and 1")
        prior_adjustment = _logit(target_prevalence) - _logit(source_prevalence)
    bias = fit_bias + prior_adjustment

    before = _sigmoid(x)
    after_fit = _sigmoid(scale * x + fit_bias)
    after = _sigmoid(scale * x + bias)
    return {
        "method": "platt",
        "scale": scale,
        "bias": float(bias),
        "fit_bias": fit_bias,
        "prior_adjustment": float(prior_adjustment),
        "n_samples": int(y.size),
        "n_positive": int(y.sum()),
        "source_prevalence": source_prevalence,
        "target_prevalence": target_prevalence,
        "optimizer_success": bool(result.success),
        "metrics_before": calibration_metrics(y, before),
        "metrics_after_fit": calibration_metrics(y, after_fit),
        "metrics_after": calibration_metrics(y, after),
    }


def apply_logit_calibration(
    logits: np.ndarray, calibration: dict[str, Any] | None
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not calibration or calibration.get("method") not in {"platt", "identity"}:
        return _sigmoid(values)
    scale = float(calibration.get("scale", 1.0))
    bias = float(calibration.get("bias", 0.0))
    return _sigmoid(scale * values + bias)
