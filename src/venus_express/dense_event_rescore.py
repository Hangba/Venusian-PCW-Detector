"""Opt-in 1-s CNN rescore of all preliminary groups in a shared batch queue."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from venus_express.calibration import apply_logit_calibration
from venus_express.event_rescue import _format_time, _seconds
from venus_express.icw_inference import (
    _build_joint_profile_inference,
    _validate_precomputed_features,
    extract_features_for_window,
)


def plan_dense_group_windows(
    results: list[dict],
    *,
    threshold: float,
    margin_seconds: int = 20,
    max_windows: int = 1_000_000,
) -> tuple[list[dict], list[dict]]:
    """Plan complete 80-s windows inside each [group.start-20, group.end+20].

    Only finite original raw CNN scores >= threshold form the preliminary
    groups. The strict gap <60-s rule is unchanged. Coordinates are UTC seconds.
    The two expanded data boundaries are NOT the min/max window start times.
    """
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if type(margin_seconds) is not int or margin_seconds < 0:
        raise ValueError("margin_seconds must be a nonnegative integer")
    if type(max_windows) is not int or max_windows <= 0:
        raise ValueError("max_windows must be a positive integer")
    accepted = []
    for index, row in enumerate(results):
        value = row.get("uncalibrated_score")
        if value is None or not math.isfinite(float(value)) or float(value) < threshold:
            continue
        start, end = _seconds(row["start"]), _seconds(row["end"])
        if end - start < 80:
            raise ValueError("original nominal 80-s CNN windows must span at least 80 seconds")
        accepted.append((start, end, index))
    accepted.sort()
    groups = []
    for start, end, index in accepted:
        if not groups or start - groups[-1]["end_s"] >= 60:
            groups.append(
                {
                    "parent_id": len(groups),
                    "start_s": start,
                    "end_s": end,
                    "source_window_indices": [index],
                }
            )
        else:
            groups[-1]["end_s"] = max(groups[-1]["end_s"], end)
            groups[-1]["source_window_indices"].append(index)
    count = sum(group["end_s"] - group["start_s"] - 80 + 2 * margin_seconds + 1 for group in groups)
    if count > max_windows:
        raise ValueError("dense-window budget exceeded; no partial scoring was started")
    planned = []
    for group in groups:
        left = group["start_s"] - margin_seconds
        last_start = group["end_s"] + margin_seconds - 80
        group["scan_start_s"], group["scan_end_s"] = left, last_start + 80
        for start in range(left, last_start + 1):
            planned.append({"parent_id": group["parent_id"], "start_s": start, "end_s": start + 80})
    return groups, planned


def make_precomputed_window_builder(precomputed: dict, *, max_gap_seconds: int = 4):
    """Reuse a feature chunk and existing full-band model profile construction.

    Return (profile, fcp_curve) per UTC-second bin or None when a full finite
    nominal 1-Hz window is unavailable. Keep the original path for intact
    windows. In missing integer-second bins, linearly interpolate the cached
    input time series only when both observed neighbours exist and at most
    max_gap_seconds are missing. No extrapolation or cache mutation occurs.
    Cached polarization features are time-interpolated, not recomputed; this
    does not claim their underlying time resolution has become 1 s. Final
    event physics must still use the original observations.
    For multiple chunks the caller may dispatch to these builders by UTC time.
    """
    n, times_us = _validate_precomputed_features(precomputed)
    if type(max_gap_seconds) is not int or max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be a nonnegative integer")
    if n > 1 and np.any(np.diff(times_us) <= 0):
        raise ValueError("cached times must be strictly increasing")
    second_bins = times_us // 1_000_000
    frequencies = np.asarray(precomputed["freq_pol"], dtype=np.float32)
    if frequencies.size == 0 or not np.isfinite(frequencies).all():
        raise ValueError("frequency grid must be finite and nonempty")
    if np.asarray(precomputed["wna_deg"]).shape != np.asarray(precomputed["ellip"]).shape:
        raise ValueError("WNA and ellipticity cache shapes must match")

    def build(start_s, end_s):
        build.last_interpolation = {"interpolated_seconds": 0, "maximum_missing_gap_seconds": 0,
                                    "maximum_allowed_gap_seconds": max_gap_seconds}
        if end_s - start_s != 80:
            raise ValueError("model window duration must be 80 seconds")
        index = int(np.searchsorted(second_bins, start_s))
        expected = start_s + np.arange(80, dtype=np.int64)
        if index + 80 <= n and np.array_equal(second_bins[index : index + 80], expected):
            _, bx, by, ellip, wna, fcp = extract_features_for_window(precomputed, index, 80)
        else:
            right = np.searchsorted(second_bins, expected)
            present = right < n
            present[present] &= second_bins[right[present]] == expected[present]
            left = right - 1
            missing = ~present
            if np.any(missing & ((left < 0) | (right >= n))):
                build.last_interpolation["unavailable_reason"] = "no_bracketing_observations"
                return None
            gap = np.zeros(80, dtype=np.int64)
            gap[missing] = second_bins[right[missing]] - second_bins[left[missing]] - 1
            if np.any(gap[missing] > max_gap_seconds):
                build.last_interpolation["unavailable_reason"] = "gap_exceeds_interpolation_limit"
                build.last_interpolation["maximum_missing_gap_seconds"] = int(gap.max())
                return None
            # First sample per observed second matches the intact-window path.
            left[missing] = np.searchsorted(second_bins, second_bins[left[missing]])
            alpha = np.zeros(80, dtype=np.float64)
            alpha[missing] = (expected[missing] - second_bins[left[missing]]) / (second_bins[right[missing]] - second_bins[left[missing]])
            arrays = []
            for key in ("bfac_x", "bfac_y", "ellip", "wna_deg", "fcp_curve"):
                values = np.asarray(precomputed[key])
                result = values[..., right].astype(np.float32, copy=True)
                result[..., missing] = (values[..., left[missing]] * (1-alpha[missing]) + values[..., right[missing]] * alpha[missing]).astype(np.float32)
                arrays.append(result)
            bx, by, ellip, wna, fcp = arrays
            build.last_interpolation.update(interpolated_seconds=int(missing.sum()),maximum_missing_gap_seconds=int(gap.max()),
                duplicate_second_bins_collapsed=bool(np.any(np.diff(second_bins[int(right[0]):int(right[-1])+1])==0)))
        if not all(np.isfinite(array).all() for array in (bx, by, ellip, wna, fcp)):
            build.last_interpolation["unavailable_reason"] = "nonfinite_input_features"
            return None
        if np.any(fcp <= 0):
            build.last_interpolation["unavailable_reason"] = "invalid_fcp"
            return None
        profile = _build_joint_profile_inference(bx, by, ellip, wna, fcp, frequencies)
        return profile, np.asarray(fcp, dtype=np.float32)

    build.last_interpolation = {}
    return build


def dense_rescore_groups(
    model: torch.nn.Module,
    results: list[dict],
    *,
    threshold: float,
    build_window: Callable[[int, int], tuple[np.ndarray, np.ndarray] | None],
    freq_pol: np.ndarray,
    batch_size: int = 256,
    device=None,
    margin_seconds: int = 20,
    max_windows: int = 1_000_000,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], dict]:
    """Rescore a global queue across groups, retaining ALL scores and failures.

    This opt-in stage performs no physical gate, boundary selection, event
    merge, or calibrated-probability filtering. threshold only defines the
    original parents; downstream processing must use the raw-score field.
    Exceptions abort the call and restore model.training. A memory-safe batch
    is shared across parents, rather than a forward pass for each event.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    frequencies = np.asarray(freq_pol, dtype=np.float32)
    if frequencies.ndim != 1 or not frequencies.size or not np.isfinite(frequencies).all():
        raise ValueError("freq_pol must be a finite nonempty vector")
    groups, plans = plan_dense_group_windows(
        results, threshold=threshold, margin_seconds=margin_seconds, max_windows=max_windows
    )
    if device is None:
        first_parameter = next(model.parameters(), None)
        device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    frequency_tensor = torch.as_tensor(frequencies, device=device)
    output = [
        {
            "parent_id": p["parent_id"],
            "start": _format_time(p["start_s"]),
            "end": _format_time(p["end_s"]),
            "uncalibrated_score": None,
            "rescore_status": "unavailable_features",
        }
        for p in plans
    ]
    pending, forward_count = [], 0

    def flush():
        nonlocal forward_count
        if not pending:
            return
        profiles = torch.as_tensor(np.stack([item[1] for item in pending]), device=device)
        fcps = torch.as_tensor(np.stack([item[2] for item in pending]), device=device)
        with torch.inference_mode():
            logits = model(profiles, fcp_curve=fcps, freq_pol=frequency_tensor)
        logits = logits.detach().cpu().numpy().reshape(-1)
        if logits.size != len(pending):
            raise ValueError("model must return one binary-classification logit per window")
        calibrated_values = apply_logit_calibration(
            logits, getattr(model, "probability_calibration", None)
        )
        for (index, _, _), logit, calibrated in zip(pending, logits, calibrated_values):
            if np.isfinite(logit):
                output[index]["uncalibrated_score"] = float(
                    1.0 / (1.0 + np.exp(-np.clip(float(logit), -60.0, 60.0)))
                )
                output[index]["rescore_status"] = "scored"
                output[index].update(
                    raw_logit=float(logit),
                    score=float(calibrated),
                    raw_score=float(calibrated),
                    calibrated_probability=float(calibrated),
                )
            else:
                output[index]["rescore_status"] = "nonfinite_model_output"
        pending.clear()
        forward_count += 1

    was_training = model.training
    try:
        model.eval()
        for index, plan in enumerate(plans):
            if on_progress is not None:
                on_progress(index, len(plans))
            item = build_window(plan["start_s"], plan["end_s"])
            interpolation = getattr(build_window, "last_interpolation", None)
            if interpolation:
                output[index]["input_interpolation"] = dict(interpolation)
            if item is None:
                continue
            profile, fcp = (np.asarray(array, dtype=np.float32) for array in item)
            if profile.shape != (4, frequencies.size, 80) or fcp.shape != (80,):
                raise ValueError("window builder must return [4,F,80] profile and [80] fcp")
            if not np.isfinite(profile).all() or not np.isfinite(fcp).all() or np.any(fcp <= 0):
                continue
            pending.append((index, profile, fcp))
            if len(pending) == batch_size:
                flush()
        flush()
        if on_progress is not None:
            on_progress(len(plans), len(plans))
    finally:
        model.train(was_training)
    scored = sum(row["rescore_status"] == "scored" for row in output)
    by_start = {(plan["parent_id"], plan["start_s"]): row for plan, row in zip(plans, output)}
    for group in groups:
        for side, start in (
            ("left", group["scan_start_s"]),
            ("right", group["scan_end_s"] - 80),
        ):
            edge = by_start[(group["parent_id"], start)]
            score = edge["uncalibrated_score"]
            # This flags a censored SEARCH, not a physical onset/end estimate.
            group[f"{side}_edge_status"] = (
                "unavailable"
                if score is None
                else "high_score_at_scan_edge"
                if score >= threshold
                else "below_threshold"
            )
    return output, {
        "method": "dense_cnn_group_rescore_v1",
        "window_seconds": 80,
        "step_seconds": 1,
        "margin_seconds": margin_seconds,
        "score_definition": "sigmoid(raw_logit)",
        "original_threshold": threshold,
        "groups": groups,
        "planned_count": len(plans),
        "scored_count": scored,
        "unavailable_count": len(plans) - scored,
        "batch_size": batch_size,
        "forward_count": forward_count,
        "event_boundaries_inferred": False,
    }
