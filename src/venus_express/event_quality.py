"""Shared event-level data coverage checks."""

from __future__ import annotations

import numpy as np


EVENT_DATA_MISSING_RATIO_MAX = 0.10


def event_data_coverage(
    times_ns: np.ndarray,
    values: tuple[np.ndarray, ...],
    *,
    start_s: int,
    end_s: int,
    fs: float,
) -> dict[str, float | int]:
    """Return finite-sample coverage for a half-open event time interval."""
    expected = max(1, int(round((end_s - start_s) * float(fs))))
    timestamps = np.asarray(times_ns, dtype=np.int64)
    start_ns = int(start_s) * 1_000_000_000
    end_ns = int(end_s) * 1_000_000_000
    in_event = (timestamps >= start_ns) & (timestamps < end_ns)
    finite = in_event.copy()
    for value in values:
        array = np.asarray(value)
        if array.shape[0] != timestamps.shape[0]:
            raise ValueError("event coverage arrays must have equal lengths")
        finite &= np.isfinite(array)
    valid = min(expected, int(np.unique(timestamps[finite]).size))
    missing_ratio = max(0.0, (expected - valid) / expected)
    return {
        "expected_sample_count": expected,
        "valid_sample_count": valid,
        "data_missing_ratio": missing_ratio,
    }


def event_data_coverage_passes(coverage: dict[str, float | int]) -> bool:
    return bool(
        float(coverage["data_missing_ratio"]) <= EVENT_DATA_MISSING_RATIO_MAX
    )
