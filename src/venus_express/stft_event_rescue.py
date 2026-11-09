"""STFT support-segment rescue promoted from the frozen full-mission search.

No experiment modules, truth labels, or catalogue boundaries enter production.
STFT proposes intervals; every output requires an independent whole six-gate check.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import ShortTimeFFT

from venus_express.event_rescue import _format_time, _full_pass
from venus_express.physics.gyro import get_gyro_frequency

METHOD = "stft_support_segments_bounded_uncapped_v1"
ENDPOINT_TARGET_STEP_S = 5
MAX_GRID_POINTS_PER_RUN = 256


def support_profile(evaluate, left, right, check_cancel=None):
    """Same three-scale proposal evidence as the prior STFT experiment."""
    check_cancel = check_cancel or (lambda: None)
    df, fs, t = evaluate.df, evaluate.fs, evaluate.t
    bx, by, bt = (df[k].to_numpy(float) for k in ["B_fac_x", "B_fac_y", "Bt"])
    scores, votes = [], []
    for seconds in [80, 120, 160]:
        check_cancel()
        n = int(round(seconds * fs))
        n += n % 2
        score, vote = np.full(len(df), np.nan), np.full(len(df), np.nan)
        if len(df) >= n:
            centers = np.arange(n // 2, len(df) - n // 2 + 1)
            stft = ShortTimeFFT(
                np.hanning(n), hop=1, fs=fs, mfft=2 * n, fft_mode="onesided2X", scale_to="psd"
            )
            power = stft.spectrogram(
                bx, detr="constant", p0=int(centers[0]), p1=int(centers[-1] + 1)
            )
            power += stft.spectrogram(
                by, detr="constant", p0=int(centers[0]), p1=int(centers[-1] + 1)
            )
            windows = np.lib.stride_tricks.sliding_window_view(bt, n)
            fcp = get_gyro_frequency(np.nanmedian(windows, axis=1) * 1e-9)
            f = stft.f[:, None]
            band = (f >= 0.8 * fcp) & (f <= 1.2 * fcp)
            positive = f > 0
            inp = np.max(np.where(band, power, 0), axis=0)
            glob = np.max(np.where(positive, power, 0), axis=0)
            background = np.nanpercentile(np.where(positive & ~band, power, np.nan), 75, axis=0)
            contrast = inp / np.maximum(background, 1e-30)
            dominance = inp / np.maximum(glob, 1e-30)
            spans = t[centers + n // 2 - 1] - t[centers - n // 2]
            valid = (
                np.isfinite(windows).all(axis=1)
                & (spans <= (n - 1) / fs * 1.1)
                & band.any(axis=0)
                & np.isfinite(contrast)
            )
            score[centers] = np.where(valid, np.log1p(contrast) * dominance, np.nan)
            vote[centers] = np.where(
                valid, ((contrast >= 3) & (dominance >= 0.5)).astype(float), np.nan
            )
        scores.append(score)
        votes.append(vote)
    scores, votes = np.array(scores), np.array(votes)
    anyvalid = np.isfinite(scores).any(axis=0)
    combined = np.zeros(len(df))
    combined[anyvalid] = np.nanmedian(scores[:, anyvalid], axis=0)
    valid = np.isfinite(votes).sum(axis=0) >= 2
    support = (np.nansum(votes, axis=0) >= 2) & valid & (t >= left) & (t < right)
    return {"times": t, "score": combined, "valid": valid, "support": support, "fs": fs}


def continuous_runs(profile):
    """Never bridge a non-supporting frame or a raw timestamp gap."""
    t, support = profile["times"], profile["support"]
    indices = np.flatnonzero(support)
    if not len(indices):
        return []
    breaks = (
        np.flatnonzero((np.diff(indices) > 1) | (np.diff(t[indices]) > 1.5 / profile["fs"])) + 1
    )
    result = []
    for i, part in enumerate(np.split(indices, breaks)):
        result.append(
            {
                "run_id": i,
                "start_s": int(np.floor(t[part[0]])),
                "end_s": int(np.floor(t[part[-1]])) + 1,
                "centers": len(part),
            }
        )
    return result


def segment_proposals(profile, left, right, budget, check_cancel=None):
    """Rank arbitrary endpoint pairs within each support run's +/-40-s envelope.

    Five-second target endpoint spacing is coarsened only to bound grid memory
    for exceptionally long runs. Exact run and parent-envelope edges are kept.
    Duration has no upper limit other than the CNN parent. The whole parent is
    reserved as the first candidate; remaining slots are distributed round-robin
    over support runs so one strong component cannot consume the whole budget.
    """
    check_cancel = check_cancel or (lambda: None)
    runs = continuous_runs(profile)
    t, valid, support, score = (profile[k] for k in ["times", "valid", "support", "score"])
    count = np.r_[0, np.cumsum(valid)]
    sup = np.r_[0, np.cumsum(support)]
    values = np.r_[0, np.cumsum(np.where(valid, score, 0))]
    center = (left + right) / 2
    ranked = []
    total = 0
    max_duration = 0
    for run in runs:
        check_cancel()
        lo = max(left, run["start_s"] - 40)
        hi = min(right, run["end_s"] + 40)
        if hi - lo < 80:
            ranked.append([])
            continue
        step = max(
            ENDPOINT_TARGET_STEP_S, int(np.ceil((hi - lo) / (MAX_GRID_POINTS_PER_RUN - 1) / 5)) * 5
        )
        points = set(range(lo, hi + 1, step)) | {lo, hi}
        for edge in [run["start_s"], run["end_s"]]:
            points.update(edge + d for d in [-20, -10, -5, 0, 5, 10, 20] if lo <= edge + d <= hi)
        points = np.array(sorted(points), dtype=np.int64)
        ii, jj = np.triu_indices(len(points), k=1)
        starts, ends = points[ii], points[jj]
        mask = ends - starts >= 80
        starts, ends = starts[mask], ends[mask]
        if not len(starts):
            ranked.append([])
            continue
        total += len(starts)
        max_duration = max(max_duration, int((ends - starts).max()))
        a, b = np.searchsorted(t, starts, side="left"), np.searchsorted(t, ends, side="left")
        nvalid = count[b] - count[a]
        nall = b - a
        merit = (sup[b] - sup[a]) * (values[b] - values[a]) / np.maximum(nvalid * nvalid, 1)
        merit *= nvalid / np.maximum(nall, 1)
        displacement = (
            np.abs((starts + ends) / 2 - center) + np.abs((ends - starts) - (right - left)) / 2
        )
        order = np.lexsort((ends, starts, displacement, -merit))[:budget]
        ranked.append(
            [
                {
                    "start_s": int(starts[k]),
                    "end_s": int(ends[k]),
                    "run_id": run["run_id"],
                    "stft_merit": float(merit[k]),
                    "displacement": float(displacement[k]),
                }
                for k in order
            ]
        )
    proposals = [
        {
            "start_s": left,
            "end_s": right,
            "run_id": "whole_parent",
            "displacement": 0.0,
            "stft_merit": None,
        }
    ]
    seen = {(left, right)}
    cursors = [0] * len(ranked)
    while len(proposals) < budget:
        added = False
        for j, items in enumerate(ranked):
            while cursors[j] < len(items):
                q = items[cursors[j]]
                cursors[j] += 1
                key = q["start_s"], q["end_s"]
                if key not in seen:
                    proposals.append(q)
                    seen.add(key)
                    added = True
                    break
            if len(proposals) >= budget:
                break
        if not added:
            break
    assert all(
        left <= q["start_s"] and q["end_s"] <= right and q["end_s"] - q["start_s"] >= 80
        for q in proposals
    )
    return proposals, {
        "support_runs": runs,
        "pre_shortlist_pair_count": total,
        "maximum_generated_duration_s": max(right - left, max_duration),
        "budget": budget,
        "proposals_checked": len(proposals),
        "endpoint_target_step_s": 5,
        "max_grid_points_per_run": MAX_GRID_POINTS_PER_RUN,
    }


def _objective(item):
    # All six gates have already passed: this is the frozen objective's
    # remaining variable ordering, including its deterministic first tie.
    return (np.log1p(item["measurement"]["metrics"]["pf_band_ratio"]), -item["displacement"])


def rescue_stft_scope(source, evaluate, left, right, *, check_cancel=None, on_progress=None):
    """Rescue an already-failed scope; never repeat its whole check."""
    check_cancel = check_cancel or (lambda: None)
    if right - left < 80:
        raise ValueError("STFT scope must span at least 80 seconds")
    budget = max(90, 90 * math.ceil((right - left) / 120))
    check_cancel()
    profile = support_profile(source, left, right, check_cancel)
    proposals, audit = segment_proposals(profile, left, right, budget, check_cancel)
    proposals = [q for q in proposals if q["run_id"] != "whole_parent"]
    passing = {}
    checked = []
    for i, q in enumerate(proposals):
        check_cancel()
        if on_progress is not None:
            on_progress(i, len(proposals))
        a, b = q["start_s"], q["end_s"]
        if not left <= a < b <= right or b - a < 80:
            raise ValueError("STFT proposal escaped its scope")
        m = evaluate(a, b)
        passed = m is not None and _full_pass(m)
        checked.append({**q, "gate_pass": bool(passed), "measurement": m})
        if passed:
            passing.setdefault(q["run_id"], []).append({**q, "measurement": m})
    events = []
    for run_id, items in passing.items():
        q = max(items, key=_objective)
        events.append(
            {
                "start": _format_time(q["start_s"]),
                "end": _format_time(q["end_s"]),
                "whole_interval_gate_pass": True,
                "whole_interval_measurement": q["measurement"],
                "model_diagnostics": {
                    "event_rescue_applied": True,
                    "event_stft_rescue_method": METHOD,
                    "event_stft_support_run_id": int(run_id),
                    "event_stft_scope_start": _format_time(left),
                    "event_stft_scope_end": _format_time(right),
                    "event_stft_merit": q["stft_merit"],
                    "event_stft_budget": budget,
                },
            }
        )
    check_cancel()
    if on_progress is not None:
        on_progress(len(proposals), len(proposals))
    return events, {
        **audit,
        "method": METHOD,
        "checked": len(checked),
        "passing_runs": len(passing),
        "checked_intervals": checked,
        "whole_parent_rechecked": False,
    }
