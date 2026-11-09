"""Two-stage event-boundary rescue for production ICW searches."""

from __future__ import annotations
from venus_express.physics.gyro import get_gyro_frequency

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from venus_express.analysis import DataAnalysis
from venus_express.event_quality import (
    EVENT_DATA_MISSING_RATIO_MAX,
    event_data_coverage,
    event_data_coverage_passes,
)
from venus_express.fast_polarization import (
    batched_polarization_analysis,
    batched_propagation_analysis,
)
from venus_express.icw_inference import (
    EVENT_MERGE_GAP_SECONDS_EXCLUSIVE,
    EVENT_ENERGY_GATE_THRESHOLD,
    PF_GATE_THRESHOLD,
    _perp_parallel_energy_ratio,
)
from venus_express.paths import VEX_DATA_DIR
from venus_express.signal.fft_metrics import (
    compute_perp_fft_metrics,
    compute_pf_band_peak_ratio,
)


def _seconds(value: str) -> int:
    return int(pd.to_datetime(value, utc=True).timestamp())


def _format_time(value: int) -> str:
    return pd.to_datetime(value, unit="s", utc=True).strftime("%Y-%m-%d/%H:%M:%S")


def _mean_at(values, frequencies, centers, target, start, end) -> float:
    values = np.asarray(values)
    frequencies = np.asarray(frequencies, dtype=float)
    if values.size == 0 or frequencies.size == 0 or not np.isfinite(target):
        return float("nan")
    column = values[:, int(np.argmin(np.abs(frequencies - target)))].astype(float)
    centers = pd.to_datetime(np.asarray(centers), utc=True)
    mask = np.asarray((centers >= start) & (centers <= end))
    if mask.size == column.size and mask.any():
        column = column[mask]
    column = column[np.isfinite(column)]
    return float(np.mean(column)) if column.size else float("nan")


def _fft_metrics(df, times_ns, start_s: int, end_s: int, fs: float):
    left = int(np.searchsorted(times_ns, start_s * 1_000_000_000, side="left"))
    right = int(np.searchsorted(times_ns, end_s * 1_000_000_000, side="right"))
    event = df.iloc[left:right]
    if len(event) < 16:
        return None
    bx = event["B_fac_x"].to_numpy(dtype=float)
    by = event["B_fac_y"].to_numpy(dtype=float)
    bz = event["B_fac_z"].to_numpy(dtype=float)
    bt = event["Bt"].to_numpy(dtype=float)
    coverage = event_data_coverage(
        times_ns[left:right],
        (bx, by, bz, bt),
        start_s=start_s,
        end_s=end_s,
        fs=fs,
    )
    if not event_data_coverage_passes(coverage):
        return None
    fft = compute_perp_fft_metrics(bx, by, bz, bt, fs=fs, pad_factor=2, smooth_points=4)
    fcp = float(fft["f_cp_ref"])
    fmax = float(fft["peak_freq"])
    return {
        "fmax": fmax,
        "fcp": fcp,
        "fmax_over_fcp": fmax / fcp,
        "pf_band_ratio": compute_pf_band_peak_ratio(
            bx, by, np.full(min(bx.size, by.size), fcp), fs=fs
        ),
        "energy_ratio": _perp_parallel_energy_ratio(bx, by, bz, get_gyro_frequency(bt * 1e-9)),
        **coverage,
    }


def _seed_events(results: list[dict], threshold: float) -> list[dict]:
    rows = []
    for index, result in enumerate(results):
        score = result.get("uncalibrated_score")
        if score is None or not np.isfinite(score) or float(score) < threshold:
            continue
        rows.append((index, _seconds(result["start"]), _seconds(result["end"])))
    rows.sort(key=lambda item: item[1])
    if not rows:
        return []

    groups = [[rows[0]]]
    current_end = rows[0][2]
    for row in rows[1:]:
        if row[1] - current_end < EVENT_MERGE_GAP_SECONDS_EXCLUSIVE:
            groups[-1].append(row)
            current_end = max(current_end, row[2])
        else:
            groups.append([row])
            current_end = row[2]

    seeds = []
    for group_id, group in enumerate(groups):
        local_peaks = []
        for position, row in enumerate(group):
            value = float(results[row[0]]["uncalibrated_score"])
            left = (
                float(results[group[position - 1][0]]["uncalibrated_score"])
                if position
                else -np.inf
            )
            right = (
                float(results[group[position + 1][0]]["uncalibrated_score"])
                if position + 1 < len(group)
                else -np.inf
            )
            if value >= left and value >= right:
                local_peaks.append(row)
        selected = []
        for row in sorted(
            local_peaks,
            key=lambda item: float(results[item[0]]["uncalibrated_score"]),
            reverse=True,
        ):
            center = (row[1] + row[2]) // 2
            if all(abs(center - (prior[1] + prior[2]) // 2) >= 120 for prior in selected):
                selected.append(row)
        for index, start_s, end_s in selected:
            seeds.append(
                {
                    "group_id": group_id,
                    "result_index": index,
                    "base_start_s": start_s,
                    "base_end_s": end_s,
                }
            )
    return seeds


def _full_pass(variant: dict) -> bool:
    metrics = variant["metrics"]
    return bool(
        0.8 <= metrics["fmax_over_fcp"] <= 1.2
        and metrics.get("data_missing_ratio", 1.0) <= EVENT_DATA_MISSING_RATIO_MAX
        and variant["ellip_at_fmax"] <= -0.5
        and variant["wna_at_fmax"] < 30.0
        and metrics["pf_band_ratio"] >= PF_GATE_THRESHOLD
        and metrics["energy_ratio"] is not None
        and metrics["energy_ratio"] >= EVENT_ENERGY_GATE_THRESHOLD
    )


def _overlap_groups(candidates: list[dict]) -> list[list[dict]]:
    """Group candidates whose final time intervals strictly overlap."""
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (_seconds(item["start"]), _seconds(item["end"])),
    )
    groups: list[list[dict]] = [[ordered[0]]]
    current_end = _seconds(ordered[0]["end"])
    for candidate in ordered[1:]:
        start_s = _seconds(candidate["start"])
        end_s = _seconds(candidate["end"])
        if start_s < current_end:
            groups[-1].append(candidate)
            current_end = max(current_end, end_s)
        else:
            groups.append([candidate])
            current_end = end_s
    return groups


def _candidate_peak_score(candidate: dict) -> float:
    return float(candidate.get("uncalibrated_score", candidate.get("raw_score", 0.0)))


def _deduplicated_representative(members: list[dict]) -> dict:
    """Keep the strongest original event when a union fails event-level gates."""
    representative = dict(max(members, key=_candidate_peak_score))
    diagnostics = dict(representative.get("model_diagnostics") or {})
    diagnostics.update(
        {
            "event_overlap_deduplicated": True,
            "event_overlap_member_count": len(members),
            "event_overlap_union_gate_pass": False,
            "event_overlap_original_intervals": [
                {"start": item["start"], "end": item["end"]} for item in members
            ],
        }
    )
    representative["model_diagnostics"] = diagnostics
    return representative


def _recompute_merged_candidate(
    members: list[dict], source_df: pd.DataFrame, fs: float, source_times_ns: np.ndarray
) -> dict | None:
    """Recompute an overlapping candidate union without inheriting old gates."""
    start_s = min(_seconds(item["start"]) for item in members)
    end_s = max(_seconds(item["end"]) for item in members)
    margin = 240
    left = int(np.searchsorted(source_times_ns, (start_s - margin) * 1_000_000_000, side="left"))
    right = int(np.searchsorted(source_times_ns, (end_s + margin) * 1_000_000_000, side="right"))
    df = source_df.iloc[left:right].reset_index(drop=True)
    times_ns = source_times_ns[left:right]
    metrics = _fft_metrics(df, times_ns, start_s, end_s, fs)
    if metrics is None:
        return None
    bx = df["B_fac_x"].to_numpy(dtype=float, copy=False)
    by = df["B_fac_y"].to_numpy(dtype=float, copy=False)
    bz = df["B_fac_z"].to_numpy(dtype=float, copy=False)
    times = pd.to_datetime(df["time_ut"], utc=True).to_numpy()
    pol = batched_polarization_analysis(bx, by, window_size=160, step_size=10, fs=fs)
    pol["window_center_times"] = times[pol["window_center_indices"]]
    wna = batched_propagation_analysis(bx, by, bz, window_size=160, step_size=10, fs=fs)
    wna["window_center_times"] = times[wna["window_center_indices"]]
    start = pd.to_datetime(start_s, unit="s", utc=True)
    end = pd.to_datetime(end_s, unit="s", utc=True)
    selected = {
        "metrics": metrics,
        "ellip_at_fmax": _mean_at(
            pol["ellip_signed"],
            pol["freq_pol"],
            pol["window_center_times"],
            metrics["fmax"],
            start,
            end,
        ),
        "wna_at_fmax": _mean_at(
            wna["wna_deg"],
            wna["freq_wna"],
            wna["window_center_times"],
            metrics["fmax"],
            start,
            end,
        ),
    }
    if not _full_pass(selected):
        return None
    representative = max(members, key=_candidate_peak_score)
    peak = dict(representative)
    diagnostics = dict(peak.get("model_diagnostics") or {})
    diagnostics.update(
        {
            "event_merge_applied": True,
            "event_merge_member_count": len(members),
            "event_merge_original_intervals": [
                {"start": item["start"], "end": item["end"]} for item in members
            ],
            "event_rescue_mode": "overlap_union",
            "event_fmax_over_fcp": metrics["fmax_over_fcp"],
            "event_ellip_at_fmax": selected["ellip_at_fmax"],
            "event_wna_at_fmax": selected["wna_at_fmax"],
            "event_pf_band_ratio": metrics["pf_band_ratio"],
            "event_perp_parallel_energy_ratio": metrics["energy_ratio"],
            "event_data_missing_ratio": metrics["data_missing_ratio"],
            "event_expected_sample_count": metrics["expected_sample_count"],
            "event_valid_sample_count": metrics["valid_sample_count"],
            "event_data_coverage_gate_pass": True,
            "event_energy_gate_pass": True,
            "pf_gate_pass": True,
            "pf_gate_status": "pass",
        }
    )
    peak.update(
        {
            "start": _format_time(start_s),
            "end": _format_time(end_s),
            "score": float(peak.get("calibrated_probability", peak.get("raw_score", 0.0))),
            "model_diagnostics": diagnostics,
        }
    )
    return peak


def _merge_overlapping_candidates(
    candidates: list[dict], *, data_dir: str
) -> tuple[list[dict], dict]:
    """Union overlapping events only when the recomputed union passes all gates."""
    if not candidates:
        return [], {
            "overlap_clusters": 0,
            "merged_clusters": 0,
            "representative_clusters": 0,
        }
    groups = _overlap_groups(candidates)
    overlapping = [group for group in groups if len(group) > 1]
    if not overlapping:
        return candidates, {
            "overlap_clusters": 0,
            "merged_clusters": 0,
            "representative_clusters": 0,
        }
    by_day: defaultdict[str, list[tuple[int, list[dict]]]] = defaultdict(list)
    scalar: list[tuple[int, list[dict]]] = []
    for index, group in enumerate(overlapping):
        start_s = min(_seconds(item["start"]) for item in group)
        end_s = max(_seconds(item["end"]) for item in group)
        start = pd.to_datetime(start_s - 240, unit="s", utc=True)
        end = pd.to_datetime(end_s + 240, unit="s", utc=True)
        target = by_day[start.strftime("%Y-%m-%d")] if start.date() == end.date() else scalar
        target.append((index, group))
    merged: list[dict | None] = [None] * len(overlapping)
    data_loads = 0
    for items in by_day.values():
        load_start = min(min(_seconds(item["start"]) for item in group) - 240 for _, group in items)
        load_end = max(max(_seconds(item["end"]) for item in group) + 240 for _, group in items)
        try:
            analysis = DataAnalysis(
                [_format_time(load_start), _format_time(load_end)], data_dir=data_dir
            )
            data_loads += 1
            if analysis.dataframe is None:
                raise ValueError("insufficient day data")
            times_ns = (
                pd.to_datetime(analysis.dataframe["time_ut"], utc=True)
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            for index, group in items:
                try:
                    merged[index] = _recompute_merged_candidate(
                        group, analysis.dataframe, float(analysis.fs), times_ns
                    )
                except Exception:
                    merged[index] = None
        except Exception:
            continue
    for index, group in scalar:
        start_s = min(_seconds(item["start"]) for item in group)
        end_s = max(_seconds(item["end"]) for item in group)
        try:
            analysis = DataAnalysis(
                [_format_time(start_s - 240), _format_time(end_s + 240)], data_dir=data_dir
            )
            data_loads += 1
            if analysis.dataframe is None:
                raise ValueError("insufficient interval data")
            times_ns = (
                pd.to_datetime(analysis.dataframe["time_ut"], utc=True)
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            merged[index] = _recompute_merged_candidate(
                group, analysis.dataframe, float(analysis.fs), times_ns
            )
        except Exception:
            merged[index] = None

    output: list[dict] = []
    merged_count = representative_count = 0
    group_index = 0
    for group in groups:
        if len(group) == 1:
            output.extend(group)
            continue
        replacement = merged[group_index]
        group_index += 1
        if replacement is None:
            output.append(_deduplicated_representative(group))
            representative_count += 1
        else:
            output.append(replacement)
            merged_count += 1
    return output, {
        "overlap_clusters": len(overlapping),
        "merged_clusters": merged_count,
        "representative_clusters": representative_count,
        "data_loads": data_loads,
    }


def _refine_seed(seed, results, source_df, fs, source_times_ns):
    margin = 240
    load_left = int(
        np.searchsorted(
            source_times_ns, (seed["base_start_s"] - margin) * 1_000_000_000, side="left"
        )
    )
    load_right = int(
        np.searchsorted(
            source_times_ns, (seed["base_end_s"] + margin) * 1_000_000_000, side="right"
        )
    )
    df = source_df.iloc[load_left:load_right].reset_index(drop=True)
    times_ns = source_times_ns[load_left:load_right]
    times = pd.to_datetime(df["time_ut"], utc=True).to_numpy()
    variants = []
    seen = set()

    def add(start_s, end_s, mode, displacement):
        if (start_s, end_s) in seen or end_s - start_s < 79:
            return
        seen.add((start_s, end_s))
        metrics = _fft_metrics(df, times_ns, start_s, end_s, fs)
        if metrics is not None:
            variants.append(
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "mode": mode,
                    "displacement": displacement,
                    "metrics": metrics,
                }
            )

    for left in (-40, -20, 0, 20, 40):
        for right in (-40, -20, 0, 20, 40):
            add(
                seed["base_start_s"] + left,
                seed["base_end_s"] + right,
                "narrow",
                abs(left) + abs(right),
            )

    bx = df["B_fac_x"].to_numpy(dtype=float, copy=False)
    by = df["B_fac_y"].to_numpy(dtype=float, copy=False)
    bz = df["B_fac_z"].to_numpy(dtype=float, copy=False)
    pol = batched_polarization_analysis(bx, by, window_size=160, step_size=10, fs=fs)
    pol["window_center_times"] = times[pol["window_center_indices"]]
    wna = batched_propagation_analysis(bx, by, bz, window_size=160, step_size=10, fs=fs)
    wna["window_center_times"] = times[wna["window_center_indices"]]

    def evaluate(items):
        for variant in items:
            start = pd.to_datetime(variant["start_s"], unit="s", utc=True)
            end = pd.to_datetime(variant["end_s"], unit="s", utc=True)
            target = variant["metrics"]["fmax"]
            variant["ellip_at_fmax"] = _mean_at(
                pol["ellip_signed"],
                pol["freq_pol"],
                pol["window_center_times"],
                target,
                start,
                end,
            )
            variant["wna_at_fmax"] = _mean_at(
                wna["wna_deg"],
                wna["freq_wna"],
                wna["window_center_times"],
                target,
                start,
                end,
            )

    evaluate(variants)
    wide_used = not any(_full_pass(item) for item in variants)
    if wide_used:
        narrow_count = len(variants)
        center = (seed["base_start_s"] + seed["base_end_s"]) // 2
        for shift in range(-120, 121, 20):
            for duration in (80, 100, 120, 160, 200):
                start_s = center + shift - duration // 2
                add(
                    start_s,
                    start_s + duration,
                    "wide",
                    abs(shift) + abs(duration - 80) // 2,
                )
        evaluate(variants[narrow_count:])

    def objective(item):
        metrics = item["metrics"]
        in_band = 0.8 <= metrics["fmax_over_fcp"] <= 1.2
        downstream = (
            item["ellip_at_fmax"] <= -0.5,
            item["wna_at_fmax"] < 30.0,
            metrics["pf_band_ratio"] >= PF_GATE_THRESHOLD,
            metrics["energy_ratio"] is not None and metrics["energy_ratio"] >= EVENT_ENERGY_GATE_THRESHOLD,
        )
        return (
            int(in_band),
            int(in_band and all(downstream)),
            sum(downstream),
            np.log1p(metrics["pf_band_ratio"]),
            -item["displacement"],
        )

    selected = max(variants, key=objective)
    selected["wide_search_used"] = wide_used
    selected["variant_count"] = len(variants)
    return selected


def rescue_event_candidates(
    results: list[dict], *, threshold: float, data_dir: str = str(VEX_DATA_DIR)
) -> tuple[list[dict], dict]:
    """Return strict event-level candidates from raw CNN window scores."""
    started = time.perf_counter()
    seeds = _seed_events(results, float(threshold))
    by_day = defaultdict(list)
    scalar = []
    for index, seed in enumerate(seeds):
        start = pd.to_datetime(seed["base_start_s"] - 240, unit="s", utc=True)
        end = pd.to_datetime(seed["base_end_s"] + 240, unit="s", utc=True)
        (by_day[start.strftime("%Y-%m-%d")] if start.date() == end.date() else scalar).append(
            (index, seed)
        )
    refined = [None] * len(seeds)
    data_loads = 0
    for items in by_day.values():
        load_start = min(seed["base_start_s"] - 240 for _, seed in items)
        load_end = max(seed["base_end_s"] + 240 for _, seed in items)
        try:
            analysis = DataAnalysis(
                [_format_time(load_start), _format_time(load_end)], data_dir=data_dir
            )
            data_loads += 1
            if analysis.dataframe is None:
                continue
            source_times_ns = (
                pd.to_datetime(analysis.dataframe["time_ut"], utc=True)
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            for index, seed in items:
                try:
                    refined[index] = _refine_seed(
                        seed,
                        results,
                        analysis.dataframe,
                        float(analysis.fs),
                        source_times_ns,
                    )
                except Exception:
                    refined[index] = None
        except Exception:
            continue
    for index, seed in scalar:
        try:
            analysis = DataAnalysis(
                [_format_time(seed["base_start_s"] - 240), _format_time(seed["base_end_s"] + 240)],
                data_dir=data_dir,
            )
            data_loads += 1
            if analysis.dataframe is None:
                continue
            times_ns = (
                pd.to_datetime(analysis.dataframe["time_ut"], utc=True)
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            )
            refined[index] = _refine_seed(
                seed, results, analysis.dataframe, float(analysis.fs), times_ns
            )
        except Exception:
            refined[index] = None

    candidates = []
    for seed, selected in zip(seeds, refined):
        if selected is None or not _full_pass(selected):
            continue
        peak = dict(results[seed["result_index"]])
        metrics = selected["metrics"]
        diagnostics = dict(peak.get("model_diagnostics") or {})
        diagnostics.update(
            {
                "event_rescue_applied": True,
                "event_rescue_mode": selected["mode"],
                "event_rescue_wide_search_used": selected["wide_search_used"],
                "event_rescue_variant_count": selected["variant_count"],
                "event_fmax_over_fcp": metrics["fmax_over_fcp"],
                "event_ellip_at_fmax": selected["ellip_at_fmax"],
                "event_wna_at_fmax": selected["wna_at_fmax"],
                "event_pf_band_ratio": metrics["pf_band_ratio"],
                "event_perp_parallel_energy_ratio": metrics["energy_ratio"],
                "event_data_missing_ratio": metrics["data_missing_ratio"],
                "event_expected_sample_count": metrics["expected_sample_count"],
                "event_valid_sample_count": metrics["valid_sample_count"],
                "event_data_coverage_gate_pass": True,
                "event_energy_gate_pass": True,
                "pf_gate_pass": True,
                "pf_gate_status": "pass",
            }
        )
        peak.update(
            {
                "start": _format_time(selected["start_s"]),
                "end": _format_time(selected["end_s"]),
                "score": float(peak.get("calibrated_probability", peak.get("raw_score", 0.0))),
                "model_diagnostics": diagnostics,
            }
        )
        candidates.append(peak)
    candidates, merge_meta = _merge_overlapping_candidates(candidates, data_dir=data_dir)
    return candidates, {
        "seed_count": len(seeds),
        "passed_count": len(candidates),
        "wide_search_count": sum(bool(item and item["wide_search_used"]) for item in refined),
        "data_loads": data_loads,
        "failed_seed_count": sum(item is None for item in refined),
        "overlap_merge": merge_meta,
        "elapsed_seconds": time.perf_counter() - started,
    }
