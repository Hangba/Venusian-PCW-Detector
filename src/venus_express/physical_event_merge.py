"""Opt-in local-physics event assembly; does not replace production rescue.

Events are connected unions of fully qualified local windows, NOT assertions
that the entire event passes the existing whole-interval FFT gate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from venus_express.event_rescue import _format_time, _full_pass, _seconds


@dataclass(frozen=True)
class LocalMergeConfig:
    window_seconds: int = 80
    step_seconds: int = 20
    parent_gap_exclusive: int = 60
    max_local_windows: int = 100_000

    def __post_init__(self):
        values = (
            self.window_seconds,
            self.step_seconds,
            self.parent_gap_exclusive,
            self.max_local_windows,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("configuration values must be positive integers")
        if self.step_seconds > self.window_seconds:
            raise ValueError("step must not exceed local window duration")


def assemble_physical_events(
    results: list[dict],
    *,
    threshold: float,
    evaluate: Callable[[int, int], dict | None],
    config: LocalMergeConfig = LocalMergeConfig(),
) -> tuple[list[dict], dict]:
    """Scan complete CNN candidate groups and retain connected local support.

    ``evaluate(start, end)`` returns the same metric structure as ``_full_pass``
    accepts, or None for an unavailable measurement. It must use magnetic data,
    not CNN scores, and is called for both local windows and full event spans.
    All input windows must have config.window_seconds duration. Unknown/failed
    windows are audited and never themselves added to support. Overlapping
    qualified windows can still support their time range. No unsupported gap is
    bridged. Exceptions and budget exhaustion propagate; no partial result is
    returned as a successful completed scan.
    """
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    rows = []
    for index, row in enumerate(results):
        score = row.get("uncalibrated_score")
        if score is None or not math.isfinite(float(score)) or float(score) < threshold:
            continue
        start, end = _seconds(row["start"]), _seconds(row["end"])
        if end - start != config.window_seconds:
            raise ValueError("accepted CNN window duration differs from local window duration")
        rows.append((start, end, float(score), index))
    rows.sort()
    groups = []
    for row in rows:
        if not groups or row[0] - groups[-1]["end"] >= config.parent_gap_exclusive:
            groups.append({"start": row[0], "end": row[1], "rows": [row]})
        else:
            groups[-1]["end"] = max(groups[-1]["end"], row[1])
            groups[-1]["rows"].append(row)

    plans = []
    planned_count = 0
    for group in groups:
        grid_count = (
            group["end"] - group["start"] - config.window_seconds
        ) // config.step_seconds + 1
        if planned_count + grid_count > config.max_local_windows:
            raise ValueError("local-window budget exceeded; split input explicitly")
        starts = set(
            range(group["start"], group["end"] - config.window_seconds + 1, config.step_seconds)
        )
        # Include the final edge and every accepted original window even if
        # their timestamps do not land on the local grid.
        starts.add(group["end"] - config.window_seconds)
        starts.update(row[0] for row in group["rows"])
        plans.append(sorted(starts))
        planned_count += len(starts)
    if sum(map(len, plans)) > config.max_local_windows:
        raise ValueError("local-window budget exceeded; split input explicitly")

    cache = {}

    def measure(start, end):
        key = (start, end)
        if key not in cache:
            cache[key] = evaluate(start, end)
        return cache[key]

    events, audit = [], []
    for parent_id, (group, starts) in enumerate(zip(groups, plans)):
        passed = []
        for start in starts:
            end = start + config.window_seconds
            variant = measure(start, end)
            status = (
                "unavailable" if variant is None else ("pass" if _full_pass(variant) else "fail")
            )
            audit.append(
                {
                    "parent_id": parent_id,
                    "start_s": start,
                    "end_s": end,
                    "status": status,
                    "measurement": variant,
                }
            )
            if status == "pass":
                passed.append((start, end))

        components = []
        for start, end in passed:
            if not components or start > components[-1]["end"]:
                components.append({"start": start, "end": end, "support": [(start, end)]})
            else:
                components[-1]["end"] = max(components[-1]["end"], end)
                components[-1]["support"].append((start, end))

        for component in components:
            start, end = component["start"], component["end"]
            anchors = [row for row in group["rows"] if row[0] < end and start < row[1]]
            # Do not promote a physically qualified background island with no
            # positive-duration overlap with an accepted CNN window.
            if not anchors:
                continue
            whole = measure(start, end)
            whole_pass = None if whole is None else bool(_full_pass(whole))
            events.append(
                {
                    "start": _format_time(start),
                    "end": _format_time(end),
                    "parent_id": parent_id,
                    "uncalibrated_score": max(row[2] for row in anchors),
                    "source_window_indices": [row[3] for row in anchors],
                    "definition": "local_six_gate_support_chain_v1",
                    "whole_interval_gate_pass": whole_pass,
                    "whole_interval_measurement": whole,
                    "support_windows": [
                        {"start_s": a, "end_s": b} for a, b in component["support"]
                    ],
                    "supported_duration_seconds": end - start,
                }
            )
    return events, {
        "definition": "local_six_gate_support_chain_v1",
        "config": vars(config),
        "parent_count": len(groups),
        "local_window_count": len(audit),
        "measurement_count": len(cache),
        "local_measurements": audit,
        "event_count": len(events),
        "whole_gate_is_diagnostic_only": True,
    }


def make_local_physics_evaluator(dataframe, fs: float):
    """Prepare the existing physical measurements once on a loaded data chunk.

    Requires DataAnalysis FAC columns, UTC-sorted timestamps, and adequate
    context for polarization windows. A lack of polarization centers inside a
    requested interval is explicitly unavailable, with no whole-chunk fallback.
    """
    import numpy as np
    import pandas as pd

    from venus_express.event_rescue import _fft_metrics
    from venus_express.fast_polarization import (
        batched_polarization_analysis,
        batched_propagation_analysis,
    )

    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    times = pd.to_datetime(dataframe["time_ut"], utc=True).to_numpy(dtype="datetime64[ns]")
    ns = times.astype(np.int64)
    if len(ns) < 160 or np.isnat(times).any() or np.any(np.diff(ns) <= 0):
        raise ValueError("need at least 160 samples with strictly increasing valid times")
    bx, by, bz = (dataframe[key].to_numpy(dtype=float) for key in ("B_fac_x", "B_fac_y", "B_fac_z"))
    pol = batched_polarization_analysis(bx, by, window_size=160, step_size=10, fs=fs)
    wna = batched_propagation_analysis(bx, by, bz, window_size=160, step_size=10, fs=fs)

    def mean_at(product, value_key, frequency_key, target, start, end):
        indices = np.asarray(product["window_center_indices"], dtype=int)
        centers = ns[indices]
        mask = (centers >= start * 1_000_000_000) & (centers < end * 1_000_000_000)
        # The shared fast routines replicate the first/last spectrum at chunk
        # edges. Those are display padding, not actual analysis-window centers.
        mask &= (indices >= 80) & (indices + 80 <= len(ns))
        frequencies = np.asarray(product[frequency_key])
        if not mask.any() or not frequencies.size:
            return None
        values = np.asarray(product[value_key])[mask, np.argmin(abs(frequencies - target))]
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else None

    def evaluate(start, end):
        metrics = _fft_metrics(dataframe, ns, start, end, fs)
        if metrics is None:
            return None
        ellip = mean_at(pol, "ellip_signed", "freq_pol", metrics["fmax"], start, end)
        angle = mean_at(wna, "wna_deg", "freq_wna", metrics["fmax"], start, end)
        if ellip is None or angle is None:
            return None
        return {"metrics": metrics, "ellip_at_fmax": ellip, "wna_at_fmax": angle}

    return evaluate


def assemble_scored_physical_events(
    parents: list[dict],
    dense_results: list[dict],
    *,
    threshold: float,
    evaluate: Callable[[int, int], dict | None],
) -> tuple[list[dict], dict]:
    """Keep whole-pass parents; use every dense high-score window on failure.

    Parents are the complete original groups from dense rescore's audit, with
    parent_id/start_s/end_s/scan_start_s/scan_end_s. Whole and local-chain
    acceptance are distinct definitions. No unmeasured gap or parent boundary
    is bridged; no accepted component is discarded in favor of a peak score.
    Final event duration must be at least 80 s. Apply this only
    after assembly: the local physics building blocks remain 80 s windows.
    This experimental policy does not select a physical onset at 1-s accuracy.
    """
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    lookup = {p["parent_id"]: p for p in parents}
    if len(lookup) != len(parents):
        raise ValueError("parent IDs must be unique")
    members = {key: [] for key in lookup}
    for row in dense_results:
        parent = lookup[row["parent_id"]]
        start, end = _seconds(row["start"]), _seconds(row["end"])
        if end - start != 80 or start < parent["scan_start_s"] or end > parent["scan_end_s"]:
            raise ValueError("dense window outside its declared parent's scan")
        members[row["parent_id"]].append((start, end, row))
    events, audit = [], []
    for parent in parents:
        key, start, end = parent["parent_id"], parent["start_s"], parent["end_s"]
        whole = evaluate(start, end)
        whole_pass = None if whole is None else bool(_full_pass(whole))
        record = {
            "parent_id": key,
            "whole_interval_gate_pass": whole_pass,
            "whole_interval_measurement": whole,
            "local_evaluations": [],
        }
        audit.append(record)
        if whole_pass:
            events.append(
                {
                    "parent_id": key,
                    "start": _format_time(start),
                    "end": _format_time(end),
                    "definition": "whole_six_gate_v1",
                    "whole_interval_gate_pass": True,
                    "whole_interval_measurement": whole,
                }
            )
            continue
        passed = []
        for a, b, row in sorted(members[key], key=lambda item: item[0]):
            score = row.get("uncalibrated_score")
            if score is None or not math.isfinite(float(score)) or float(score) < threshold:
                continue
            measurement = evaluate(a, b)
            status = (
                "unavailable"
                if measurement is None
                else ("pass" if _full_pass(measurement) else "fail")
            )
            record["local_evaluations"].append(
                {"start_s": a, "end_s": b, "status": status, "measurement": measurement}
            )
            if status == "pass":
                passed.append((a, b))
        components = []
        for a, b in passed:
            if not components or a > components[-1]["end_s"]:
                components.append({"start_s": a, "end_s": b, "support_windows": []})
            else:
                components[-1]["end_s"] = max(components[-1]["end_s"], b)
            components[-1]["support_windows"].append({"start_s": a, "end_s": b})
        for component in components:
            a, b = component["start_s"], component["end_s"]
            measurement = evaluate(a, b)
            events.append(
                {
                    "parent_id": key,
                    "start": _format_time(a),
                    "end": _format_time(b),
                    "definition": "local_six_gate_support_chain_v1",
                    "support_windows": component["support_windows"],
                    "whole_interval_gate_pass": None
                    if measurement is None
                    else bool(_full_pass(measurement)),
                    "whole_interval_measurement": measurement,
                }
            )
    kept, rejected_duration = [], []
    for event in events:
        duration = _seconds(event["end"]) - _seconds(event["start"])
        if duration >= 80:
            kept.append(event)
        else:
            rejected_duration.append(
                {
                    "parent_id": event["parent_id"],
                    "start": event["start"],
                    "end": event["end"],
                    "duration_seconds": duration,
                    "reason": "final_duration_must_be_at_least_80_seconds",
                }
            )
    return kept, {
        "method": "whole_parent_then_dense_local_v2",
        "parents": audit,
        "threshold": threshold,
        "minimum_duration_seconds_inclusive": 80,
        "rejected_duration": rejected_duration,
        "event_count": len(kept),
    }


def check_merged_events_then_rescue(
    candidates: list[dict],
    dense_results: list[dict],
    *,
    threshold: float,
    evaluate: Callable[[int, int], dict | None],
    rescue: Callable[[dict, list[dict]], tuple[list[dict], dict]] | None = None,
) -> tuple[list[dict], dict]:
    """Whole-candidate physics only; invoke event rescue only on non-pass.

    The default rescue is the existing production seed/boundary-search routine.
    It performs its own event physics checks, not unconditional local screening.
    CNN windows remain 80 s; only final output events require duration >=80 s.
    Scores are located at window centers. CNN input context may extend outside
    the candidate; require centers, rather than all 80 samples, to lie inside.
    """
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    lookup = {c["parent_id"]: c for c in candidates}
    if len(lookup) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    windows = {key: [] for key in lookup}
    for row in dense_results:
        parent = lookup[row["parent_id"]]
        start, end = _seconds(row["start"]), _seconds(row["end"])
        center = (start + end) / 2
        if end - start != 80 or center < parent["start_s"] or center > parent["end_s"]:
            raise ValueError("dense window center outside its candidate")
        windows[row["parent_id"]].append(row)
    output, records, rejected = [], [], []
    for candidate in candidates:
        key, start, end = candidate["parent_id"], candidate["start_s"], candidate["end_s"]
        whole = evaluate(start, end)
        passed = None if whole is None else bool(_full_pass(whole))
        record = {
            "parent_id": key,
            "whole_interval_gate_pass": passed,
            "whole_interval_measurement": whole,
            "rescue_called": False,
        }
        records.append(record)
        if passed:
            events = [
                {
                    "parent_id": key,
                    "start": _format_time(start),
                    "end": _format_time(end),
                    "definition": "whole_merged_six_gate_v1",
                    "whole_interval_gate_pass": True,
                    "whole_interval_measurement": whole,
                }
            ]
        else:
            # Only finite, genuinely scored windows can become rescue seeds.
            scored = [
                row
                for row in windows[key]
                if row.get("uncalibrated_score") is not None
                and math.isfinite(float(row["uncalibrated_score"]))
            ]
            record["rescue_called"] = True
            if rescue is None:
                from venus_express.event_rescue import rescue_event_candidates

                events, metadata = rescue_event_candidates(scored, threshold=threshold)
            else:
                events, metadata = rescue(candidate, scored)
            record["rescue_metadata"] = metadata
            events = [
                {**event, "parent_id": key, "definition": "event_rescue_after_whole_failure_v1"}
                for event in events
            ]
        for event in events:
            duration = _seconds(event["end"]) - _seconds(event["start"])
            if duration >= 80:
                output.append(event)
            else:
                rejected.append(
                    {
                        "parent_id": key,
                        "start": event["start"],
                        "end": event["end"],
                        "duration_seconds": duration,
                        "reason": "final_duration_must_be_at_least_80_seconds",
                    }
                )
    return output, {
        "method": "whole_merged_check_then_event_rescue_v1",
        "candidates": records,
        "threshold": threshold,
        "local_window_prescreening": False,
        "minimum_duration_seconds_inclusive": 80,
        "rejected_duration": rejected,
        "event_count": len(output),
    }
