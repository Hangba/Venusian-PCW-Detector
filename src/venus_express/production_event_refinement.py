"""Production PCW: continuous dense-score segments, exact physics, preserved overlaps."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from venus_express.analysis import DataAnalysis
from venus_express.dense_event_rescore import (
    dense_rescore_groups,
    make_precomputed_window_builder,
    plan_dense_group_windows,
)
from venus_express.event_rescue import (
    _format_time,
    _full_pass,
    _overlap_groups,
    _seconds,
)
from venus_express.physical_event_merge import (
    make_local_physics_evaluator,
)
from venus_express.stft_event_rescue import rescue_stft_scope

METHOD = "dense_contiguous_centers20_min80_stft_preserve_overlaps_v6"


def overlap_member_quality(event, reference_left, reference_right):
    """Rank already-qualified members by PF, then original-parent displacement.

    The cluster envelope is only a fallback for imported/legacy records that
    lack original-parent diagnostics. Final ties use earlier start/end times.
    No new interval or unmeasured boundary is manufactured by selection.
    """
    diagnostics = event.get("model_diagnostics") or {}
    measurement = event.get("whole_interval_measurement") or {}
    pf = measurement.get("metrics", {}).get("pf_band_ratio", diagnostics.get("event_pf_band_ratio"))
    pf = float(pf) if pf is not None and math.isfinite(float(pf)) else -math.inf
    a, b = _seconds(event["start"]), _seconds(event["end"])
    left = _seconds(diagnostics.get("event_original_group_start") or _format_time(reference_left))
    right = _seconds(diagnostics.get("event_original_group_end") or _format_time(reference_right))
    displacement = abs((a + b - left - right) / 2) + abs((b - a) - (right - left)) / 2
    return pf, float(displacement), (math.log1p(pf) if pf > -1 else -math.inf, -displacement, -a, -b)


class CachedWindowProvider:
    """One active feature slice; the shared daily cache owns its bounded LRU.

    Only midnight-crossing windows request both dates. Normal days include the
    last subsecond sample without forcing an unnecessary next-day cache build.
    """

    def __init__(self, cache, check_cancel=None):
        self.cache = cache
        self.check_cancel = check_cancel or (lambda: None)
        self.key = None
        self.builder = None
        self.freq_pol = None
        self.coverage = None

    def prepare(self, start_s, end_s):
        self.check_cancel()
        if self.coverage is not None and self.coverage[0] <= start_s and end_s <= self.coverage[1]:
            return
        start = pd.Timestamp(start_s, unit="s", tz="UTC")
        day = start.normalize()
        crossing = end_s > int((day + pd.Timedelta(days=1)).timestamp())
        key = (day.isoformat(), crossing)
        if key == self.key:
            return
        # Release old chunk arrays before loading the next feature slice.
        self.builder = None
        self.key = key
        if crossing:
            left = day + pd.Timedelta(days=1, seconds=-80)
            right = day + pd.Timedelta(days=1, seconds=80)
        else:
            left, right = day, day + pd.Timedelta(days=1, microseconds=-1)
        self.coverage = (int(left.timestamp()), math.ceil(right.timestamp()))
        try:
            features = self.cache.load_features_for_trange(
                [
                    left.strftime("%Y-%m-%d/%H:%M:%S.%f"),
                    right.strftime("%Y-%m-%d/%H:%M:%S.%f"),
                ]
            )
        except ValueError as exc:
            if "No available daily data" in str(exc) or "No cached samples" in str(exc):
                return
            raise
        frequencies = np.asarray(features["freq_pol"], dtype=np.float32)
        if self.freq_pol is not None and not np.array_equal(self.freq_pol, frequencies):
            raise ValueError("Frequency grid changed across dense-scoring cache chunks")
        self.freq_pol = frequencies.copy()
        self.builder = make_precomputed_window_builder(features)

    def __call__(self, start_s, end_s):
        self.prepare(start_s, end_s)
        if self.builder is None:
            self.last_interpolation = {"unavailable_reason": "no_cached_features"}
            return None
        result = self.builder(start_s, end_s)
        self.last_interpolation = dict(self.builder.last_interpolation)
        return result


def center_candidates(dense, threshold, parent_groups):
    """Split high centers at every missing/low second; pad each run to80s.

    The ±20s context and minimum-duration padding may overlap, but never join
    the score runs. Unknown scores break continuity just like low scores.
    """
    groups = {}
    for row in dense:
        score = row.get("uncalibrated_score")
        if score is None or not math.isfinite(float(score)) or float(score) < threshold:
            continue
        start, end = _seconds(row["start"]), _seconds(row["end"])
        if end - start != 80:
            raise ValueError("Dense CNN windows must be 80 s")
        groups.setdefault(row["parent_id"], set()).add(start + 40)
    parents = {g["parent_id"]: g for g in parent_groups}
    candidates = []
    for key, values in sorted(groups.items()):
        centers = sorted(values)
        runs = np.split(np.asarray(centers), np.flatnonzero(np.diff(centers) != 1) + 1)
        left, right = parents[key]["start_s"], parents[key]["end_s"]
        if right - left < 80:
            raise ValueError("CNN parent cannot accommodate an80-second event")
        for number, run in enumerate(runs, 1):
            first, last = int(run[0]), int(run[-1])
            a, b = max(left, first - 20), min(right, last + 20)
            original_a, original_b = a, b
            if b - a < 80:
                a = max(left, min(math.floor((a + b - 80) / 2), right - 80))
                b = a + 80
            candidates.append({
                "parent_id": key, "segment_id": number, "scope_kind": "fine_segment",
                "start_s": a, "end_s": b,
                "unpadded_start_s": original_a, "unpadded_end_s": original_b,
                "minimum_duration_padded": original_b - original_a < 80,
                "high_center_start_s": first, "high_center_end_s": last,
                "high_center_count": len(run),
            })
    return candidates


class WholeEventEvaluator:
    def __init__(self, groups, data_dir, check_cancel=None):
        self.groups = groups
        self.data_dir = data_dir
        self.check_cancel = check_cancel or (lambda: None)
        self.loaded = None
        self.evaluate = None
        self.source = None
        self.load_count = 0
        self.call_count = 0
        self.measurement_errors = []

    def __call__(self, start_s, end_s):
        self.check_cancel()
        # Keep one fixed context for callers sharing a parent group.
        # The dense scan extends 20 s beyond the original coarse group.
        group = next(
            (g for g in self.groups if g["scan_start_s"] <= start_s and end_s <= g["scan_end_s"]),
            None,
        )
        left = min(start_s, group["scan_start_s"]) if group else start_s
        right = max(end_s, group["scan_end_s"]) if group else end_s
        key = (left - 240, right + 240)
        if key != self.loaded:
            self.evaluate = None
            self.source = None
            self.loaded = key
            self.load_count += 1
            try:
                analysis = DataAnalysis(
                    [_format_time(key[0]), _format_time(key[1])], data_dir=self.data_dir
                )
                if analysis.dataframe is not None and len(analysis.dataframe) >= 160:
                    self.evaluate = make_local_physics_evaluator(
                        analysis.dataframe, float(analysis.fs)
                    )
                    self.source = SimpleNamespace(
                        df=analysis.dataframe,
                        fs=float(analysis.fs),
                        t=pd.to_datetime(analysis.dataframe.time_ut, utc=True)
                        .to_numpy(dtype="datetime64[ns]")
                        .astype(np.int64)
                        / 1e9,
                    )
            except (FileNotFoundError, ValueError) as exc:
                self.measurement_errors.append(
                    {"start_s": start_s, "end_s": end_s, "error": str(exc)}
                )
        return self.measure_loaded(start_s, end_s)

    def measure_loaded(self, start_s, end_s):
        """Measure subintervals in the exact already-loaded scope context."""
        self.check_cancel()
        self.call_count += 1
        if self.evaluate is None:
            return None
        try:
            return self.evaluate(start_s, end_s)
        except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
            self.measurement_errors.append({"start_s": start_s, "end_s": end_s, "error": str(exc)})
            return None


def refine_production_events(
    model,
    coarse_results,
    *,
    threshold,
    feature_cache,
    data_dir,
    device=None,
    batch_size=256,
    max_dense_windows=1_000_000,
    check_cancel=None,
    on_progress=None,
    window_provider=None,
    evaluate=None,
    rescue=None,
    source_provider=None,
):
    """Global cross-group dense scoring followed only by whole-event checks.

    Dependency callbacks support numerical/unit verification without database
    writes. No CNN threshold adjustment or local physical prescreening occurs.
    """
    check_cancel = check_cancel or (lambda: None)
    check_cancel()
    groups, plans = plan_dense_group_windows(
        coarse_results, threshold=threshold, max_windows=max_dense_windows
    )
    metadata = {
        "method": METHOD,
        "coordinate": "80s_window_center=start+40s",
        "candidate_margin_seconds": 20,
        "minimum_event_seconds_inclusive": 80,
        "local_physical_prescreening": False,
        "coarse_group_count": len(groups),
    }
    if not plans:
        return [], {
            **metadata,
            "candidate_count": 0,
            "dense_planned": 0,
            "dense_scored": 0,
            "whole_pass_count": 0,
            "rescue_parent_count": 0,
        }
    provider = window_provider or CachedWindowProvider(feature_cache, check_cancel)
    for group in groups:
        start = _seconds(coarse_results[group["source_window_indices"][0]]["start"])
        provider(start, start + 80)
        if provider.freq_pol is not None:
            break

    def progress(done, total):
        check_cancel()
        if on_progress is not None:
            on_progress("dense_scoring", done, total)

    if provider.freq_pol is None:
        # Stored coarse detections remain eligible even when the daily feature
        # cache is unavailable. Raw-data physics can still validate the parent.
        dense = [
            {
                "parent_id": q["parent_id"],
                "start": _format_time(q["start_s"]),
                "end": _format_time(q["end_s"]),
                "uncalibrated_score": None,
            }
            for q in plans
        ]
        dense_meta = {
            "planned_count": len(plans),
            "scored_count": 0,
            "unavailable_count": len(plans),
            "forward_count": 0,
            "groups": [],
            "frequency_grid_unavailable": True,
        }
        progress(len(plans), len(plans))
    else:
        dense, dense_meta = dense_rescore_groups(
            model,
            coarse_results,
            threshold=threshold,
            build_window=provider,
            freq_pol=provider.freq_pol,
            batch_size=batch_size,
            device=device,
            max_windows=max_dense_windows,
            on_progress=progress,
        )
    del plans
    candidates = center_candidates(dense, threshold, groups)
    by_parent = {}
    for row in dense:
        by_parent.setdefault(row["parent_id"], []).append(row)
    # A raw parent is provenance, never a fallback boundary. Every continuous
    # dense high-score segment is independently checked, including raw-pass days.
    physical_evaluate = evaluate or WholeEventEvaluator([], data_dir, check_cancel)
    group_lookup = {g["parent_id"]: g for g in groups}
    candidate_lookup = {(c["parent_id"], c["segment_id"]): c for c in candidates}
    events, records, rejected = [], [], []
    total_scopes = len(candidates)
    completed = 0
    for group in groups:
        key = group["parent_id"]
        scopes = [c for c in candidates if c["parent_id"] == key]
        for candidate in scopes:
            check_cancel()
            a, b = candidate["start_s"], candidate["end_s"]
            if on_progress is not None:
                on_progress("whole_event_physics", completed, total_scopes)
            whole = physical_evaluate(a, b)
            passed = whole is not None and bool(_full_pass(whole))
            completed += 1
            record = {
                "parent_id": key,
                "scope_kind": candidate["scope_kind"],
                "segment_id": candidate["segment_id"],
                "start_s": a,
                "end_s": b,
                "whole_interval_gate_pass": passed if whole is not None else None,
                "whole_interval_measurement": whole,
                "rescue_called": not passed,
            }
            records.append(record)
            if passed:
                selected = [
                    {
                        "start": _format_time(a),
                        "end": _format_time(b),
                        "whole_interval_gate_pass": True,
                        "whole_interval_measurement": whole,
                    }
                ]
            else:
                rows = [
                    r
                    for r in by_parent.get(key, [])
                    if a <= _seconds(r["start"]) + 40 <= b
                    and r.get("uncalibrated_score") is not None
                    and math.isfinite(float(r["uncalibrated_score"]))
                ]
                if on_progress is not None:
                    on_progress("event_rescue", 0, 1)
                if rescue is not None:
                    selected, rescue_meta = rescue(candidate, rows)
                else:
                    source = (
                        source_provider(a, b)
                        if source_provider is not None
                        else getattr(physical_evaluate, "source", None)
                    )
                    # Freeze the exact scope context for every subinterval FFT;
                    # the scope evaluator above has already loaded a/b +/-240s.
                    scope_evaluate = getattr(physical_evaluate, "measure_loaded", physical_evaluate)
                    if source is None or scope_evaluate is None:
                        selected, rescue_meta = (
                            [],
                            {
                                "method": "stft_support_segments_bounded_uncapped_v1",
                                "status": "raw_context_unavailable",
                                "checked": 0,
                            },
                        )
                    else:
                        selected, rescue_meta = rescue_stft_scope(
                            source,
                            scope_evaluate,
                            a,
                            b,
                            check_cancel=check_cancel,
                            on_progress=(
                                lambda done, total: on_progress("event_rescue", done, total)
                            )
                            if on_progress is not None
                            else None,
                        )
                record["rescue_metadata"] = rescue_meta
                check_cancel()
                if on_progress is not None:
                    on_progress("event_rescue", 1, 1)
            for event in selected:
                event = {
                    **event,
                    "parent_id": key,
                    "segment_id": candidate["segment_id"],
                    "definition": "whole_scope_six_gate_v1"
                    if passed
                    else "stft_after_whole_failure_v1",
                }
                event["model_diagnostics"] = {
                    **(event.get("model_diagnostics") or {}),
                    "event_scope_kind": candidate["scope_kind"],
                    "event_scope_start": _format_time(a),
                    "event_scope_end": _format_time(b),
                }
                duration = _seconds(event["end"]) - _seconds(event["start"])
                if duration < 80:
                    rejected.append(
                        {
                            "parent_id": key,
                            "duration_seconds": duration,
                            "reason": "final_duration_must_be_at_least_80_seconds",
                        }
                    )
                else:
                    # Production kernels guarantee this; an invalid dependency
                    # callback must not silently write an out-of-parent event.
                    if (
                        rescue is None
                        and not candidate["start_s"]
                        <= _seconds(event["start"])
                        < _seconds(event["end"])
                        <= candidate["end_s"]
                    ):
                        raise ValueError("STFT output escaped original CNN parent")
                    events.append(event)
    physical_meta = {
        "method": "continuous_fine_segments_whole_then_stft_v2",
        "candidates": records,
        "rejected_duration": rejected,
        "local_window_prescreening": False,
    }
    if on_progress is not None:
        on_progress("whole_event_physics", total_scopes, total_scopes)
    for event in events:
        key = event["parent_id"]
        a, b = _seconds(event["start"]), _seconds(event["end"])
        sources = [
            r
            for r in by_parent[key]
            if r.get("uncalibrated_score") is not None
            and float(r["uncalibrated_score"]) >= threshold
            and a <= _seconds(r["start"]) + 40 <= b
        ]
        score_source = "dense_high_window"
        if not sources:
            sources = [
                r
                for r in by_parent[key]
                if r.get("uncalibrated_score") is not None
                and float(r["uncalibrated_score"]) >= threshold
            ]
        if not sources:
            sources = [coarse_results[i] for i in group_lookup[key]["source_window_indices"]]
            score_source = "coarse_high_window"
        peak = max(sources, key=lambda row: row["uncalibrated_score"])
        fine = candidate_lookup[(key, event["segment_id"])]
        # Rescued events keep their own physics/peak diagnostics. Whole-pass
        # events need the production fields consumed by DB and sample views.
        event.setdefault("score", peak.get("score", peak["uncalibrated_score"]))
        event.setdefault("raw_score", peak.get("raw_score", peak["uncalibrated_score"]))
        event.setdefault("uncalibrated_score", peak["uncalibrated_score"])
        event.setdefault("calibrated_probability", peak.get("calibrated_probability"))
        diagnostics = dict(event.get("model_diagnostics") or {})
        diagnostics.update(
            event_postprocess_method=METHOD,
            event_dense_segment_id=event["segment_id"],
            event_high_center_continuity_seconds=1,
            event_score_source=score_source,
            event_candidate_coordinate="window_center",
            event_candidate_margin_seconds=20,
            event_minimum_duration_seconds_inclusive=80,
            event_raw_score_threshold=float(threshold),
            event_original_group_id=key,
            event_original_group_start=_format_time(group_lookup[key]["start_s"]),
            event_original_group_end=_format_time(group_lookup[key]["end_s"]),
            event_candidate_start=_format_time((fine or group_lookup[key])["start_s"]),
            event_candidate_end=_format_time((fine or group_lookup[key])["end_s"]),
            event_candidate_minimum_duration_padded=fine["minimum_duration_padded"]
            if fine
            else False,
            event_candidate_unpadded_start=_format_time(
                (fine or group_lookup[key]).get("unpadded_start_s", group_lookup[key]["start_s"])
            ),
            event_candidate_unpadded_end=_format_time(
                (fine or group_lookup[key]).get("unpadded_end_s", group_lookup[key]["end_s"])
            ),
            event_high_score_center_start=_format_time(fine["high_center_start_s"])
            if fine
            else None,
            event_high_score_center_end=_format_time(fine["high_center_end_s"]) if fine else None,
        )
        if event.get("whole_interval_gate_pass"):
            v = event["whole_interval_measurement"]
            m = v["metrics"]
            diagnostics.update(
                event_fmax_over_fcp=m["fmax_over_fcp"],
                event_ellip_at_fmax=v["ellip_at_fmax"],
                event_wna_at_fmax=v["wna_at_fmax"],
                event_pf_band_ratio=m["pf_band_ratio"],
                event_perp_parallel_energy_ratio=m["energy_ratio"],
                event_data_missing_ratio=m["data_missing_ratio"],
                event_data_coverage_gate_pass=True,
                event_energy_gate_pass=True,
                pf_gate_pass=True,
                pf_gate_status="pass",
            )
            diagnostics.setdefault("event_rescue_applied", False)
        event["model_diagnostics"] = diagnostics
    check_cancel()
    # Collapse only identical boundaries. Different overlapping intervals remain
    # independently measured alternatives; never manufacture their union.
    unique = {}
    for event in events:
        boundary = (event["start"], event["end"])
        if boundary in unique:
            unique[boundary]["model_diagnostics"].setdefault("event_identical_sources", []).append(
                {"parent_id": event["parent_id"], "segment_id": event["segment_id"]})
        else:
            unique[boundary] = event
    merged_events, overlap_audit = [], []
    for number, cluster in enumerate(_overlap_groups(list(unique.values())), 1):
        cluster.sort(key=lambda e: (_seconds(e["start"]), _seconds(e["end"])))
        for member, event in enumerate(cluster, 1):
            event["model_diagnostics"].update(
                event_overlap_group=number, event_overlap_member=member,
                event_overlap_member_count=len(cluster), event_merge_applied=False,
                event_overlap_policy="identical_only_dedup_preserve_distinct_boundaries")
            merged_events.append(event)
        if len(cluster) > 1:
            overlap_audit.append({"action": "preserve_members", "group": number,
                "member_count": len(cluster),
                "members": [{"start": e["start"], "end": e["end"]} for e in cluster]})
    # Apply the inclusive minimum after every rescue and merge stage.
    rejected_final_duration = [
        {
            "start": e["start"],
            "end": e["end"],
            "duration_seconds": _seconds(e["end"]) - _seconds(e["start"]),
            "reason": "production_final_duration_must_be_at_least_80_seconds",
        }
        for e in merged_events
        if _seconds(e["end"]) - _seconds(e["start"]) < 80
    ]
    events = [e for e in merged_events if _seconds(e["end"]) - _seconds(e["start"]) >= 80]
    events.sort(key=lambda e: (_seconds(e["start"]), _seconds(e["end"])))
    check_cancel()
    return events, {
        **metadata,
        "candidate_count": len(events),
        "final_output_strictly_nonoverlapping": not overlap_audit,
        "distinct_overlapping_outputs_preserved": True,
        "failed_union_best_member_selections": sum(a.get("action") == "keep_best_member" for a in overlap_audit),
        "production_minimum_duration_seconds_inclusive": 80,
        "rejected_final_duration": rejected_final_duration,
        "candidates": candidates,
        "dense_planned": dense_meta["planned_count"],
        "dense_scored": dense_meta["scored_count"],
        "dense_unavailable": dense_meta["unavailable_count"],
        "forward_batches": dense_meta["forward_count"],
        "dense_batch_size": batch_size,
        "minimum_duration_padded_candidates": sum(c["minimum_duration_padded"] for c in candidates),
        "interpolated_dense_windows": sum(
            r.get("input_interpolation", {}).get("interpolated_seconds", 0) > 0 for r in dense
        ),
        "interpolated_dense_seconds": sum(
            r.get("input_interpolation", {}).get("interpolated_seconds", 0) for r in dense
        ),
        "maximum_single_interpolation_gap_seconds": 4,
        "dense_edge_status": dense_meta["groups"],
        "whole_pass_count": sum(
            c["whole_interval_gate_pass"] is True for c in physical_meta["candidates"]
        ),
        "rescue_parent_count": len({c["parent_id"] for c in records if c["rescue_called"]}),
        "rescue_scope_count": sum(c["rescue_called"] for c in records),
        "physical_audit": physical_meta,
        "overlap_reconciliation": overlap_audit,
        "unavailable_physical_measurements": getattr(physical_evaluate, "measurement_errors", []),
        "groups_without_high_dense_scores": [
            g["parent_id"] for g in groups if not any(c["parent_id"] == g["parent_id"] for c in candidates)
        ],
    }
