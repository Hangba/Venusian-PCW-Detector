"""Merge qualified intervals and compute exposure-normalized occurrence."""

import numpy as np
import pandas as pd
from .localization import utc
from .model import PROTOCOL
from .settings import BUILD, INPUTS

CUTS = PROTOCOL["mission_cuts"]
GAPS = [PROTOCOL["cluster_gap_s"] // 60]


def grouped_intervals(records, gap_s):
    """Connected intervals, including a gap exactly equal to the threshold."""
    groups, end = [], None
    for r in sorted(records, key=lambda r: (r["start_s"], r["end_s"], r["interval_id"])):
        if end is None or r["start_s"] - end > gap_s:
            groups.append([r])
            end = r["end_s"]
        else:
            groups[-1].append(r)
            end = max(end, r["end_s"])
    return groups


def union_seconds(records):
    return sum(
        max(r["end_s"] for r in g) - min(r["start_s"] for r in g)
        for g in grouped_intervals(records, 0)
    )


def aggregate(rows, exposure_days):
    occ = BUILD / "occurrence"
    occ.mkdir(parents=True, exist_ok=True)
    items = []
    for r in rows:
        for q in r["qualified"]:
            items.append(dict(threshold=r["threshold"], parent_id=r["parent_id"], **q))
    q = pd.DataFrame(items, columns=[
        "threshold", "parent_id", "start_s", "end_s", "path", "final_score",
        "fmax_over_fcp", "pf_band_ratio", "energy_ratio", "data_missing_ratio", "ellip", "wna",
    ] + sorted({key for item in items for key in item} - {
        "threshold", "parent_id", "start_s", "end_s", "path", "final_score",
        "fmax_over_fcp", "pf_band_ratio", "energy_ratio", "data_missing_ratio", "ellip", "wna",
    })).sort_values(["threshold", "start_s", "end_s", "parent_id"])
    q["duration_s"] = q.end_s - q.start_s
    q = q.drop_duplicates(["threshold", "start_s", "end_s"]).reset_index(drop=True)
    passed = (
        q.fmax_over_fcp.between(0.8, 1.2)
        & q.pf_band_ratio.ge(3)
        & q.energy_ratio.ge(5)
        & q.data_missing_ratio.le(0.1)
        & q.ellip.le(-0.5)
        & q.wna.lt(30)
        & q.duration_s.ge(80)
    )
    assert passed.all()
    q["interval_id"] = [
        f"Q{int(round(c * 100)):03d}-{i:05d}"
        for c, i in zip(q.threshold, q.groupby("threshold").cumcount() + 1)
    ]
    q["start_utc"] = [utc(x) for x in q.start_s]
    q["end_utc"] = [utc(x) for x in q.end_s]
    q.to_csv(occ / "qualified_intervals.csv", index=False)

    observed = exposure_days.assign(year=exposure_days.day.str[:4].astype(int))
    exp = observed.groupby("year", as_index=False).agg(
        observed_seconds=("observed_seconds", "sum"),
        scorable_calendar_days=("observed_seconds", lambda x: x.gt(0).sum()),
    )
    exp = exp[exp.observed_seconds > 0].assign(observed_days=lambda t: t.observed_seconds / 86400)
    total = int(exp.observed_seconds.sum())
    if not total:
        raise ValueError("No valid MAG exposure on the selected dates")

    clusters, members, annual, mission = [], [], [], []
    for cut in CUTS:
        records = q[q.threshold == cut].to_dict("records")
        for gap in GAPS:
            groups = grouped_intervals(records, gap * 60)
            own = []
            for i, group in enumerate(groups, 1):
                cid = f"C{int(round(cut * 100)):03d}-G{gap:02d}-{i:05d}"
                a = min(r["start_s"] for r in group)
                b = max(r["end_s"] for r in group)
                rep = min(
                    group,
                    key=lambda r: (-r["duration_s"], r["start_s"], r["end_s"], r["interval_id"]),
                )
                row = dict(
                    threshold=cut,
                    gap_minutes=gap,
                    cluster_id=cid,
                    start_s=int(a),
                    end_s=int(b),
                    start_utc=utc(a),
                    end_utc=utc(b),
                    start_year=pd.Timestamp(a, unit="s", tz="UTC").year,
                    member_count=len(group),
                    qualified_union_seconds=union_seconds(group),
                    representative_interval_id=rep["interval_id"],
                    representative_start_s=int(rep["start_s"]),
                    representative_end_s=int(rep["end_s"]),
                    representative_time_s=(rep["start_s"] + rep["end_s"]) / 2,
                    max_final_score=max(r["final_score"] for r in group),
                )
                own.append(row)
                members.extend(
                    dict(
                        threshold=cut,
                        gap_minutes=gap,
                        cluster_id=cid,
                        interval_id=r["interval_id"],
                        start_s=int(r["start_s"]),
                        end_s=int(r["end_s"]),
                        path=r["path"],
                        representative=r["interval_id"] == rep["interval_id"],
                    )
                    for r in group
                )
            clusters.extend(own)
            mission.append(
                dict(
                    threshold=cut,
                    gap_minutes=gap,
                    qualified_intervals=len(records),
                    clusters=len(own),
                    single_member_clusters=sum(r["member_count"] == 1 for r in own),
                    observed_seconds=total,
                    observed_days=total / 86400,
                    clusters_per_observed_day=len(own) * 86400 / total,
                    clusters_per_100_observed_days=len(own) * 8640000 / total,
                )
            )
            for e in exp.itertuples():
                n = sum(r["start_year"] == e.year for r in own)
                annual.append(
                    dict(
                        threshold=cut,
                        gap_minutes=gap,
                        year=e.year,
                        clusters=n,
                        observed_seconds=e.observed_seconds,
                        observed_days=e.observed_days,
                        scorable_calendar_days=e.scorable_calendar_days,
                        clusters_per_observed_day=n / e.observed_days,
                        clusters_per_100_observed_days=100 * n / e.observed_days,
                    )
                )
    cf = pd.DataFrame(clusters, columns=[
        "threshold", "gap_minutes", "cluster_id", "start_s", "end_s", "start_utc", "end_utc",
        "start_year", "member_count", "qualified_union_seconds", "representative_interval_id",
        "representative_start_s", "representative_end_s", "representative_time_s", "max_final_score",
    ])
    mf = pd.DataFrame(members, columns=["threshold", "gap_minutes", "cluster_id", "interval_id",
                                      "start_s", "end_s", "path", "representative"])
    af, sf = pd.DataFrame(annual), pd.DataFrame(mission)
    for name, frame in [
        ("clusters", cf),
        ("cluster_members", mf),
        ("annual_metrics", af),
        ("mission_metrics", sf),
    ]:
        frame.to_csv(occ / f"{name}.csv", index=False)
        frame[frame.gap_minutes == 10].to_csv(occ / f"{name}_10min.csv", index=False)
    for (cut, gap), a in cf.groupby(["threshold", "gap_minutes"]):
        a = a.sort_values("start_s")
        assert (a.start_s.to_numpy()[1:] - a.end_s.to_numpy()[:-1] > gap * 60).all()
        assert a.member_count.sum() == (q.threshold == cut).sum()
    po = pd.DataFrame(
        [
            dict(
                threshold=r["threshold"],
                parent_id=r["parent_id"],
                status=r["status"],
                mode=r["mode"],
                qualified=len(r["qualified"]),
                span_s=(r["span_end_s"] - r["span_start_s"]) if r["span_start_s"] else None,
                parent_s=r["parent_end_s"] - r["parent_start_s"],
                leaves_parent=bool(
                    r["span_start_s"] is not None
                    and (
                        r["span_start_s"] < r["parent_start_s"]
                        or r["span_end_s"] > r["parent_end_s"]
                    )
                ),
            )
            for r in rows
        ]
    )
    if po.empty:
        return sf[(sf.threshold == 0.7) & (sf.gap_minutes == 10)].iloc[0].to_dict()
    outcome = (
        po.groupby("threshold")
        .apply(
            lambda g: pd.Series(
                dict(
                    parents=len(g),
                    resolved=int((g.status == "resolved").sum()),
                    unresolved=int((g.status != "resolved").sum()),
                    whole_pass=int((g["mode"] == "whole_pass").sum()),
                    rescue_attempted=int((g["mode"] == "rescue").sum()),
                    rescued=int(((g["mode"] == "rescue") & (g.qualified > 0)).sum()),
                    parents_with_output=int((g.qualified > 0).sum()),
                    span_leaves_parent=int(g.leaves_parent.sum()),
                    median_span_s=float(g.span_s.median()),
                    median_parent_s=float(g.parent_s.median()),
                )
            ),
            include_groups=False,
        )
        .reset_index()
    )
    outcome.to_csv(occ / "parent_outcomes_summary.csv", index=False)
    # Overlap with the manual catalogue: a coverage description, not recall (the catalogue is incomplete).
    cat = pd.read_csv(INPUTS / "manual/event_list.csv")
    ta = (
        pd.to_datetime(cat.start, utc=True).to_numpy(dtype="datetime64[ns]").astype("int64")
        // 10**9
    )
    tb = pd.to_datetime(cat.end, utc=True).to_numpy(dtype="datetime64[ns]").astype("int64") // 10**9
    cover = []
    for cut, e in q.groupby("threshold"):
        s, t = e.start_s.to_numpy(), e.end_s.to_numpy()
        hit = np.array([(np.minimum(t, b) - np.maximum(s, a) > 0).any() for a, b in zip(ta, tb)])
        cover.append(
            dict(
                threshold=cut,
                catalogue_events=len(cat),
                overlapped=int(hit.sum()),
                fraction=float(hit.mean()),
            )
        )
    pd.DataFrame(cover).to_csv(occ / "catalogue_overlap.csv", index=False)

    primary = sf[(sf.threshold == 0.7) & (sf.gap_minutes == 10)].iloc[0].to_dict()
    return primary
