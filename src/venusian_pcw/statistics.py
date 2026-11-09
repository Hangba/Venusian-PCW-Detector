"""Aggregate search results generated locally from raw MAG."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr

from venus_express.repro.model import PROTOCOL
from venus_express.repro.raw import raw_root
from venus_express.repro.settings import ROOT, INPUTS, BUILD, checked_file, read, write

from .solar import annual_sunspots


def verify_inputs():
    models = [*PROTOCOL["models"], PROTOCOL["holdout2013"]]
    for spec in models:
        checked_file(ROOT / spec["checkpoint"], spec["checkpoint_sha256"])
    from .catalogue import read_events

    events = read_events(INPUTS / "manual/event_list.csv")
    print(f"Verified {len(models)} model weights and {len(events)} annotated events")


def review():
    labels = pd.read_csv(INPUTS / "holdout/review.csv")
    ranked = read(BUILD / "raw/holdout/ranked.json")
    queue = pd.DataFrame(ranked[:50], columns=["start_s", "end_s", "final_score"])
    queue["rank"] = np.arange(1, len(queue) + 1)
    result = queue.merge(labels, on=["start_s", "end_s"], how="left", validate="one_to_one")
    result.to_csv(BUILD / "review_ranked.csv", index=False)
    intervals = {}
    for k in (10, 25, 50):
        top = result.head(k)
        if len(top) < k or top.human_label.isna().any():
            continue
        hits = int(top.human_label.eq("PCW").sum())
        ci = binomtest(hits, k).proportion_ci(method="wilson")
        intervals[f"P@{k}"] = dict(
            PCW=hits, K=k, precision=hits / k, wilson95=[round(ci.low, 4), round(ci.high, 4)]
        )
    return dict(
        candidates=len(ranked),
        reviewed_in_top_50=int(result.human_label.notna().sum()),
        precision=intervals,
    )


def cluster_positions(grid):
    from venus_express.repro.spatial import position_day

    out = BUILD / "occurrence"
    clusters = pd.read_csv(out / "clusters_10min.csv")
    clusters = clusters[np.isclose(clusters.threshold, 0.7)].copy()
    clusters["day"] = pd.to_datetime(
        clusters.representative_time_s, unit="s", utc=True
    ).dt.strftime("%Y-%m-%d")
    rows = [
        row
        for day, group in clusters.groupby("day")
        for row in position_day((day, group.to_dict("records")))[1]
    ]
    positions = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["AVSO_X_Rv", "AVSO_R_Rv"])
    positions = positions.drop(columns="raw_file", errors="ignore")
    bins = pd.DataFrame(
        {
            "x_bin_min_rv": np.floor(positions.AVSO_X_Rv.astype(float) / 0.5) * 0.5,
            "r_bin_min_rv": np.floor(positions.AVSO_R_Rv.astype(float) / 0.5) * 0.5,
        }
    )
    keys = ["x_bin_min_rv", "r_bin_min_rv"]
    counts = bins.value_counts().rename("cluster_count").reset_index()
    grid = grid.merge(counts, on=keys, how="outer")
    grid["cluster_count"] = grid.cluster_count.fillna(0).astype(int)
    grid["observed_seconds"] = grid.observed_seconds.fillna(0).astype(int)
    grid["observed_days"] = grid.observed_seconds / 86400
    grid["displayed"] = grid.observed_days.ge(2)
    grid["clusters_per_observed_day"] = grid.cluster_count.div(
        grid.observed_days.where(grid.observed_days.gt(0))
    )
    positions.to_csv(out / "cluster_positions.csv", index=False)
    grid.to_csv(out / "cluster_occurrence.csv", index=False)


def rates():
    out = BUILD / "occurrence"
    annual = pd.read_csv(out / "annual_metrics_10min.csv")
    sun = annual_sunspots()
    annual = annual.merge(sun, on="year", validate="many_to_one")
    annual.to_csv(out / "annual_rates_with_silso.csv", index=False)
    primary = annual[np.isclose(annual.threshold, 0.7)].sort_values("year")
    result = {}
    if (
        len(primary) >= 2
        and primary.clusters_per_observed_day.nunique() >= 2
        and primary.sunspot_number.nunique() >= 2
    ):
        r = pearsonr(primary.clusters_per_observed_day, primary.sunspot_number)
        result.update(r=float(r.statistic), p=float(r.pvalue))
    low = primary[primary.year.isin([2008, 2009])]
    high = primary[primary.year.isin([2013, 2014])]
    if len(low) == len(high) == 2 and low.clusters.sum() and high.clusters.sum():
        ratio = (high.clusters.sum() / high.observed_seconds.sum()) / (
            low.clusters.sum() / low.observed_seconds.sum()
        )
        interval = np.exp(
            np.log(ratio)
            + np.array([-1.0, 1.0])
            * 1.96
            * np.sqrt(1 / high.clusters.sum() + 1 / low.clusters.sum())
        )
        result.update(high_low_ratio=ratio, poisson_ratio_ci95=interval.tolist())
    return result


def peak_summary(frame):
    usable = frame.status.isin(["in_band", "out_of_band"])
    outside = frame.status.eq("out_of_band")
    ratio = frame.get("peak_over_fci", pd.Series(np.nan, index=frame.index))
    return dict(
        events=len(frame),
        usable=int(usable.sum()),
        outside=int(outside.sum()),
        below=int((outside & ratio.lt(0.8)).sum()),
        above=int((outside & ratio.gt(1.2)).sum()),
    )


def compute(workers=4):
    from venus_express.repro import exposure, occurrence

    mission = BUILD / "raw/mission"
    if not (mission / "complete.json").is_file():
        raise FileNotFoundError("Run search mission on the desired MAG days before replay")
    context = read(mission / "complete.json")["inputs"]
    if context["raw"] != str(raw_root()):
        raise ValueError("Replay requires the same MAG data source used for the search")
    days = context["days"]
    cover = BUILD / "raw/exposure"
    meta = cover / "complete.json"
    wanted = dict(raw=context["raw"], days=days)
    if (
        not meta.exists()
        or any(read(meta).get(k) != v for k, v in wanted.items())
        or not all((cover / name).is_file() for name in ("days.csv", "grid.csv"))
    ):
        exposure.compute(workers, days)
    primary = occurrence.aggregate(
        read(mission / "parent_outcomes.json"), pd.read_csv(cover / "days.csv")
    )
    cluster_positions(pd.read_csv(cover / "grid.csv"))
    result = dict(occurrence=primary, annual_occurrence=rates())
    holdout = BUILD / "raw/holdout/complete.json"
    if holdout.exists() and read(holdout)["inputs"]["raw"] == context["raw"]:
        result["holdout_review"] = review()
    write(BUILD / "summary.json", result)
    print(
        f"{int(primary['clusters'])} clusters, {int(primary['observed_seconds'])} observed seconds"
    )
    return result
