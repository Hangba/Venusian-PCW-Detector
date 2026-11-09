"""Compute the paper tables from frozen inputs or a completed raw search."""

import gzip
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from venus_express.constants import R_VENUS
from venus_express.repro import occurrence
from venus_express.repro.density import compute as density
from venus_express.repro.model import PROTOCOL
from venus_express.repro.raw import raw_root
from venus_express.repro.settings import ROOT, INPUTS, BUILD, checked_file, read, write
from venus_express.sample_analysis.calculations.chamberlain import chamberlain_density_cm3
from venus_express.sample_analysis.calculations.paper_integrated_density import (
    SPICAV_HOT_H_ALTITUDE_KM,
    spicav_hot_h_density_cm3,
)

from . import boundaries, energy, windows


def verify_inputs():
    items = read(ROOT / "reference/input_lineage.json")
    for name, spec in items.items():
        checked_file(ROOT / name, spec["sha256"])
    print(f"Verified {len(items)} frozen inputs, checkpoints and reference assets")


def review(raw):
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest

    labels = pd.read_csv(INPUTS / "holdout/review.csv")
    ranked = read(
        BUILD / "raw/holdout/ranked.json" if raw else INPUTS / "holdout/deduplicated_ranked.json"
    )
    assert len(ranked) == 245, f"Expected 245 held-out candidates, got {len(ranked)}"
    queue = pd.DataFrame(ranked[:50]).assign(rank=range(1, 51))
    # A label belongs to its reviewed interval, never to whichever event now occupies a rank.
    result = queue.merge(
        labels.drop(columns=["rank", "precision_at_k"]),
        on=["start_s", "end_s"],
        how="left",
        validate="one_to_one",
    )
    if result.human_label.isna().any():
        raise ValueError(
            "The recomputed top-50 queue contains intervals without frozen human labels"
        )
    result["precision_at_k"] = result.human_label.eq("PCW").cumsum() / result["rank"]
    np.testing.assert_array_equal(result.human_label, labels.sort_values("rank").human_label)
    result.to_csv(BUILD / "review_ranked.csv", index=False)
    summary = {
        "candidates": len(ranked),
        "labels": result.human_label.value_counts().to_dict(),
        "reviews_to_confirm_PCW": {},
    }
    cumulative = result.human_label.eq("PCW").cumsum()
    intervals = {}
    for k in [10, 25, 50]:
        hits = int(cumulative.iloc[k - 1])
        ci = binomtest(hits, k).proportion_ci(method="wilson")
        intervals[f"P@{k}"] = dict(
            PCW=hits, K=k, precision=hits / k, wilson95=[round(ci.low, 4), round(ci.high, 4)]
        )
    for hits in [10, 20, 30]:
        summary["reviews_to_confirm_PCW"][str(hits)] = int(
            np.flatnonzero(cumulative >= hits)[0] + 1
        )
    write(BUILD / "review_summary.json", summary)
    write(BUILD / "precision_ci.json", intervals)
    return intervals


def cluster_positions(raw):
    import numpy as np
    import pandas as pd

    grid = pd.read_csv(INPUTS / "mission/cluster_occurrence.csv")
    if raw and (BUILD / "raw/exposure/complete.json").exists():
        from venus_express.repro.raw import raw_root

        meta = read(BUILD / "raw/exposure/complete.json")
        if meta["raw"] != str(raw_root()):
            raise ValueError("Exposure and search must use the same MAG directory")
        rebuilt = pd.read_csv(BUILD / "raw/exposure/grid.csv")
        np.testing.assert_array_equal(grid.observed_seconds, rebuilt.observed_seconds)
        grid["observed_seconds"] = rebuilt.observed_seconds
        grid["observed_days"] = grid.observed_seconds / 86400
    clusters = pd.read_csv(BUILD / "occurrence/clusters_10min.csv")
    clusters = clusters[np.isclose(clusters.threshold, 0.7)].copy()
    if raw:
        from venus_express.repro.spatial import position_day

        clusters["day"] = pd.to_datetime(
            clusters.representative_time_s, unit="s", utc=True
        ).dt.strftime("%Y-%m-%d")
        rows = []
        for day, group in clusters.groupby("day"):
            _, pos = position_day((day, group.to_dict("records")))
            rows.extend(pos)
        positions = pd.DataFrame(rows).drop(columns="raw_file")
    else:
        positions = pd.read_csv(INPUTS / "mission/cluster_positions.csv")
    comparison = clusters[["cluster_id", "representative_interval_id", "representative_time_s"]]
    pd.testing.assert_frame_equal(
        comparison.sort_values("cluster_id").reset_index(drop=True),
        positions[comparison.columns].sort_values("cluster_id").reset_index(drop=True),
        check_dtype=False,
    )
    bins = pd.DataFrame(
        {
            "x_bin_min_rv": np.floor(positions.AVSO_X_Rv / 0.5) * 0.5,
            "r_bin_min_rv": np.floor(positions.AVSO_R_Rv / 0.5) * 0.5,
        }
    )
    keys = ["x_bin_min_rv", "r_bin_min_rv"]
    counts = bins.groupby(keys).size()
    grid["cluster_count"] = counts.reindex(
        pd.MultiIndex.from_frame(grid[keys]), fill_value=0
    ).to_numpy()
    assert grid.cluster_count.sum() == len(positions)
    assert grid.observed_seconds.sum() == 233963418
    grid["clusters_per_observed_day"] = (grid.cluster_count / grid.observed_days).where(
        grid.observed_days > 0
    )
    positions.to_csv(BUILD / "occurrence/cluster_positions.csv", index=False)
    grid.to_csv(BUILD / "occurrence/cluster_occurrence.csv", index=False)
    return positions, grid


def verify_results(density):
    import numpy as np
    import pandas as pd

    for name in (
        "mission_metrics_10min",
        "annual_metrics_10min",
        "catalogue_overlap",
        "parent_outcomes_summary",
    ):
        pd.testing.assert_frame_equal(
            pd.read_csv(BUILD / f"occurrence/{name}.csv"),
            pd.read_csv(ROOT / f"reference/{name}.csv"),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    expected = read(ROOT / "reference/density_fit.json")["cases"]["no_intersection"]
    for key in (
        "events",
        "days",
        "n25",
        "n25_95",
        "T",
        "T_95",
        "n0",
        "log10_rmse",
        "log10_r_squared",
    ):
        np.testing.assert_allclose(density[key], expected[key], rtol=3e-6, atol=1e-9)


def rates():
    dest = BUILD / "occurrence"
    annual = pd.read_csv(dest / "annual_metrics_10min.csv")
    sun = pd.read_csv(
        INPUTS / "silso.csv", sep=";", header=None, usecols=[0, 1], names=["year", "sunspot_number"]
    )
    sun["year"] = sun.year.astype(int)
    annual = annual.merge(sun, on="year", validate="many_to_one")
    annual.to_csv(dest / "annual_rates_with_silso.csv", index=False)
    primary = annual[np.isclose(annual.threshold, 0.7)].sort_values("year")
    r = pearsonr(primary.clusters_per_observed_day, primary.sunspot_number)
    low = primary[primary.year.isin([2008, 2009])]
    high = primary[primary.year.isin([2013, 2014])]
    ratio = (high.clusters.sum() / high.observed_seconds.sum()) / (
        low.clusters.sum() / low.observed_seconds.sum()
    )
    interval = np.exp(
        np.log(ratio)
        + np.array([-1.0, 1.0]) * 1.96 * np.sqrt(1 / high.clusters.sum() + 1 / low.clusters.sum())
    )
    return dict(
        r=float(r.statistic),
        p=float(r.pvalue),
        high_low_ratio=ratio,
        poisson_ratio_ci95=interval.tolist(),
    ), primary.sunspot_number.to_numpy()


def density_profile(result):
    dest = BUILD / "density"
    boot = pd.read_csv(dest / "no_intersection_bootstrap.csv")
    a, b = result["altitude_range_km"]
    low, high = 0.7 * R_VENUS, 12 * R_VENUS
    spicav = SPICAV_HOT_H_ALTITUDE_KM
    h = np.unique(
        np.r_[
            np.geomspace(low, high, 1000),
            25000,
            8000,
            a,
            b,
            spicav[(spicav >= low) & (spicav <= high)],
        ]
    )
    rv = 1 + h / R_VENUS
    fit = chamberlain_density_cm3(rv, result["n0"], result["T"])
    curves = [chamberlain_density_cm3(rv, row.n0, row.T) for row in boot.itertuples()]
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    pd.DataFrame(
        dict(
            alt_km=h,
            n_fit=fit,
            n_lo=lo,
            n_hi=hi,
            SPICAV_hot_cm3=spicav_hot_h_density_cm3(h),
            within_event_altitude_range=(h >= a) & (h <= b),
        )
    ).to_csv(dest / "square_12Rv_bootstrap_profile.csv", index=False)


def peak_summary(frame):
    usable = frame.status.isin(["in_band", "out_of_band"])
    outside = frame.status.eq("out_of_band")
    return dict(
        events=len(frame),
        usable=int(usable.sum()),
        outside=int(outside.sum()),
        below=int((outside & frame.peak_over_fci.lt(0.8)).sum()),
        above=int((outside & frame.peak_over_fci.gt(1.2)).sum()),
    )


def compute(raw_search=False, raw_density=False, rescore_windows=False, device="cpu"):
    verify_inputs()
    BUILD.mkdir(parents=True, exist_ok=True)
    if raw_search:
        for scope, key in [("mission", "mission_days"), ("holdout", "holdout_days")]:
            inputs = read(BUILD / f"raw/{scope}/complete.json")["inputs"]
            if inputs["raw"] != str(raw_root()) or inputs["days"] != PROTOCOL[key]:
                raise ValueError(
                    f"Complete the {scope} search with the current data provider first"
                )
        outcomes = read(BUILD / "raw/mission/parent_outcomes.json")
    else:
        with gzip.open(INPUTS / "mission/parent_outcomes.json.gz", "rt") as handle:
            outcomes = json.load(handle)
    primary = occurrence.aggregate(outcomes)
    precision = review(raw_search)
    cluster_positions(raw_search)
    fit = density(raw_density)
    verify_results(fit)
    density_profile(fit)
    annual, sun = rates()
    peak = peak_summary(pd.read_csv(INPUTS / "manual/psd_audit.csv"))
    assert peak == dict(events=946, usable=917, outside=245, below=223, above=22)
    runtime = pd.read_csv(INPUTS / "model_runtime.csv")
    escape = 4 * np.pi * (5 * 6052e3) ** 2 * 5e3 * 400e3
    result = dict(
        occurrence=primary,
        precision=precision,
        density=fit,
        annual_occurrence=annual,
        wave_energy=energy.compute(sun),
        boundary=boundaries.compute(),
        psd_audit=peak,
        escape_ions_s=escape,
        training_runtime_s=runtime.elapsed_seconds.sum(),
        holdout_runtime=read(INPUTS / "holdout/inference_runtime.json"),
        window_metrics=windows.compute(rescore_windows, device),
    )
    write(BUILD / "summary.json", result)
    print(
        f"Paper checks: {int(primary['clusters'])} clusters, {precision['P@50']['PCW']}/50 reviewed PCWs, "
        f"H at 25,000 km = {fit['n25']:.3f} cm^-3, {result['boundary']['imb_identified']}/50 IMB"
    )
