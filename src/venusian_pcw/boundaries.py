"""Read the bow-shock measurements and tabulate the IMF annotations."""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from venus_express.repro.settings import INPUTS, BUILD


def bow_shock_curve(x0, L, eps, theta=np.linspace(-2.4, 2.4, 600)):
    rr = L / (1 + eps * np.cos(theta))
    x, y = x0 + rr * np.cos(theta), rr * np.sin(theta)
    keep = y >= 0
    return x[keep], y[keep]


def bow_shock_data():
    measurements = pd.read_csv(INPUTS / "boundaries/compression.csv", float_precision="round_trip")
    compression = {
        (row.period, row.sza): {"mean": row.mean, "sd": row.sd}
        for row in measurements.itertuples(index=False)
    }
    locations = pd.read_csv(INPUTS / "boundaries/shock_positions.csv", float_precision="round_trip")
    points = {
        period: group[["X_AVSO_Rv", "R_AVSO_Rv"]].to_numpy()
        for period, group in locations.groupby("period", sort=False)
    }
    return compression, points


def compute():
    compression, points = bow_shock_data()
    counts = {"low": [99, 40], "high": [68, 24]}
    rows = []
    for i, group in enumerate(["0—45°", "80—100°"]):
        low, high = (compression[period, group] for period in ["low", "high"])
        test = ttest_ind_from_stats(
            low["mean"],
            low["sd"],
            counts["low"][i],
            high["mean"],
            high["sd"],
            counts["high"][i],
            equal_var=False,
        )
        rows.append(
            dict(
                sza=group,
                low=low,
                high=high,
                change_percent=100 * (high["mean"] / low["mean"] - 1),
                welch_p=float(test.pvalue),
            )
        )
    workbook = pd.read_csv(INPUTS / "boundaries/radial_imf.csv")
    states = workbook[["inbound", "outbound"]]
    category = np.select(
        [states.eq("有").any(axis=1), states.eq("无").all(axis=1)], ["有", "无"], default="无法判定"
    )
    table = pd.crosstab(category, workbook.pcw_count.gt(0)).reindex(["有", "无", "无法判定"])
    np.testing.assert_array_equal(
        table.reindex(columns=[True, False]).to_numpy(), [[12, 20], [5, 9], [0, 4]]
    )
    assert len(workbook) == 50 and workbook.pcw_count.sum() == 55
    table.rename(columns={True: "pcw_recorded", False: "no_pcw_record"}).to_csv(
        BUILD / "radial_imf_counts.csv"
    )
    for period, coords in points.items():
        pd.DataFrame(coords, columns=["X_AVSO_Rv", "R_AVSO_Rv"]).to_csv(
            BUILD / f"shock_{period}_vector_points.csv", index=False
        )
    distances = {"subsolar": (1.396, 1.465), "terminator": (2.108, 2.263)}
    geometry = {
        key: dict(low=a, high=b, percent=(b / a - 1) * 100, km=(b - a) * 6052)
        for key, (a, b) in distances.items()
    }
    return dict(
        compression=rows,
        geometry=geometry,
        radial_examples=50,
        imb_identified=32,
        pcw_intervals=17,
        pcw_events=55,
    )
