"""Reconstruct the archived bow-shock plots and tabulate frozen IMF annotations."""

import re
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from venus_express.repro.settings import INPUTS, BUILD


def bow_shock_curve(x0, L, eps, theta=np.linspace(-2.4, 2.4, 600)):
    rr = L / (1 + eps * np.cos(theta))
    x, y = x0 + rr * np.cos(theta), rr * np.sin(theta)
    keep = y >= 0
    return x[keep], y[keep]


def svg_of(pdf):
    out = BUILD / "boundary_sources" / pdf.with_suffix(".svg").name
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(out)], check=True, capture_output=True)
    return out.read_text()


def bow_shock_data():
    s = svg_of(INPUTS / "boundaries/source_compression.pdf")
    k = (252.595937 - 38.342461) / 4.5
    bars = {}
    for p in re.findall(r"<path ([^>]*)>", s):
        if 'fill="none"' in p and 'stroke="rgb(25%' in p:
            n = list(map(float, re.findall(r"-?\d+\.?\d*", re.search(r' d="([^"]+)"', p).group(1))))
            if abs(n[0] - n[2]) < 0.01:
                bars[round(n[0], 1)] = ((n[1] - 38.342461) / k, (n[3] - 38.342461) / k)
    order = sorted(bars)  # low 0-45, high 0-45, low 80-100, high 80-100 by x
    xs = {
        100.6: ("low", "0—45°"),
        178.5: ("high", "0—45°"),
        329.7: ("low", "80—100°"),
        407.6: ("high", "80—100°"),
    }
    comp = {xs[x]: dict(mean=(lo + hi) / 2, sd=(hi - lo) / 2) for x, (lo, hi) in bars.items()}
    b = svg_of(INPUTS / "boundaries/source_geometry.pdf")
    pts = {}
    for name, col in [
        ("low", "rgb(17.64679%, 42.352295%, 63.528442%)"),
        ("high", "rgb(69.802856%, 22.744751%, 22.744751%)"),
    ]:
        cs = []
        for p in re.findall(r"<path ([^>]*)>", b):
            if f'fill="{col}"' in p:
                n = list(
                    map(
                        float,
                        re.findall(
                            r"-?\d+\.?\d*(?:e-?\d+)?", re.search(r' d="([^"]+)"', p).group(1)
                        ),
                    )
                )
                cs.append(((min(n[0::2]) + max(n[0::2])) / 2, (min(n[1::2]) + max(n[1::2])) / 2))
        pts[name] = np.array(cs)
    disks = []
    for p in re.findall(r"<path ([^>]*)>", b):
        if 'fill="rgb(10.19' in p:
            n = list(
                map(
                    float,
                    re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", re.search(r' d="([^"]+)"', p).group(1)),
                )
            )
            if max(n[0::2]) - min(n[0::2]) > 15:
                disks.append((min(n[0::2]), max(n[0::2]), min(n[1::2]), max(n[1::2])))
    disks.sort()
    conv = {}
    for name, (x0, x1, y0, y1) in zip(["low", "high"], disks[:2]):
        sx, sy = x1 - x0, y1 - y0
        P = pts[name]
        conv[name] = np.c_[(x0 - P[:, 0]) / sx, (y1 - P[:, 1]) / sy]
    return comp, conv


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
