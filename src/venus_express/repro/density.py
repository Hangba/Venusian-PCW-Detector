"""Recompute the paper's dayside no-intersection inversion and its bootstrap."""

import numpy as np
import pandas as pd

from venus_express.sample_analysis.calculations.chamberlain import (
    chamberlain_density_cm3,
    fit_chamberlain_direct,
)
from venus_express.constants import R_VENUS

from .settings import write


def fit(events):
    return fit_chamberlain_direct(
        events.VSO_R_Rv.to_numpy(),
        events.n_H_neutral_cm3.to_numpy(),
        minimum_sample_count=3,
        density_space="log",
        include_satellites=True,
    )


def fit_with_bootstrap(events, output):
    output.mkdir(parents=True, exist_ok=True)
    events = events.assign(altitude_Rv=events.VSO_R_Rv - 1)
    events.to_csv(output / "exospheric_events.csv", index=False)
    result = fit(events)
    groups = [frame for _, frame in events.groupby("day", sort=False)]
    rng = np.random.default_rng(2026)
    draws = []
    for i in range(300):
        sample = pd.concat(
            [groups[k] for k in rng.integers(len(groups), size=len(groups))], ignore_index=True
        )
        if len(sample) < 3:
            continue
        trial = fit(sample)
        draws.append(
            dict(
                draw=i,
                n25=trial.reference_density_cm3,
                T=trial.temperature_k,
                n0=trial.n0_cm3,
                events=len(sample),
            )
        )
    bootstrap = pd.DataFrame(draws)
    bootstrap.to_csv(output / "no_intersection_bootstrap.csv", index=False)
    summary = dict(
        events=len(events),
        days=len(groups),
        bootstrap_draws=len(bootstrap),
        n25=result.reference_density_cm3,
        n25_95=bootstrap.n25.quantile([0.025, 0.975]).tolist(),
        T=result.temperature_k,
        T_95=bootstrap["T"].quantile([0.025, 0.975]).tolist(),
        n0=result.n0_cm3,
        log10_rmse=result.log10_rmse,
        log10_r_squared=result.log10_r_squared,
        altitude_range_km=[float(events.alt_km.min()), float(events.alt_km.max())],
    )
    lo_alt, hi_alt = summary["altitude_range_km"]
    h = np.unique(np.r_[np.geomspace(0.7 * R_VENUS, 12 * R_VENUS, 1000), 25000, lo_alt, hi_alt])
    rv = 1 + h / R_VENUS
    curves = [chamberlain_density_cm3(rv, row.n0, row.T) for row in bootstrap.itertuples()]
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    pd.DataFrame(dict(
        alt_km=h, n_fit=chamberlain_density_cm3(rv, result.n0_cm3, result.temperature_k),
        n_lo=lo, n_hi=hi, within_event_altitude_range=(h >= lo_alt) & (h <= hi_alt),
    )).to_csv(output / "square_12Rv_bootstrap_profile.csv", index=False)
    write(output / "summary.json", {"cases": {"no_intersection": summary}})
    return summary
