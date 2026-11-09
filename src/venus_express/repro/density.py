"""Recompute the paper's dayside no-intersection inversion and its bootstrap."""

import numpy as np
import pandas as pd

from venus_express.physics.bow_shock import vso_to_aberrated_vso
from venus_express.sample_analysis.calculations.chamberlain import fit_chamberlain_direct
from venus_express.sample_analysis.calculations.paper_integrated_density import (
    spicav_hot_h_density_cm3,
)
from venus_express.sample_analysis.calculations.physics import field_line_bow_shock_intersection

from .settings import BUILD, INPUTS, write


def connection(row):
    position = np.array([getattr(row, f"mean_position_AVSO_{a}_Rv") for a in "xyz"])
    field = vso_to_aberrated_vso(np.array([getattr(row, f"mean_B0_VSO_{a}_nT") for a in "xyz"]))
    try:
        field_line_bow_shock_intersection(position, field)
    except ValueError as exc:
        if "does not intersect" not in str(exc):
            raise
        return "no_intersection"
    return "intersects"


def fit(events):
    return fit_chamberlain_direct(
        events.VSO_R_Rv.to_numpy(),
        events.n_H_neutral_cm3.to_numpy(),
        minimum_sample_count=3,
        density_space="log",
        include_satellites=True,
    )


def compute(raw=False):
    output = BUILD / "density"
    events = pd.read_csv(INPUTS / "manual/density.csv")
    states = [connection(row) for row in events.itertuples()]
    np.testing.assert_array_equal(states, events.B0ShockConnection)
    events = events[events.B0ShockConnection == "no_intersection"].copy()
    if raw:
        from .continuum import pickup_density
        from .raw import cache, require_days

        require_days(events.day)
        source = cache()
        recomputed = [pickup_density(source, row) for row in events.itertuples()]
        np.testing.assert_allclose(recomputed, events.n_Hplus_pickup_cm3, rtol=5e-6, atol=1e-12)
        events["n_Hplus_pickup_cm3"] = recomputed
    events["n_H_neutral_cm3"] = (
        events.paper_fgyro_hz
        * events.n_Hplus_pickup_cm3
        / (80 * events.paper_total_ionization_rate_s)
    )
    assert len(events) == 78
    return fit_with_bootstrap(events, output)


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
        ratio_to_spicav=float(result.reference_density_cm3 / spicav_hot_h_density_cm3(25000)),
        altitude_range_km=[float(events.alt_km.min()), float(events.alt_km.max())],
    )
    write(output / "summary.json", {"cases": {"no_intersection": summary}})
    return summary
