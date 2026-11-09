"""Year intercepts and distance-standardized wave energy for the Venus Express catalogue."""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, pearsonr, spearmanr, t
from venus_express.physics.bow_shock import BOW_SHOCK_SOLAR_MINIMUM, vso_to_aberrated_vso
from venus_express.repro.settings import ROOT, INPUTS, BUILD

YEARS = np.arange(2006, 2015)


def shock_x(rho):
    m = BOW_SHOCK_SOLAR_MINIMUM
    l, e = m.semi_latus_rectum_rv, m.eccentricity
    return m.focus_x_rv + (l * l - rho * rho) / (l * e + np.sqrt(l * l + (e * e - 1) * rho * rho))


def nearest_shock_distance(avso):
    """Signed shortest Euclidean distance to the axisymmetric reference surface.

    The nose provides an upper bound on the minimum distance, which bounds rho.
    Bracket every grid minimum and include the axis endpoint explicitly.
    """
    output = []
    for x, y, z in avso:
        rho = np.hypot(y, z)
        upper = rho + np.hypot(x - shock_x(0), rho) + 1e-8
        grid = np.linspace(0, upper, 2049)

        def objective(u):
            return (shock_x(u) - x) ** 2 + (u - rho) ** 2

        values = objective(grid)
        candidates = [(values[0], 0.0), (values[-1], upper)]
        for i in np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])) + 1:
            result = minimize_scalar(
                objective,
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-12},
            )
            assert result.success
            candidates.append((result.fun, result.x))
        minimum, closest_rho = min(candidates)
        assert minimum <= values.min() + 1e-9
        assert (
            abs(
                BOW_SHOCK_SOLAR_MINIMUM.signed_implicit_distance_rv(
                    shock_x(closest_rho), closest_rho, 0
                )
            )
            < 1e-10
        )
        sign = np.sign(BOW_SHOCK_SOLAR_MINIMUM.signed_implicit_distance_rv(x, y, z))
        output.append(sign * np.sqrt(minimum))
    return np.asarray(output)


def basis(values):
    """Natural cubic spline with four outcome-independent quantile knots."""
    knots = np.quantile(values, [0, 1 / 3, 2 / 3, 1])
    assert np.all(np.diff(knots) > 0)
    return CubicSpline(knots, np.eye(4), bc_type="natural")(values)[:, 1:], knots


def design(data, flexible=True):
    year = data.year.to_numpy()
    indicator = (year[:, None] == YEARS).astype(float)
    distances = [np.log(data.planet_distance_Rv.to_numpy()), data.shock_distance_Rv.to_numpy()]
    knots = []
    if flexible:
        pieces = []
        for value in distances:
            matrix, knot = basis(value)
            pieces.append(matrix)
            knots.append(knot.tolist())
        geometry = np.column_stack(pieces)
    else:
        geometry = np.column_stack(distances)
    # Centering makes the year intercept a standardized log-energy mean,
    # evaluated over the same pooled empirical distance distribution each year.
    geometry -= geometry.mean(axis=0)
    return np.column_stack([indicator, geometry]), knots


def fit(data, flexible=True):
    x, knots = design(data, flexible)
    y = np.log(data.wave_energy_pJ_m3.to_numpy())
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    assert rank == x.shape[1]
    return beta, x, y, knots


def correlation(sun, log_energy):
    regression = linregress(sun, log_energy)
    half = t.ppf(0.975, len(sun) - 2) * regression.stderr
    return dict(
        n_years=len(sun),
        pearson_r=float(pearsonr(sun, log_energy).statistic),
        pearson_p=float(pearsonr(sun, log_energy).pvalue),
        spearman_rho=float(spearmanr(sun, log_energy).statistic),
        slope_log_energy_per_sunspot=float(regression.slope),
        ratio_per_100_sunspots=float(np.exp(100 * regression.slope)),
        ratio_per_100_sunspots_ci95=np.exp(
            100 * np.array([regression.slope - half, regression.slope + half])
        ).tolist(),
    )


def compute(sun, events=None, output=None):
    frozen = events is None
    data = (
        pd.read_csv(INPUTS / "manual/wave_energy.csv", float_precision="round_trip")
        if frozen
        else events.reset_index(drop=True).copy()
    )
    if frozen:
        assert len(data) == 796 and data.plasma_region.eq("solar_wind").sum() == 446
    position = data[["VSO_X_Rv", "VSO_Y_Rv", "VSO_Z_Rv"]].to_numpy()
    data["planet_distance_Rv"] = np.linalg.norm(position, axis=1)
    data["shock_distance_Rv"] = nearest_shock_distance(vso_to_aberrated_vso(position))
    np.testing.assert_allclose(
        data.wave_energy_pJ_m3, data.integrated_band_psd_nT2 * 1e-6 / (8 * np.pi * 1e-7), rtol=1e-12
    )
    beta, x, y, _ = fit(data)
    groups = [
        [g.index.to_numpy() for _, g in data[data.year.eq(year)].groupby("day")] for year in YEARS
    ]
    rng = np.random.default_rng(20261028)
    boot = []
    for _ in range(1500):
        idx = np.concatenate(
            [np.concatenate([g[i] for i in rng.integers(len(g), size=len(g))]) for g in groups]
        )
        weights, _, rank, _ = np.linalg.lstsq(x[idx], y[idx], rcond=None)
        if rank == x.shape[1]:
            boot.append(weights[: len(YEARS)])
    assert len(boot) >= 0.99 * 1500
    lo, hi = np.exp(np.quantile(boot, [0.025, 0.975], axis=0))
    annual = pd.DataFrame(
        dict(
            year=YEARS,
            n=data.groupby("year").size().reindex(YEARS).to_numpy(),
            sunspot_number=sun,
            adjusted_geometric_mean_energy=np.exp(beta[: len(YEARS)]),
            adjusted_ci95_low=lo,
            adjusted_ci95_high=hi,
        )
    )
    if frozen:
        expected = pd.read_csv(ROOT / "reference/annual_energy.csv")
        pd.testing.assert_frame_equal(
            annual, expected[annual.columns], check_dtype=False, rtol=1e-7, atol=1e-10
        )
    out = output if output is not None else BUILD / "energy"
    out.mkdir(parents=True, exist_ok=True)
    annual.to_csv(out / "annual_energy.csv", index=False)

    support = np.ones(len(data), dtype=bool)
    for key in ["planet_distance_Rv", "shock_distance_Rv"]:
        years = data.groupby("year")[key]
        support &= data[key].between(years.min().max(), years.max().min())
    result = dict(
        events=len(data),
        crossing_confirmed=int(data.plasma_region.eq("solar_wind").sum()),
        adjusted=correlation(sun, beta[:9]),
        linear=correlation(sun, fit(data, False)[0][:9]),
    )
    for key, subset in [
        ("common", data[support]),
        ("crossing_only", data[data.plasma_region == "solar_wind"]),
    ]:
        if subset.year.nunique() == len(YEARS) and len(subset) >= 15:
            result[key] = correlation(sun, fit(subset)[0][:9])
    return result
