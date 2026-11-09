"""Chamberlain exosphere density profile fitting for a single suprathermal atomic-H population.

The density shape follows the Chamberlain spherical collisionless partition
function (ballistic, satellite and escaping populations), normalized to the fitted
exobase density. The population terms follow Cui Jun's Planetary Atmospheres II,
Lecture 3, PDF pp. 5-9. Production always defaults to sources 1+2+3a.
Explicit exclusion of source 2 is reserved for historical/control experiments.
The two fitted parameters are the neutral-H number density ``n0`` and
temperature ``T0`` at 250 km ABOVE THE SURFACE. All radius arguments are
planet-centred, so r0 = R_V + 250 km and altitude h=5 R_V means r=6 R_V.
This population convention must not be assumed identical to Weichbold's
unspecified Chamberlain implementation merely because the parameter names agree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammainc, gammaincc

from venus_express.constants import M_PROTON, R_VENUS

BOLTZMANN = 1.380649e-23
VENUS_GM_M3_S2 = 3.24859e14
DEFAULT_INCLUDE_SATELLITES = True
DEFAULT_EXOBASE_ALTITUDE_KM = 250.0
DEFAULT_REFERENCE_ALTITUDE_KM = 25_000.0
DEFAULT_ALTITUDE_BIN_WIDTH_KM = 5_000.0
DEFAULT_ALTITUDE_WEIGHT_ALPHA = 0.5
DEFAULT_ALTITUDE_COUNT_FLOOR = 5


def altitude_fit_weights(
    radius_rv: np.ndarray,
    *,
    bin_width_km: float = DEFAULT_ALTITUDE_BIN_WIDTH_KM,
    alpha: float = DEFAULT_ALTITUDE_WEIGHT_ALPHA,
    count_floor: int = DEFAULT_ALTITUDE_COUNT_FLOOR,
    bin_offset_km: float = 0.0,
    measurement_sigma_log: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build separate height-balance and measurement-precision fit weights.

    Every original event remains in the fit.  For event ``i`` in altitude bin
    ``b``, the representativeness factor is

        a_i = max(N_b, count_floor) ** (-alpha).

    If a positive log-density uncertainty is supplied, it contributes the
    independent inverse-variance factor ``1 / sigma_log**2``.  The combined
    weights are normalized to mean one, which does not change the optimum.
    """

    radius = np.asarray(radius_rv, dtype=np.float64).ravel()
    if radius.size == 0 or np.any(~np.isfinite(radius)):
        raise ValueError("Chamberlain radii must be a non-empty finite array")
    if not np.isfinite(bin_width_km) or bin_width_km <= 0:
        raise ValueError("Altitude bin width must be finite and positive")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Altitude-weight alpha must be in [0, 1]")
    if int(count_floor) != count_floor or count_floor < 1:
        raise ValueError("Altitude-bin count floor must be a positive integer")
    if not np.isfinite(bin_offset_km):
        raise ValueError("Altitude-bin offset must be finite")

    altitude_km = (radius - 1.0) * R_VENUS
    scaled_altitude = (altitude_km - float(bin_offset_km)) / float(bin_width_km)
    # Radius-to-altitude conversion can put an exact edge one ulp below its bin.
    nearest = np.rint(scaled_altitude)
    scaled_altitude = np.where(
        np.isclose(scaled_altitude, nearest, rtol=0, atol=1e-12), nearest, scaled_altitude
    )
    bin_id = np.floor(scaled_altitude).astype(np.int64)
    _, inverse, counts = np.unique(bin_id, return_inverse=True, return_counts=True)
    event_counts = counts[inverse]
    height_weight = np.maximum(event_counts.astype(np.float64), float(count_floor)) ** (
        -float(alpha)
    )

    if measurement_sigma_log is None:
        sigma_log = np.ones(radius.size, dtype=np.float64)
        measurement_weight = np.ones(radius.size, dtype=np.float64)
        measurement_source = "uniform_no_reliable_event_sigma"
    else:
        sigma_log = np.asarray(measurement_sigma_log, dtype=np.float64).ravel()
        if sigma_log.size != radius.size:
            raise ValueError("Measurement uncertainties must match radius length")
        if np.any(~np.isfinite(sigma_log)) or np.any(sigma_log <= 0.0):
            raise ValueError("Measurement uncertainties must be finite and positive")
        measurement_weight = sigma_log**-2
        measurement_source = "inverse_variance_log_density"

    combined = height_weight * measurement_weight
    combined /= float(np.mean(combined))
    return pd.DataFrame(
        {
            "altitude_km": altitude_km,
            "altitude_bin_id": bin_id,
            "altitude_bin_lower_km": bin_id * float(bin_width_km) + float(bin_offset_km),
            "altitude_bin_upper_km": (bin_id + 1) * float(bin_width_km) + float(bin_offset_km),
            "altitude_bin_count": event_counts,
            "altitude_representativeness_weight": height_weight,
            "measurement_sigma_log": sigma_log,
            "measurement_precision_weight": measurement_weight,
            "combined_fit_weight": combined,
            "measurement_weight_source": measurement_source,
        }
    )


def _chamberlain_partition(
    radius_m: np.ndarray,
    lambda0: float,
    exobase_radius_m: float,
    *,
    include_satellites: bool = DEFAULT_INCLUDE_SATELLITES,
) -> np.ndarray:
    """Return the selected Chamberlain density factor.

    This is the incomplete-gamma form of the 1963 Chamberlain solution.  The
    satellite population is included by default with full phase-space occupancy.
    This is an explicit production assumption: satellite trajectories do not
    intersect the exobase, so their population is not supplied solely by the
    collisionless lower-boundary distribution.
    """

    radius = np.asarray(radius_m, dtype=np.float64)
    ratio = exobase_radius_m / radius
    local_lambda = lambda0 * ratio
    psi = local_lambda**2 / (local_lambda + lambda0)
    angular_cutoff = np.sqrt(np.maximum(1.0 - ratio**2, 0.0)) * np.exp(-psi)

    # With P and Q the regularized lower and upper incomplete gamma
    # functions, respectively, these are Chamberlain's ballistic and escape
    # partition functions for a Maxwellian exobase boundary.
    ballistic = gammainc(1.5, local_lambda) - angular_cutoff * gammainc(
        1.5, np.maximum(local_lambda - psi, 0.0)
    )
    escaping = 0.5 * (
        gammaincc(1.5, local_lambda)
        - angular_cutoff * gammaincc(1.5, np.maximum(local_lambda - psi, 0.0))
    )
    satellite = (
        angular_cutoff * gammainc(1.5, np.maximum(local_lambda - psi, 0.0))
        if include_satellites
        else 0.0
    )
    barometric = np.exp(local_lambda - lambda0)
    return np.maximum(barometric * (ballistic + escaping + satellite), np.finfo(np.float64).tiny)


def chamberlain_density_shape(
    radius_rv: np.ndarray | float,
    temperature_k: float,
    *,
    exobase_altitude_km: float = DEFAULT_EXOBASE_ALTITUDE_KM,
    include_satellites: bool = DEFAULT_INCLUDE_SATELLITES,
) -> np.ndarray:
    """Return n(r)/n0 for H, retaining sources 1+2+3a by default.

    The lecture convention is n(r)=n_exo exp(lambda-lambda0) (zeta_1+zeta_2+zeta_3a).
    Here n0 is the actual retained-population density at the 250-km boundary:
    n0 = n_exo * (zeta_1+zeta_2+zeta_3a)(r0). Dividing by that boundary partition
    is a parameter convention, not another population or a reference fit.
    T is an effective temperature of this single suprathermal component.
    With ``include_satellites=True``, add source 2 to the numerator and boundary
    normalization. The same flag must be used when evaluating a fitted profile.
    """

    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Chamberlain temperature must be finite and positive")
    radius = np.asarray(radius_rv, dtype=np.float64) * R_VENUS * 1e3
    exobase_radius = (R_VENUS + float(exobase_altitude_km)) * 1e3
    if np.any(~np.isfinite(radius)) or np.any(radius < exobase_radius):
        raise ValueError("Chamberlain radii must be finite and at or above the exobase")
    lambda0 = VENUS_GM_M3_S2 * M_PROTON / (BOLTZMANN * temperature * exobase_radius)
    local_partition = _chamberlain_partition(
        radius, lambda0, exobase_radius, include_satellites=include_satellites
    )
    exobase_partition = float(
        _chamberlain_partition(
            np.asarray([exobase_radius]),
            lambda0,
            exobase_radius,
            include_satellites=include_satellites,
        )[0]
    )
    return local_partition / exobase_partition


def chamberlain_density_cm3(
    radius_rv: np.ndarray | float,
    n0_cm3: float,
    temperature_k: float,
    *,
    exobase_altitude_km: float = DEFAULT_EXOBASE_ALTITUDE_KM,
    include_satellites: bool = DEFAULT_INCLUDE_SATELLITES,
) -> np.ndarray:
    if not np.isfinite(n0_cm3) or n0_cm3 <= 0.0:
        raise ValueError("Chamberlain exobase density must be finite and positive")
    return float(n0_cm3) * chamberlain_density_shape(
        radius_rv,
        temperature_k,
        exobase_altitude_km=exobase_altitude_km,
        include_satellites=include_satellites,
    )


def _fit_once(
    radius_rv: np.ndarray,
    density_cm3: np.ndarray,
    *,
    sample_weights: np.ndarray | None,
    exobase_altitude_km: float,
    temperature_bounds_k: tuple[float, float],
    density_space: str = "log",
    reference_altitude_km: float = DEFAULT_REFERENCE_ALTITUDE_KM,
    include_satellites: bool = DEFAULT_INCLUDE_SATELLITES,
) -> tuple[float, float]:
    log_density = np.log(density_cm3)
    weights = np.ones_like(log_density) if sample_weights is None else sample_weights
    weight_sum = float(np.sum(weights))

    def normalization(shape):
        if density_space == "linear":
            return float(np.sum(weights * shape * density_cm3) / np.sum(weights * shape**2))
        return float(np.exp(np.sum(weights * (log_density - np.log(shape))) / weight_sum))

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        shape = chamberlain_density_shape(
            radius_rv,
            temperature,
            exobase_altitude_km=exobase_altitude_km,
            include_satellites=include_satellites,
        )
        shape = shape / chamberlain_density_shape(
            1.0 + reference_altitude_km / R_VENUS,
            temperature,
            exobase_altitude_km=exobase_altitude_km,
            include_satellites=include_satellites,
        )
        if density_space == "linear":
            residual = (density_cm3 - normalization(shape) * shape) / np.median(density_cm3)
            return float(np.sum(weights * residual**2) / weight_sum)
        log_shape = np.log(shape)
        log_reference_density = float(np.sum(weights * (log_density - log_shape)) / weight_sum)
        residual = log_density - (log_reference_density + log_shape)
        return float(np.sum(weights * residual**2) / weight_sum)

    lower, upper = temperature_bounds_k
    if not (0 < lower < upper):
        raise ValueError("Temperature bounds must be positive and increasing")
    # The partition model can have multiple minima. A single bounded search
    # assumes unimodality and can miss the better high-temperature solution.
    grid = np.linspace(np.log(lower), np.log(upper), 256)
    losses = np.asarray([objective(x) for x in grid])
    candidates = [(float(losses[0]), float(grid[0])), (float(losses[-1]), float(grid[-1]))]
    for i in range(1, len(grid) - 1):
        if losses[i] <= losses[i - 1] and losses[i] <= losses[i + 1]:
            result = minimize_scalar(
                objective,
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 300},
            )
            if result.success and np.isfinite(result.fun):
                candidates.append((float(result.fun), float(result.x)))
    loss, log_temperature = min(candidates)
    if not np.isfinite(loss):
        raise ValueError("Chamberlain temperature optimization failed")
    temperature = float(np.exp(log_temperature))
    shape = chamberlain_density_shape(
        radius_rv,
        temperature,
        exobase_altitude_km=exobase_altitude_km,
        include_satellites=include_satellites,
    )
    reference_shape = float(
        chamberlain_density_shape(
            1.0 + reference_altitude_km / R_VENUS,
            temperature,
            exobase_altitude_km=exobase_altitude_km,
            include_satellites=include_satellites,
        )
    )
    n0 = normalization(shape / reference_shape) / reference_shape
    return n0, temperature


@dataclass(frozen=True)
class ChamberlainFitResult:
    n0_cm3: float
    temperature_k: float
    log10_rmse: float
    log10_r_squared: float
    sample_count: int
    exobase_altitude_km: float
    profile: pd.DataFrame
    event_count: int = 0
    method: str = "event_log_density_least_squares"
    fit_samples: pd.DataFrame | None = None
    density_space: str = "log"
    linear_rmse_cm3: float = np.nan
    reference_altitude_km: float = DEFAULT_REFERENCE_ALTITUDE_KM
    reference_density_cm3: float = np.nan
    reference_within_data: bool = False
    parameter_geometry: dict | None = None
    include_satellites: bool = DEFAULT_INCLUDE_SATELLITES


def fit_chamberlain_direct(
    radius_rv: np.ndarray,
    density_cm3: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    exobase_altitude_km: float = DEFAULT_EXOBASE_ALTITUDE_KM,
    temperature_bounds_k: tuple[float, float] = (200.0, 50000.0),
    profile_points: int = 240,
    minimum_sample_count: int = 8,
    density_space: str = "log",
    reference_altitude_km: float = DEFAULT_REFERENCE_ALTITUDE_KM,
    include_satellites: bool = DEFAULT_INCLUDE_SATELLITES,
) -> ChamberlainFitResult:
    """Fit supplied samples directly (legacy event-level fitting and numerical core)."""

    radius = np.asarray(radius_rv, dtype=np.float64).ravel()
    density = np.asarray(density_cm3, dtype=np.float64).ravel()
    if density_space not in ("log", "linear"):
        raise ValueError("density_space must be 'log' or 'linear'")
    if not np.isfinite(reference_altitude_km) or reference_altitude_km < exobase_altitude_km:
        raise ValueError("Reference altitude must be finite and at or above the exobase")
    if radius.size != density.size:
        raise ValueError("Radius and density arrays must have equal lengths")
    if minimum_sample_count < 3 or int(minimum_sample_count) != minimum_sample_count:
        raise ValueError("At least 3 samples are required for a two-parameter fit")
    exobase_radius_rv = (R_VENUS + exobase_altitude_km) / R_VENUS
    valid = (
        np.isfinite(radius) & np.isfinite(density) & (radius >= exobase_radius_rv) & (density > 0.0)
    )
    radius, density = radius[valid], density[valid]
    weights = None
    if sample_weights is not None:
        raw_weights = np.asarray(sample_weights, dtype=np.float64).ravel()
        if raw_weights.size != valid.size:
            raise ValueError("Sample weights must match radius and density lengths")
        weights = raw_weights[valid]
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("Sample weights must be finite and positive")
        if not np.any(weights > 0.0):
            raise ValueError("At least one sample weight must be positive")
    if radius.size < minimum_sample_count:
        raise ValueError(
            f"Chamberlain fitting requires at least {minimum_sample_count} valid density samples"
        )
    n0, temperature = _fit_once(
        radius,
        density,
        sample_weights=weights,
        exobase_altitude_km=exobase_altitude_km,
        include_satellites=include_satellites,
        temperature_bounds_k=temperature_bounds_k,
        density_space=density_space,
        reference_altitude_km=reference_altitude_km,
    )

    predicted = chamberlain_density_cm3(
        radius,
        n0,
        temperature,
        exobase_altitude_km=exobase_altitude_km,
        include_satellites=include_satellites,
    )
    reference_radius = 1.0 + reference_altitude_km / R_VENUS
    reference_density = float(
        chamberlain_density_cm3(
            reference_radius,
            n0,
            temperature,
            exobase_altitude_km=exobase_altitude_km,
            include_satellites=include_satellites,
        )
    )

    def temperature_derivative(r):
        return (
            np.log(
                chamberlain_density_shape(
                    r,
                    temperature * np.exp(1e-4),
                    exobase_altitude_km=exobase_altitude_km,
                    include_satellites=include_satellites,
                )
            )
            - np.log(
                chamberlain_density_shape(
                    r,
                    temperature * np.exp(-1e-4),
                    exobase_altitude_km=exobase_altitude_km,
                    include_satellites=include_satellites,
                )
            )
        ) / 2e-4

    derivative = temperature_derivative(radius)
    scale = np.sqrt(np.ones(radius.size) if weights is None else weights)
    if density_space == "linear":
        scale = scale * predicted
    geometry = {"coordinates": "log_amplitude_and_log_temperature", "objective": density_space}
    for name, d in (
        ("n0", derivative),
        ("nref", derivative - temperature_derivative(reference_radius)),
    ):
        jacobian = scale[:, None] * np.column_stack((np.ones(radius.size), d))
        norm = float(np.linalg.norm(jacobian[:, 0]) * np.linalg.norm(jacobian[:, 1]))
        geometry[f"jacobian_condition_{name}"] = float(np.linalg.cond(jacobian))
        geometry[f"local_parameter_correlation_{name}_temperature"] = (
            -float(jacobian[:, 0] @ jacobian[:, 1]) / norm if norm > 0 else np.nan
        )
    observed_log = np.log10(density)
    predicted_log = np.log10(predicted)
    residual = observed_log - predicted_log
    eval_weights = np.ones_like(residual) if weights is None else weights
    eval_weight_sum = float(np.sum(eval_weights))
    rmse = float(np.sqrt(np.sum(eval_weights * residual**2) / eval_weight_sum))
    observed_mean = float(np.sum(eval_weights * observed_log) / eval_weight_sum)
    denominator = float(np.sum(eval_weights * (observed_log - observed_mean) ** 2))
    r_squared = (
        1.0 - float(np.sum(eval_weights * residual**2)) / denominator
        if denominator > 0.0
        else np.nan
    )

    grid = np.linspace(exobase_radius_rv, float(np.max(radius)), int(profile_points))
    profile = pd.DataFrame(
        {
            "radius_Rv": grid,
            "altitude_Rv": grid - 1.0,
            "fitted_density_cm3": chamberlain_density_cm3(
                grid,
                n0,
                temperature,
                exobase_altitude_km=exobase_altitude_km,
                include_satellites=include_satellites,
            ),
        }
    )
    return ChamberlainFitResult(
        n0_cm3=n0,
        temperature_k=temperature,
        log10_rmse=rmse,
        log10_r_squared=r_squared,
        sample_count=int(radius.size),
        exobase_altitude_km=float(exobase_altitude_km),
        profile=profile,
        event_count=int(radius.size),
        method=f"event_{density_space}_density_least_squares",
        density_space=density_space,
        linear_rmse_cm3=float(
            np.sqrt(np.average((density - predicted) ** 2, weights=eval_weights))
        ),
        reference_altitude_km=float(reference_altitude_km),
        reference_density_cm3=reference_density,
        reference_within_data=bool(np.min(radius) <= reference_radius <= np.max(radius)),
        parameter_geometry=geometry,
        include_satellites=include_satellites,
    )


def altitude_bin_medians(
    radius_rv: np.ndarray,
    density_cm3: np.ndarray,
    *,
    bin_width_km: float = DEFAULT_ALTITUDE_BIN_WIDTH_KM,
    bin_offset_km: float = 0.0,
    exobase_altitude_km: float = DEFAULT_EXOBASE_ALTITUDE_KM,
) -> pd.DataFrame:
    """One linear-density median and altitude median per occupied [lower, upper) bin.

    Invalid densities and radii below the exobase are removed before grouping.
    Empty bins are omitted; all other bins, including singletons, have equal
    fit weight regardless of their event count. Density is aggregated before log.
    Standard deviations describe within-bin event scatter (ddof=1), not the
    uncertainty of the median. Singletons retain NaN standard deviations.
    """
    radius = np.asarray(radius_rv, dtype=np.float64).ravel()
    density = np.asarray(density_cm3, dtype=np.float64).ravel()
    if radius.size != density.size:
        raise ValueError("Radius and density arrays must have equal lengths")
    valid = (
        np.isfinite(radius)
        & np.isfinite(density)
        & (density > 0)
        & (radius >= 1.0 + exobase_altitude_km / R_VENUS)
    )
    events = altitude_fit_weights(
        radius[valid], bin_width_km=bin_width_km, bin_offset_km=bin_offset_km, alpha=0.0
    )
    events["density_cm3"] = density[valid]
    events["log10_density"] = np.log10(density[valid])
    groups = events.groupby("altitude_bin_id", sort=True, as_index=False).agg(
        altitude_bin_lower_km=("altitude_bin_lower_km", "first"),
        altitude_bin_upper_km=("altitude_bin_upper_km", "first"),
        event_count=("density_cm3", "size"),
        altitude_km=("altitude_km", "median"),
        density_cm3=("density_cm3", "median"),
        altitude_std_km=("altitude_km", "std"),
        density_std_cm3=("density_cm3", "std"),
        density_log10_std=("log10_density", "std"),
    )
    groups["density_logsd_lower_cm3"] = groups.density_cm3 / 10**groups.density_log10_std
    groups["density_logsd_upper_cm3"] = groups.density_cm3 * 10**groups.density_log10_std
    groups["radius_Rv"] = 1.0 + groups.altitude_km / R_VENUS
    groups["fit_weight"] = 1.0
    return groups


def fit_chamberlain_altitude_medians(
    radius_rv: np.ndarray,
    density_cm3: np.ndarray,
    *,
    bin_width_km: float = DEFAULT_ALTITUDE_BIN_WIDTH_KM,
    bin_offset_km: float = 0.0,
    exobase_altitude_km: float = DEFAULT_EXOBASE_ALTITUDE_KM,
    temperature_bounds_k: tuple[float, float] = (200.0, 50000.0),
    profile_points: int = 240,
    density_space: str = "log",
    reference_altitude_km: float = DEFAULT_REFERENCE_ALTITUDE_KM,
    include_satellites: bool = DEFAULT_INCLUDE_SATELLITES,
) -> ChamberlainFitResult:
    """Fit equal-weight altitude-bin medians in the requested density space.

    ``sample_count`` and fit metrics describe the median points; ``event_count``
    records how many valid original events contributed to those points.
    """
    groups = altitude_bin_medians(
        radius_rv,
        density_cm3,
        bin_width_km=bin_width_km,
        bin_offset_km=bin_offset_km,
        exobase_altitude_km=exobase_altitude_km,
    )
    if len(groups) < 3:
        raise ValueError("Chamberlain median fitting requires at least 3 occupied altitude bins")
    fit = fit_chamberlain_direct(
        groups.radius_Rv,
        groups.density_cm3,
        exobase_altitude_km=exobase_altitude_km,
        include_satellites=include_satellites,
        temperature_bounds_k=temperature_bounds_k,
        profile_points=profile_points,
        minimum_sample_count=3,
        density_space=density_space,
        reference_altitude_km=reference_altitude_km,
    )
    return replace(
        fit,
        event_count=int(groups.event_count.sum()),
        method=f"equal_weight_altitude_bin_median_{density_space}_density_least_squares",
        fit_samples=groups,
    )
