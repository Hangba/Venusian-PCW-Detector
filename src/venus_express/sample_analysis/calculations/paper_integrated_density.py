"""Paper-style ICW spectral integration and pickup H+ density inversion.

This module uses the user-defined energy efficiency and Weichbold neutral-density equation.
The user-defined transverse PSD mean and integration band are retained. The perpendicular
magnetic PSD is integrated around the local proton gyrofrequency, converted to
pickup-proton density, and then to neutral H. The energy-transfer efficiency
``phi`` and the SPICAV-calibrated normalization ``Phi`` are separate quantities.
The finite-growth-time helper remains available for historical calculations.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from venus_express.cache import ICWFeatureCache
from venus_express.constants import M_PROTON, R_VENUS

MU0 = 4.0 * np.pi * 1e-7
DEFAULT_PHOTOIONIZATION_RATE_S = 2.35e-7
DEFAULT_N_GYRO_PERIODS = 100.0
# Dayside log-L2 SPICAV calibration, 311 band-energy-filtered catalogue events, phi=0.30,
# PSD=(P1+P2)/2, original dynamic band. See result/weichbold_current/summary.json.
DEFAULT_WEICHBOLD_PHI = 4340.927226577496
E_CHARGE = 1.602176634e-19
LINDSAY_H_H_ENERGY_RANGE_KEV = (0.005, 250.0)

# Chaufray et al. (2012), Icarus 217, Table 3: average dayside hot-H
# density profile inferred from VEX/SPICAV Lyman-alpha observations.  The
# profile is directly constrained by observations around 2,000-7,000 km and
# tabulated/extrapolated over the wider altitude range used here.
SPICAV_HOT_H_ALTITUDE_KM = np.asarray(
    [
        250,
        350,
        500,
        750,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        7500,
        10000,
        15000,
        20000,
        30000,
        40000,
        50000,
        75000,
        100000,
    ],
    dtype=np.float64,
)
SPICAV_HOT_H_DENSITY_CM3 = np.asarray(
    [
        2500,
        2100,
        1700,
        1200,
        900,
        510,
        310,
        210,
        140,
        78,
        47,
        19,
        10,
        4.1,
        2.2,
        0.9,
        0.5,
        0.3,
        0.13,
        0.07,
    ],
    dtype=np.float64,
)
# Chaufray et al. (2012), Table 3: cold/thermal H profile.
SPICAV_COLD_H_DENSITY_CM3 = np.asarray(
    [
        121700,
        84500,
        50000,
        22000,
        10200,
        2600,
        790,
        270,
        110,
        22,
        6,
        0.5,
        0.1,
        0.01,
        2.3e-3,
        3.6e-4,
        1.1e-4,
        4.5e-5,
        1.1e-5,
        4.3e-6,
    ],
    dtype=np.float64,
)
SPICAV_DIRECT_CONSTRAINT_ALTITUDE_KM = (2000.0, 7000.0)


def spicav_hot_h_density_cm3(altitude_km: np.ndarray | float) -> np.ndarray:
    """Interpolate Chaufray et al. (2012) Table 3 in log-log space."""

    altitude = np.asarray(altitude_km, dtype=np.float64)
    result = np.full(altitude.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(altitude)
        & (altitude >= SPICAV_HOT_H_ALTITUDE_KM[0])
        & (altitude <= SPICAV_HOT_H_ALTITUDE_KM[-1])
    )
    if np.any(valid):
        result[valid] = 10.0 ** np.interp(
            np.log10(altitude[valid]),
            np.log10(SPICAV_HOT_H_ALTITUDE_KM),
            np.log10(SPICAV_HOT_H_DENSITY_CM3),
        )
    return result


def compare_with_spicav(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach reference diagnostics without changing any inverted density."""
    result = events.copy()
    altitude = (result["VSO_R_Rv"].to_numpy(dtype=float) - 1) * R_VENUS
    reference = spicav_hot_h_density_cm3(altitude)
    density = pd.to_numeric(
        result.get("n_H_neutral_cm3", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = (
        result.paper_density_status.eq("ok").to_numpy()
        & np.isfinite(density)
        & (density > 0)
        & np.isfinite(reference)
        & (reference > 0)
    )
    residual = np.full(len(result), np.nan)
    residual[valid] = np.log10(density[valid] / reference[valid])
    result["paper_spicav_altitude_km"] = altitude
    result["paper_spicav_hot_h_reference_cm3"] = reference
    result["paper_spicav_log10_residual"] = residual
    low, high = SPICAV_DIRECT_CONSTRAINT_ALTITUDE_KM
    diagnostics = {
        "n_gyro_periods": None if "paper_growth_Phi" in result else DEFAULT_N_GYRO_PERIODS,
        "calibration_applied": "paper_growth_Phi" in result,
        "growth_Phi": (
            float(result.paper_growth_Phi.dropna().iloc[0])
            if "paper_growth_Phi" in result and result.paper_growth_Phi.notna().any()
            else None
        ),
        "spicav_comparison_count": int(valid.sum()),
        "spicav_direct_overlap_count": int((valid & (altitude >= low) & (altitude <= high)).sum()),
        "spicav_log10_rmse": float(np.sqrt(np.mean(residual[valid] ** 2))) if valid.any() else None,
        "spicav_altitude_min_km": float(altitude[valid].min()) if valid.any() else None,
        "spicav_altitude_max_km": float(altitude[valid].max()) if valid.any() else None,
    }
    return result, diagnostics


def _one_sided_hann_psd(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-sided, variance-preserving Hann-window PSD."""

    values = np.asarray(signal, dtype=np.float64).ravel()
    if values.size < 16 or not np.all(np.isfinite(values)):
        raise ValueError("PSD segment must contain at least 16 finite samples")
    values = values - float(np.mean(values))
    window = np.hanning(values.size)
    window_power = float(np.sum(window * window))
    if window_power <= 0.0:
        raise ValueError("invalid Hann window normalization")
    spectrum = np.fft.rfft(values * window)
    psd = (np.abs(spectrum) ** 2) / (float(fs_hz) * window_power)
    if values.size % 2 == 0:
        if psd.size > 2:
            psd[1:-1] *= 2.0
    elif psd.size > 1:
        psd[1:] *= 2.0
    return np.fft.rfftfreq(values.size, d=1.0 / float(fs_hz)), psd


def paper_perpendicular_psd(
    bfac_x_nt: np.ndarray,
    bfac_y_nt: np.ndarray,
    *,
    fs_hz: float = 1.0,
    full_event: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Average the paper's three 50%-overlapped subinterval PSD estimates.

    The full event is split into three equal-length-overlap placements. For a
    240-s paper window this yields three 120-s segments beginning at 0, 60 and
    120 s. Average the two orthogonal component PSDs, then average across the three
    temporal estimates. With full_event=True, use one full-length Hann segment.
    This convention is half the total transverse variance.
    """

    bx = np.asarray(bfac_x_nt, dtype=np.float64).ravel()
    by = np.asarray(bfac_y_nt, dtype=np.float64).ravel()
    n = min(bx.size, by.size)
    if n < 32:
        raise ValueError("event needs at least 32 magnetic-field samples")
    bx, by = bx[:n], by[:n]
    finite = np.isfinite(bx) & np.isfinite(by)
    if int(np.count_nonzero(finite)) < n:
        indices = np.arange(n)
        good = indices[finite]
        if good.size < 32:
            raise ValueError("event has too few finite FAC samples")
        bx = np.interp(indices, good, bx[finite])
        by = np.interp(indices, good, by[finite])

    segment_length = n if full_event else n // 2
    if segment_length < 16:
        raise ValueError("PSD subinterval is too short")
    offsets = (0,) if full_event else (0, (n - segment_length) // 2, n - segment_length)
    estimates = []
    frequency = None
    for offset in offsets:
        stop = offset + segment_length
        freq_x, psd_x = _one_sided_hann_psd(bx[offset:stop], fs_hz)
        freq_y, psd_y = _one_sided_hann_psd(by[offset:stop], fs_hz)
        if not np.array_equal(freq_x, freq_y):
            raise ValueError("FAC PSD frequency axes do not match")
        frequency = freq_x
        estimates.append((psd_x + psd_y) / 2.0)
    return np.asarray(frequency), np.mean(estimates, axis=0), segment_length


def _resample_uniform_cadence(
    times_s: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    *,
    fs_hz: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], dict[str, float | int]]:
    """Linearly resample an event onto the uniform cadence required by the FFT."""

    times = np.asarray(times_s, dtype=np.float64).ravel()
    if times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("event timestamps must contain at least two finite samples")
    order = np.argsort(times, kind="stable")
    times = times[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    if unique_times.size < 2:
        raise ValueError("event timestamps must contain at least two unique samples")
    cadence = np.diff(unique_times)
    dt = 1.0 / float(fs_hz)
    target = np.arange(unique_times[0], unique_times[-1] + 0.5 * dt, dt)
    interpolated = []
    for values in arrays:
        values = np.asarray(values, dtype=np.float64).ravel()[order][unique_indices]
        finite = np.isfinite(values)
        if int(np.count_nonzero(finite)) < 2:
            raise ValueError("event has too few finite samples for interpolation")
        interpolated.append(np.interp(target, unique_times[finite], values[finite]))
    return (
        target,
        tuple(interpolated),
        {
            "paper_gap_count": int(np.count_nonzero(cadence > 1.5 * dt)),
            "paper_missing_seconds": float(np.maximum(cadence - dt, 0.0).sum()),
            "paper_max_gap_s": float(np.max(cadence)),
            "paper_nonuniform_resampled": 1,
            "paper_original_sample_count": int(times.size),
            "paper_resampled_sample_count": int(target.size),
        },
    )


def integrate_paper_icw_band(
    bfac_x_nt: np.ndarray,
    bfac_y_nt: np.ndarray,
    fcp_curve_hz: np.ndarray,
    *,
    phi: float,
    fs_hz: float = 1.0,
    broaden_by_fcp_std: bool = True,
    sample_times_s: np.ndarray | None = None,
    resample_nonuniform: bool = True,
) -> dict[str, float | int]:
    """Integrate P_perp over the frequency interval defined in the paper."""

    if not (0.0 < float(phi) <= 1.0):
        raise ValueError("phi must be in (0, 1]")
    bx = np.asarray(bfac_x_nt, dtype=np.float64).ravel()
    by = np.asarray(bfac_y_nt, dtype=np.float64).ravel()
    fcp = np.asarray(fcp_curve_hz, dtype=np.float64).ravel()
    if not (bx.size == by.size == fcp.size):
        raise ValueError("magnetic samples and gyrofrequency samples must have equal lengths")
    original_count = int(bx.size)
    cadence_diagnostics: dict[str, float | int] = {
        "paper_gap_count": 0,
        "paper_missing_seconds": 0.0,
        "paper_max_gap_s": 1.0 / float(fs_hz),
        "paper_nonuniform_resampled": 0,
        "paper_original_sample_count": original_count,
        "paper_resampled_sample_count": original_count,
    }
    if sample_times_s is not None:
        times = np.asarray(sample_times_s, dtype=float).ravel()
        if len(times) != original_count:
            raise ValueError("Spectrum timestamps and magnetic samples must have equal lengths")
        cadence = np.diff(times)
        if not np.all(np.isfinite(times)) or not len(cadence):
            raise ValueError("Spectrum timestamps must be finite and contain multiple samples")
        if not np.allclose(cadence, 1.0 / fs_hz, rtol=0, atol=0.02 / fs_hz):
            if not resample_nonuniform:
                raise ValueError("Nonuniform or gapped samples cannot use the uniform-cadence PSD")
            _, (bx, by, fcp), cadence_diagnostics = _resample_uniform_cadence(
                times,
                (bx, by, fcp),
                fs_hz=fs_hz,
            )
    valid_fcp = fcp[np.isfinite(fcp) & (fcp > 0.0)]
    if valid_fcp.size < 2:
        raise ValueError("event has no valid local proton gyrofrequency")

    frequency, p_perp, segment_length = paper_perpendicular_psd(bx, by, fs_hz=fs_hz)
    fgyro = float(np.mean(valid_fcp))
    delta_fgyro = float(np.std(valid_fcp, ddof=0))
    broadening = delta_fgyro if broaden_by_fcp_std else 0.0
    band_low = max(0.0, 0.8 * (fgyro - broadening))
    band_high = 1.2 * (fgyro + broadening)
    band_mask = (frequency >= band_low) & (frequency <= band_high) & (frequency > 0.0)
    if not np.any(band_mask):
        raise ValueError("paper gyrofrequency band contains no PSD bins")
    if frequency.size < 2:
        raise ValueError("PSD frequency axis is too short")
    df_hz = float(frequency[1] - frequency[0])
    integrated_psd_nt2 = float(np.sum(p_perp[band_mask]) * df_hz)
    if not np.isfinite(integrated_psd_nt2) or integrated_psd_nt2 <= 0.0:
        raise ValueError("integrated perpendicular PSD is not positive")

    band_indices = np.flatnonzero(band_mask)
    peak_index = int(band_indices[int(np.argmax(p_perp[band_mask]))])
    wave_energy_j_m3 = integrated_psd_nt2 * 1e-18 / (2.0 * MU0)
    # User definition phi * E_free = E_ICW; preserve the PSD mean/band.
    # The PSD integral is magnetic wave energy, not ring-beam kinetic energy.
    e_free_j_m3 = wave_energy_j_m3 / float(phi)
    return {
        "paper_sample_count": original_count,
        "paper_segment_length": int(segment_length),
        "paper_segment_count": 3,
        "paper_frequency_resolution_hz": df_hz,
        "paper_fgyro_hz": fgyro,
        "paper_delta_fgyro_hz": delta_fgyro,
        "paper_band_broadened_by_fcp_std": int(bool(broaden_by_fcp_std)),
        "paper_band_low_hz": band_low,
        "paper_band_high_hz": band_high,
        "paper_band_bin_count": int(np.count_nonzero(band_mask)),
        "paper_peak_frequency_hz": float(frequency[peak_index]),
        "paper_peak_psd_nT2_per_Hz": float(p_perp[peak_index]),
        "paper_integrated_psd_nT2": integrated_psd_nt2,
        "paper_wave_energy_J_m3": wave_energy_j_m3,
        "paper_E_free_J_m3": e_free_j_m3,
        "paper_energy_convention": "User definition phi*E_free=E_ICW; E_free=W_B/phi; PSD=(P1+P2)/2; band=scaled_edges_v1",
        **cadence_diagnostics,
    }


def proton_projectile_energy_kev(v_sw_km_s: float) -> float:
    """Return the laboratory projectile energy of a solar-wind proton in keV."""

    velocity = float(v_sw_km_s)
    if not np.isfinite(velocity) or velocity <= 0.0:
        raise ValueError("solar-wind speed must be finite and positive")
    energy_j = 0.5 * M_PROTON * (velocity * 1e3) ** 2
    return energy_j / (E_CHARGE * 1e3)


def lindsay_stebbings_h_h_cross_section_cm2(v_sw_km_s: float) -> tuple[float, float]:
    """Return ``(projectile_energy_keV, sigma_CX_cm2)`` for H+ + H.

    Lindsay & Stebbings (2005), Table 1:

    sigma / 1e-16 cm2 = (4.15 - 0.531 ln E)^2
                         * (1 - exp(-67.3 / E))^4.5,

    where E is the proton projectile energy in keV.
    """

    energy_kev = proton_projectile_energy_kev(v_sw_km_s)
    low, high = LINDSAY_H_H_ENERGY_RANGE_KEV
    if not low <= energy_kev <= high:
        raise ValueError(
            f"proton energy {energy_kev:.6g} keV is outside the Lindsay-Stebbings "
            f"H+-H fit range [{low}, {high}] keV"
        )
    sigma_units = (4.15 - 0.531 * np.log(energy_kev)) ** 2
    sigma_units *= (1.0 - np.exp(-67.3 / energy_kev)) ** 4.5
    sigma_cm2 = float(sigma_units * 1e-16)
    if not np.isfinite(sigma_cm2) or sigma_cm2 <= 0.0:
        raise ValueError("Lindsay-Stebbings cross section is not positive")
    return energy_kev, sigma_cm2


def neutral_h_from_pickup_density(
    pickup_density_m3: float,
    fgyro_hz: float,
    *,
    n_gyro_periods: float = DEFAULT_N_GYRO_PERIODS,
    photoionization_rate_s: float,
    n_sw_cm3: float,
    v_sw_km_s: float,
    growth_phi: float | None = None,
) -> dict[str, float | str]:
    """Convert pickup H+ density to neutral H using a finite growth time.

    The balance is ``n_pickup ~= n_H * nu_total * tau`` with
    ``tau = n_gyro_periods / fgyro_hz`` because ``fgyro_hz`` is cycles per
    second. Therefore the conversion is ``n_H = fgyro*n_pickup /
    (n_gyro_periods*nu_total)`` and contains no extra ``2*pi``. The optional
    ``growth_phi`` keyword is retained only as a backwards-compatible alias for
    the old parameter name; it denotes a gyroperiod count here, not an energy
    efficiency.
    """

    if growth_phi is not None:
        if n_gyro_periods != DEFAULT_N_GYRO_PERIODS:
            raise ValueError("set either n_gyro_periods or growth_phi, not both")
        n_gyro_periods = float(growth_phi)

    parameters = (
        pickup_density_m3,
        fgyro_hz,
        n_gyro_periods,
        photoionization_rate_s,
        n_sw_cm3,
        v_sw_km_s,
    )
    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("Eq. (6) parameters must be finite")
    if pickup_density_m3 <= 0.0 or fgyro_hz <= 0.0 or n_gyro_periods <= 0.0:
        raise ValueError("pickup density, gyrofrequency and gyroperiod count must be positive")
    if photoionization_rate_s < 0.0 or n_sw_cm3 <= 0.0 or v_sw_km_s <= 0.0:
        raise ValueError("ionization and solar-wind parameters are invalid")
    collision_energy_kev, charge_exchange_cross_section_cm2 = (
        lindsay_stebbings_h_h_cross_section_cm2(v_sw_km_s)
    )
    charge_exchange_rate_s = (
        float(n_sw_cm3) * float(v_sw_km_s) * 1e5 * float(charge_exchange_cross_section_cm2)
    )
    total_ionization_rate_s = float(photoionization_rate_s) + charge_exchange_rate_s
    if total_ionization_rate_s <= 0.0:
        raise ValueError("total ionization rate must be positive")
    conversion_factor = float(fgyro_hz) / (float(n_gyro_periods) * total_ionization_rate_s)
    neutral_density_m3 = float(pickup_density_m3) * conversion_factor
    return {
        "paper_photoionization_rate_s": float(photoionization_rate_s),
        "paper_collision_energy_keV": collision_energy_kev,
        "paper_charge_exchange_cross_section_cm2": charge_exchange_cross_section_cm2,
        "paper_charge_exchange_rate_s": charge_exchange_rate_s,
        "paper_total_ionization_rate_s": total_ionization_rate_s,
        "paper_n_gyro_periods": float(n_gyro_periods),
        "paper_effective_accumulation_time_s": float(n_gyro_periods) / float(fgyro_hz),
        "paper_frequency_convention": "f_cp in Hz; tau=N_gyro/f_cp; no 2pi",
        "paper_eq6_conversion_factor": conversion_factor,
        "n_H_neutral_m3": neutral_density_m3,
        "n_H_neutral_cm3": neutral_density_m3 / 1e6,
    }


def apply_weichbold_neutral_conversion(
    events: pd.DataFrame,
    *,
    growth_phi: float | None = DEFAULT_WEICHBOLD_PHI,
) -> pd.DataFrame:
    """Apply Eq. (6) to already inverted H+; fit dayside Phi only if None.

    A fitted day normalization is always shared with nightside. Recalibration
    requires dayside solar-wind events and a fixed external SPICAV hot-H profile.
    """
    result = events.copy()
    unit = (
        2
        * np.pi
        * result.paper_fgyro_hz
        * result.n_Hplus_pickup_cm3
        / result.paper_total_ionization_rate_s
    )
    if growth_phi is None:
        reference = spicav_hot_h_density_cm3((result.VSO_R_Rv - 1) * R_VENUS)
        valid = (
            result.paper_density_status.eq("ok")
            & result.solar_zenith_angle_deg.lt(90)
            & np.isfinite(unit)
            & (unit > 0)
            & np.isfinite(reference)
            & (reference > 0)
        )
        if "plasma_region" in result:
            valid &= result.plasma_region.eq("solar_wind")
        if not valid.any():
            raise ValueError("Fitting Phi requires positive dayside solar-wind/SPICAV pairs")
        growth_phi = float(10 ** np.mean(np.log10(unit[valid] / reference[valid])))
    if not np.isfinite(growth_phi) or growth_phi <= 0:
        raise ValueError("Weichbold Phi must be finite and positive")
    result["paper_growth_Phi"] = float(growth_phi)
    result["paper_n_gyro_periods"] = np.nan
    result["paper_effective_accumulation_time_s"] = np.nan
    result["paper_frequency_convention"] = "f_cp in Hz; n_H=2pi*f_cp*n_pickup/(Phi*nu)"
    result["paper_eq6_conversion_factor"] = (
        2 * np.pi * result.paper_fgyro_hz / (growth_phi * result.paper_total_ionization_rate_s)
    )
    result["n_H_neutral_cm3"] = unit / growth_phi
    result["n_H_neutral_m3"] = result.n_H_neutral_cm3 * 1e6
    result, diagnostics = compare_with_spicav(result)
    result.attrs["spicav_comparison"] = diagnostics
    return result


def calculate_paper_integrated_density(
    events: pd.DataFrame,
    *,
    data_dir: str | Path,
    cache_dir: str | Path,
    phi: float,
    n_sw_cm3: float,
    v_sw_km_s: float,
    photoionization_rate_s: float = DEFAULT_PHOTOIONIZATION_RATE_S,
    broaden_by_fcp_std: bool = True,
    growth_phi: float = DEFAULT_WEICHBOLD_PHI,
    progress_callback: Callable[[int, int, int], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Append paper-style integrated energy and pickup H+ density columns."""

    required = {
        "start",
        "end",
        "B_bg_mean_nT",
        "injection_speed_km_s",
        "injection_angle_deg",
        "VSO_R_Rv",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError("Missing event context columns: " + ", ".join(missing))
    if n_sw_cm3 <= 0.0:
        raise ValueError("solar-wind density must be positive")

    result = events.copy()
    metric_rows: list[dict[str, object]] = []
    cache = ICWFeatureCache(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        num_freqs=64,
        memory_cache_days=2,
        preload_days_ahead=0,
        daily_cache_workers=1,
    )
    valid_count = 0
    total = len(result)
    for completed, row in enumerate(result.itertuples(index=False), 1):
        if cancel_callback is not None:
            cancel_callback()
        values: dict[str, object] = {"paper_density_status": "unavailable"}
        try:
            start = pd.Timestamp(row.start)
            end = pd.Timestamp(row.end)
            features = cache.load_features_for_trange(
                [start.strftime("%Y-%m-%d/%H:%M:%S"), end.strftime("%Y-%m-%d/%H:%M:%S")]
            )
            values.update(
                integrate_paper_icw_band(
                    features["bfac_x"],
                    features["bfac_y"],
                    features["fcp_curve"],
                    sample_times_s=features["times_s"],
                    phi=phi,
                    broaden_by_fcp_std=broaden_by_fcp_std,
                )
            )
            b0_t = float(row.B_bg_mean_nT) * 1e-9
            v_a_m_s = b0_t / np.sqrt(MU0 * float(n_sw_cm3) * 1e6 * M_PROTON)
            cos_alpha = float(np.cos(np.radians(float(row.injection_angle_deg))))
            angle_factor = (1.0 + cos_alpha) ** 2 + (1.0 - cos_alpha) ** 2
            denominator = M_PROTON * float(row.injection_speed_km_s) * 1e3 * v_a_m_s * angle_factor
            if denominator <= 0.0 or not np.isfinite(denominator):
                raise ValueError("invalid Eq. (5) denominator")
            density_m3 = 4.0 * float(values["paper_E_free_J_m3"]) / denominator
            if density_m3 <= 0.0 or not np.isfinite(density_m3):
                raise ValueError("inverted pickup H+ density is not positive")
            values.update(
                {
                    "paper_alfven_speed_km_s": v_a_m_s / 1e3,
                    "n_Hplus_pickup_m3": density_m3,
                    "n_Hplus_pickup_cm3": density_m3 / 1e6,
                    "paper_density_status": "ok",
                }
            )
            values.update(
                neutral_h_from_pickup_density(
                    density_m3,
                    float(values["paper_fgyro_hz"]),
                    n_gyro_periods=DEFAULT_N_GYRO_PERIODS,
                    photoionization_rate_s=photoionization_rate_s,
                    n_sw_cm3=n_sw_cm3,
                    v_sw_km_s=v_sw_km_s,
                )
            )
            valid_count += 1
        except (KeyError, TypeError, ValueError, OSError) as exc:
            values["paper_density_status"] = f"unavailable: {exc}"
        metric_rows.append(values)
        if progress_callback is not None:
            progress_callback(completed, total, valid_count)

    metrics = pd.DataFrame(metric_rows, index=result.index)
    combined = pd.concat([result, metrics], axis=1)
    if valid_count == 0:
        compared, diagnostics = compare_with_spicav(combined)
        compared.attrs["spicav_comparison"] = diagnostics
        return compared
    return apply_weichbold_neutral_conversion(combined, growth_phi=growth_phi)
