"""Per-event magnetic-wave energy and pickup-ion density inversion."""

import numpy as np

from venus_express.constants import M_PROTON

MU0 = 4.0 * np.pi * 1e-7
DEFAULT_PHOTOIONIZATION_RATE_S = 2.35e-7
DEFAULT_N_GYRO_PERIODS = 100.0
E_CHARGE = 1.602176634e-19
LINDSAY_H_H_ENERGY_RANGE_KEV = (0.005, 250.0)

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
