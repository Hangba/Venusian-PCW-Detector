"""Per-event tangential cyclotron-band magnetic energy metrics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from venus_express.cache import ICWFeatureCache
from venus_express.constants import E_CHARGE, M_PROTON
from venus_express.signal.fft_metrics import compute_perp_fft_metrics

MU0 = 4.0 * np.pi * 1e-7
FCP_BAND = (0.8, 1.2)


def fcp_to_bt_nt(fcp_hz: np.ndarray) -> np.ndarray:
    """Invert proton gyro-frequency to magnetic-field magnitude in nT."""

    return np.asarray(fcp_hz, dtype=np.float64) * (2.0 * np.pi * M_PROTON / E_CHARGE) * 1e9


def tangential_band_peak_energy(
    bfac_x: np.ndarray,
    bfac_y: np.ndarray,
    fcp_curve_hz: np.ndarray,
) -> dict[str, float]:
    """Return the maximum tangential PSD and energy inside 0.8-1.2 f_cp."""

    metrics = compute_perp_fft_metrics(
        bfac_x,
        bfac_y,
        None,
        fcp_to_bt_nt(fcp_curve_hz),
        fs=1.0,
        pad_factor=2,
        smooth_points=4,
        fcp_band=FCP_BAND,
    )
    peak_psd = metrics.get("in_band_peak_psd")
    frequency = np.asarray(metrics["freq"], dtype=float)
    psd_perp = np.asarray(metrics["psd_perp"], dtype=float)
    f_cp_ref = metrics.get("f_cp_ref")
    if f_cp_ref is None or not np.isfinite(f_cp_ref) or f_cp_ref <= 0:
        raise ValueError("No positive proton gyrofrequency reference")
    band_mask = (
        (frequency >= 0.8 * float(f_cp_ref))
        & (frequency <= 1.2 * float(f_cp_ref))
        & np.isfinite(psd_perp)
    )
    if np.count_nonzero(band_mask) < 2:
        raise ValueError("No usable P1+P2 samples in the f_cp band")
    integrated_psd = float(np.trapezoid(psd_perp[band_mask], frequency[band_mask]))
    if not np.isfinite(integrated_psd) or integrated_psd <= 0:
        raise ValueError("Integrated P1+P2 wave power is not positive")
    if peak_psd is None or not np.isfinite(peak_psd) or peak_psd <= 0:
        raise ValueError("No positive tangential PSD sample in the f_cp band")
    peak_energy_j = float(peak_psd) * 1e-18 / (2.0 * MU0)
    wave_energy_j = integrated_psd * 1e-18 / (2.0 * MU0)
    return {
        "f_cp_ref_hz": float(metrics["f_cp_ref"]),
        "band_peak_frequency_hz": float(metrics["peak_freq"]),
        "in_band_peak_psd_nT2_per_Hz": float(peak_psd),
        "peak_band_energy_J_m3_per_Hz": peak_energy_j,
        "peak_band_energy_pJ_m3_per_Hz": peak_energy_j * 1e12,
        "integrated_band_psd_nT2": integrated_psd,
        "wave_energy_J_m3": wave_energy_j,
        "wave_energy_pJ_m3": wave_energy_j * 1e12,
    }


def calculate_event_band_energy(
    events: pd.DataFrame,
    *,
    data_dir: str | Path,
    cache_dir: str | Path,
    start_column: str = "start",
    end_column: str = "end",
    progress_callback: Callable[[int, int, int], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Append tangential in-band peak-energy metrics to an event table."""

    missing = sorted({start_column, end_column}.difference(events.columns))
    if missing:
        raise ValueError(f"Missing event time columns: {', '.join(missing)}")
    result = events.copy()
    starts = pd.to_datetime(result[start_column], utc=True, errors="coerce")
    ends = pd.to_datetime(result[end_column], utc=True, errors="coerce")
    invalid = starts.isna() | ends.isna() | (ends <= starts)
    if bool(invalid.any()):
        raise ValueError("Event table contains invalid start/end times")

    metric_columns = {
        "band_energy_sample_count": np.zeros(len(result), dtype=np.int64),
        "f_cp_ref_hz": np.full(len(result), np.nan),
        "band_peak_frequency_hz": np.full(len(result), np.nan),
        "in_band_peak_psd_nT2_per_Hz": np.full(len(result), np.nan),
        "peak_band_energy_J_m3_per_Hz": np.full(len(result), np.nan),
        "peak_band_energy_pJ_m3_per_Hz": np.full(len(result), np.nan),
        "integrated_band_psd_nT2": np.full(len(result), np.nan),
        "wave_energy_J_m3": np.full(len(result), np.nan),
        "wave_energy_pJ_m3": np.full(len(result), np.nan),
        "band_energy_status": np.full(len(result), "unavailable", dtype=object),
    }
    cache = ICWFeatureCache(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        num_freqs=64,
        memory_cache_days=2,
        preload_days_ahead=0,
        daily_cache_workers=1,
    )
    order = np.argsort(starts.to_numpy(dtype="datetime64[ns]"))
    valid_count = 0
    total = len(result)
    for completed, row_index in enumerate(order, 1):
        if cancel_callback is not None:
            cancel_callback()
        try:
            features = cache.load_features_for_trange(
                [
                    starts.iloc[row_index].strftime("%Y-%m-%d/%H:%M:%S"),
                    ends.iloc[row_index].strftime("%Y-%m-%d/%H:%M:%S"),
                ]
            )
            bfac_x = np.asarray(features["bfac_x"], dtype=np.float64)
            bfac_y = np.asarray(features["bfac_y"], dtype=np.float64)
            fcp_curve = np.asarray(features["fcp_curve"], dtype=np.float64)
            metric_columns["band_energy_sample_count"][row_index] = min(
                bfac_x.size, bfac_y.size, fcp_curve.size
            )
            values = tangential_band_peak_energy(bfac_x, bfac_y, fcp_curve)
            for column, value in values.items():
                metric_columns[column][row_index] = value
            metric_columns["band_energy_status"][row_index] = "ok"
            valid_count += 1
        except (KeyError, TypeError, ValueError, OSError) as exc:
            metric_columns["band_energy_status"][row_index] = f"unavailable: {exc}"
        if progress_callback is not None:
            progress_callback(completed, total, valid_count)

    for column, values in metric_columns.items():
        result[column] = values
    return result
