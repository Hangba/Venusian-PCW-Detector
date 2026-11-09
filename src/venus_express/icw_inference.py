"""
ICW (Ion Cyclotron Wave) inference library
Provides sliding-window inference, candidate merging, visualization and related helpers for the webapp and other modules.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from itertools import repeat

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from venus_express.analysis import DataAnalysis
from venus_express.cache import (
    ICWFeatureCache,
    band_normalized_psd_log,
    compute_window_perp_peak_freq,
)
from venus_express.calibration import apply_logit_calibration
from venus_express.paths import CACHE_DIR, VEX_DATA_DIR
from venus_express.physics.gyro import get_gyro_frequency
from venus_express.signal.fft_metrics import compute_pf_band_peak_ratio
from venus_express.signal.filters import compute_delta_bt_from_total

PF_GATE_THRESHOLD = 3.0
PF_GATE_VERSION = "2026-07-29"
PF_GATE_REASON = (
    "Catalogue P/F study: an f_cp-band peak below 3x the out-of-band p75 is "
    "associated with broad-band ULF false positives. Applied to unconfirmed "
    "search output to reduce human review; training labels are unchanged."
)
from venus_express.signal.band_energy_gate import cyclotron_band_energy
from venus_express.physics.gyro import get_gyro_frequency

EVENT_ENERGY_GATE_THRESHOLD = 5.0
EVENT_ENERGY_GATE_VERSION = "2026-09-26-cyclotron-band-energy5-direct"
EVENT_MERGE_GAP_SECONDS_EXCLUSIVE = 60.0


def _perp_parallel_energy_ratio(bx, by, bz, fci_hz) -> float | None:
    """Return the integrated mean-transverse / parallel cyclotron-band energy."""
    try:
        return cyclotron_band_energy(bx, by, bz, fci_hz)["energy_ratio"]
    except ValueError:
        return None


def event_perp_parallel_energy_ratio(
    trange: list[str] | tuple[str, str], data_dir: str = str(VEX_DATA_DIR)
) -> float | None:
    """Return the cyclotron-band energy ratio for a merged H event."""
    analysis = DataAnalysis(list(trange), data_dir=data_dir)
    if analysis.dataframe is None or len(analysis.dataframe) == 0:
        return None
    df = analysis.dataframe
    return _perp_parallel_energy_ratio(
        df["B_fac_x"].to_numpy(dtype=np.float64, copy=False),
        df["B_fac_y"].to_numpy(dtype=np.float64, copy=False),
        df["B_fac_z"].to_numpy(dtype=np.float64, copy=False),
        get_gyro_frequency(df["Bt"].to_numpy(dtype=float) * 1e-9),
    )


def _batched_event_perp_parallel_energy_ratios(
    candidates: list[dict], *, data_dir: str
) -> list[tuple[float | None, str | None]]:
    """Compute event energy ratios with one FAC data load per UTC day.

    Same-day events share a single ``DataAnalysis`` instance and are sliced
    from its dataframe. Cross-day and malformed intervals use the scalar path
    so the batching optimization does not change their established behavior.
    """
    output: list[tuple[float | None, str | None]] = [
        (None, None) for _ in candidates
    ]
    by_day: dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp]]] = {}
    scalar_indices: list[int] = []

    for index, candidate in enumerate(candidates):
        try:
            start = pd.to_datetime(candidate["start"], utc=True)
            end = pd.to_datetime(candidate["end"], utc=True)
            if pd.isna(start) or pd.isna(end) or end < start:
                raise ValueError("invalid event interval")
        except Exception:
            scalar_indices.append(index)
            continue
        if start.date() != end.date():
            scalar_indices.append(index)
            continue
        by_day.setdefault(start.strftime("%Y-%m-%d"), []).append(
            (index, start, end)
        )

    for items in by_day.values():
        load_start = min(item[1] for item in items)
        load_end = max(item[2] for item in items)
        trange = [
            load_start.strftime("%Y-%m-%d/%H:%M:%S"),
            load_end.strftime("%Y-%m-%d/%H:%M:%S"),
        ]
        try:
            analysis = DataAnalysis(trange, data_dir=data_dir)
            df = analysis.dataframe
            if df is None or len(df) == 0:
                raise ValueError("no event data")
            times = pd.to_datetime(df["time_ut"], utc=True)
            for index, start, end in items:
                event_df = df[(times >= start) & (times <= end)]
                if len(event_df) == 0:
                    output[index] = (None, None)
                    continue
                output[index] = (
                    _perp_parallel_energy_ratio(
                        event_df["B_fac_x"].to_numpy(dtype=np.float64, copy=False),
                        event_df["B_fac_y"].to_numpy(dtype=np.float64, copy=False),
                        event_df["B_fac_z"].to_numpy(dtype=np.float64, copy=False),
                        get_gyro_frequency(event_df["Bt"].to_numpy(dtype=float) * 1e-9),
                    ),
                    None,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for index, _, _ in items:
                output[index] = (None, error)

    for index in scalar_indices:
        candidate = candidates[index]
        try:
            ratio = event_perp_parallel_energy_ratio(
                [candidate["start"], candidate["end"]], data_dir=data_dir
            )
            output[index] = (ratio, None)
        except Exception as exc:
            output[index] = (None, f"{type(exc).__name__}: {exc}")

    return output


def filter_event_energy_search_candidates(
    candidates: list[dict],
    *,
    data_dir: str = str(VEX_DATA_DIR),
    minimum_inclusive: float = EVENT_ENERGY_GATE_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """Apply the inclusive event-level band-energy gate to merged search candidates."""
    kept: list[dict] = []
    rejected: list[dict] = []
    ratios = _batched_event_perp_parallel_energy_ratios(
        candidates, data_dir=data_dir
    )
    for candidate, (ratio, ratio_error) in zip(candidates, ratios):
        candidate = dict(candidate)
        diagnostics = dict(candidate.get("model_diagnostics") or {})
        passed = ratio is not None and ratio >= minimum_inclusive
        diagnostics.update({
            "event_perp_parallel_energy_ratio": ratio,
            "event_energy_gate_threshold": float(minimum_inclusive),
            "event_energy_gate_pass": bool(passed),
            "event_energy_gate_status": "pass" if passed else "fail",
            "event_energy_gate_applied_to_search_output": True,
            "event_energy_gate_version": EVENT_ENERGY_GATE_VERSION,
            "event_energy_gate_definition": (
                "E_perp/E_parallel, PSD_perp=(P1+P2)/2; band=[0.8(fci-sigma),1.2(fci+sigma)]; "
                "greater than or equal to threshold"
            ),
        })
        if ratio_error is not None:
            diagnostics["event_energy_gate_error"] = ratio_error
        candidate["model_diagnostics"] = diagnostics
        if passed:
            kept.append(candidate)
        else:
            rejected.append(candidate)
    return kept, rejected


def _build_joint_profile_inference(
    bfac_x, bfac_y, ellip, wna_deg, fcp_curve, freq_pol,
):
    """Build joint_profile [4, F', T] for inference (mirrors ICWDataset._build_joint_profile).

    v14: coherence, DOP and fft_peak_transverse_ratio are gone -- no longer
    accepted here, and as of cache schema v14 no longer stored at all.
    """
    from scipy.fft import rfft, rfftfreq
    # Ch0: log PSD f_cp-band-normalized
    n = len(bfac_x); nfft = n * 2
    by_f = rfft(bfac_x - bfac_x.mean(), n=nfft)
    bz_f = rfft(bfac_y - bfac_y.mean(), n=nfft)
    psd = (np.abs(by_f)**2 + np.abs(bz_f)**2).astype(np.float32)
    freq_fft = rfftfreq(nfft, d=1.0).astype(np.float32)
    fcp_med = float(np.nanmedian(fcp_curve[np.isfinite(fcp_curve) & (fcp_curve > 0)]))
    if fcp_med > 0:
        lo, hi = 0.6 * fcp_med, 1.4 * fcp_med
        band_mask = (freq_fft >= lo) & (freq_fft <= hi)
        peak_in_band = float(psd[band_mask].max()) if np.any(band_mask) else float(psd.max())
    else:
        peak_in_band = float(psd.max())
    peak_in_band = max(peak_in_band, 1e-30)
    psd_log = band_normalized_psd_log(psd, peak_in_band)
    # interp to freq_pol if needed
    if psd_log.shape[0] != ellip.shape[0]:
        psd_log = np.interp(freq_pol, freq_fft[:len(psd_log)], psd_log).astype(np.float32)
    psd_2d = np.broadcast_to(psd_log[:, np.newaxis], (ellip.shape[0], ellip.shape[1])).astype(np.float32, copy=True)
    # Ch3: fcp_distance
    fcp = np.asarray(fcp_curve, dtype=np.float32).ravel().reshape(1, -1)
    freq_grid = freq_pol.reshape(-1, 1)
    valid_fcp = np.where((fcp > 0) & np.isfinite(fcp), fcp, np.nan)
    fcp_dist = (freq_grid - valid_fcp) / np.maximum(np.abs(valid_fcp), 1e-6)
    fcp_dist = np.nan_to_num(fcp_dist, nan=0.0).astype(np.float32)
    # stack (ch2 = frequency-resolved WNA, normalized angle/90 → [0,1])
    wna_norm = np.asarray(wna_deg, dtype=np.float32) / 90.0
    return np.stack([psd_2d, ellip, wna_norm, fcp_dist], axis=0).astype(np.float32)


def _build_joint_profile_batch_inference(
    precomputed: dict,
    start_indices: list[int],
    window_size: int,
    freq_pol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a batch of joint profiles with vectorized PSD computation.

    v14: 4 channels (see _build_joint_profile_inference). coh/dop/trans_ratio
    are not read here -- as of cache schema v14 they are no longer stored.

    Returns:
        profiles: [B, 4, F', T]
        fcp_curves: [B, T]
    """
    starts = np.asarray(start_indices, dtype=np.int64)
    offsets = np.arange(int(window_size), dtype=np.int64)
    idx = starts[:, None] + offsets[None, :]

    bfac_x = np.asarray(precomputed['bfac_x'], dtype=np.float32)[idx]
    bfac_y = np.asarray(precomputed['bfac_y'], dtype=np.float32)[idx]
    ellip = np.asarray(precomputed['ellip'], dtype=np.float32)[:, idx].transpose(1, 0, 2)
    wna_deg = np.asarray(precomputed['wna_deg'], dtype=np.float32)[:, idx].transpose(1, 0, 2)
    fcp_curve = np.asarray(precomputed['fcp_curve'], dtype=np.float32)[idx]

    n = int(window_size)
    nfft = n * 2
    x = np.nan_to_num(bfac_x - np.nanmean(bfac_x, axis=1, keepdims=True), nan=0.0)
    y = np.nan_to_num(bfac_y - np.nanmean(bfac_y, axis=1, keepdims=True), nan=0.0)
    psd = (np.abs(np.fft.rfft(x, n=nfft, axis=1)) ** 2 + np.abs(np.fft.rfft(y, n=nfft, axis=1)) ** 2).astype(np.float32)
    freq_fft = np.fft.rfftfreq(nfft, d=1.0).astype(np.float32)
    fcp_med = np.nanmedian(np.where(np.isfinite(fcp_curve) & (fcp_curve > 0), fcp_curve, np.nan), axis=1)
    peak_in_band = np.nanmax(psd, axis=1)
    for row_idx, fcp in enumerate(fcp_med):
        if np.isfinite(fcp) and fcp > 0:
            mask = (freq_fft >= 0.6 * fcp) & (freq_fft <= 1.4 * fcp)
            if np.any(mask):
                peak_in_band[row_idx] = float(np.max(psd[row_idx, mask]))
    peak_in_band = np.maximum(peak_in_band, 1e-30).astype(np.float32)
    psd_log = band_normalized_psd_log(psd, peak_in_band)
    if psd_log.shape[1] != freq_pol.shape[0]:
        psd_log = np.stack([
            np.interp(freq_pol, freq_fft[:psd_log.shape[1]], row).astype(np.float32)
            for row in psd_log
        ], axis=0)
    psd_2d = np.broadcast_to(psd_log[:, :, None], ellip.shape).astype(np.float32, copy=True)

    valid_fcp = np.where((fcp_curve > 0) & np.isfinite(fcp_curve), fcp_curve, np.nan)
    fcp_dist = (freq_pol[None, :, None] - valid_fcp[:, None, :]) / np.maximum(np.abs(valid_fcp[:, None, :]), 1e-6)
    fcp_dist = np.nan_to_num(fcp_dist, nan=0.0).astype(np.float32)


    profiles = np.stack([psd_2d, ellip, wna_deg / np.float32(90.0), fcp_dist], axis=1)
    return profiles.astype(np.float32, copy=False), fcp_curve.astype(np.float32, copy=False)


def _fcp_band_gate_value(ratio: float, band: tuple, softness: float) -> float:
    """Soft gate on FFT-peak/f_cp ratio: 1 inside `band`, Gaussian decay outside.

    `softness` is the relative distance outside the nearest band edge at which
    the gate falls to 1/e (e.g. 0.15 -> a peak 15% beyond the edge keeps ~37%).
    """
    lo, hi = float(band[0]), float(band[1])
    if not np.isfinite(ratio) or ratio <= 0.0:
        return 1.0  # no reliable peak/f_cp: leave the model score untouched
    if lo <= ratio <= hi:
        return 1.0
    d = (lo - ratio) / lo if ratio < lo else (ratio - hi) / hi
    return float(np.exp(-((d / max(softness, 1e-6)) ** 2)))


def _ellip_lh_gate_value(ellip_val: float, lh_threshold: float, softness: float) -> float:
    """Soft gate on signed ellipticity at the FFT peak.

    Left-hand ICW has ellipticity approaching -1. The gate is 1 for windows at
    least as left-hand as `lh_threshold` (default -0.5) and decays as a Gaussian
    toward linear / right-hand polarization.

    `softness` is the absolute ellipticity distance above the threshold at which
    the gate falls to 1/e (e.g. 0.3 -> ellip = lh_threshold + 0.3 keeps ~37%).
    A NaN ellipticity (no reliable measurement) leaves the score untouched.
    """
    if not np.isfinite(ellip_val):
        return 1.0  # no reliable ellipticity: leave the model score untouched
    if ellip_val <= lh_threshold:
        return 1.0
    d = ellip_val - lh_threshold
    return float(np.exp(-((d / max(softness, 1e-6)) ** 2)))


def _window_ellip_at_freq(
    precomputed: dict, start_idx: int, window_size: int,
    freq_pol: np.ndarray, f_target: float,
) -> float:
    """Mean signed ellipticity at the freq bin nearest `f_target`, over the
    window's time columns. Mirrors export.py `_mean_at` so the gated value
    matches the figure's `Ellip @ f_max`.
    """
    ellip = precomputed.get('ellip')
    if ellip is None or freq_pol.size == 0 or not np.isfinite(f_target) or f_target <= 0:
        return float('nan')
    bin_i = int(np.argmin(np.abs(freq_pol - f_target)))
    col = np.asarray(ellip[bin_i, start_idx:start_idx + window_size], dtype=np.float64)
    finite = col[np.isfinite(col)]
    return float(np.mean(finite)) if finite.size else float('nan')


def _window_wna_at_freq(
    precomputed: dict, start_idx: int, window_size: int,
    freq_pol: np.ndarray, f_target: float,
) -> float:
    """Mean WNA at the frequency bin nearest the window FFT peak."""
    wna = precomputed.get('wna_deg')
    if wna is None or freq_pol.size == 0 or not np.isfinite(f_target) or f_target <= 0:
        return float('nan')
    bin_i = int(np.argmin(np.abs(freq_pol - f_target)))
    col = np.asarray(wna[bin_i, start_idx:start_idx + window_size], dtype=np.float64)
    finite = col[np.isfinite(col)]
    return float(np.mean(finite)) if finite.size else float('nan')


def normalize_score(raw_score: float) -> float:
    """Convert model output to a stable [0, 1] score."""
    if not np.isfinite(raw_score):
        return 0.0

    # If a logits-style output is detected, map it through sigmoid.
    if raw_score < 0.0 or raw_score > 1.0:
        raw_score = float(1.0 / (1.0 + np.exp(-np.clip(raw_score, -60.0, 60.0))))

    return float(np.clip(raw_score, 0.0, 1.0))


def precompute_features(
    trange: list,
    data_dir: str,
    num_freqs: int,
    cache_dir: str,
    progress_callback=None,
    cache: ICWFeatureCache | None = None,
):
    """
    Precompute features for the whole range (cached by date); windows are sliced from it afterwards.

    Args:
        cache: optional external ICWFeatureCache instance. When provided, its in-memory cache
            (`_daily_mem_cache`) is reused, avoiding repeated disk reads in scenarios such as
            the supervised loop. When omitted, a new instance is built as before.

    Returns:
        dict (v14 schema): {
            'delta_bt': [T],
            'bfac_x': [T],
            'bfac_y': [T],
            'ellip': [F', T],          # signed only
            'wna_deg': [F', T],        # WNA angle (deg), frequency-resolved (v12)
            'fcp_curve': [T],          # local proton cyclotron frequency, Hz
            'freq_pol': [F'],          # ellip frequency axis, Hz
            'times_s': [T]
        }
        As of v14, coh / dop / fft_peak_transverse_ratio are no longer present.
    """
    if cache is None:
        cache = ICWFeatureCache(
            data_dir=data_dir,
            cache_dir=cache_dir,
            num_freqs=num_freqs,
            pol_window_size=160,  # v9 default
            pol_step_size=20,
            progress_callback=progress_callback,
        )
        return cache.load_features_for_trange(trange=trange)

    # Shared cache: temporarily install this call's progress callback and
    # restore it on exit so subsequent jobs (e.g. training right after
    # inference in the supervised loop) don't fire stale callbacks.
    saved_cb = cache.progress_callback
    cache.progress_callback = progress_callback
    try:
        return cache.load_features_for_trange(trange=trange)
    finally:
        cache.progress_callback = saved_cb


def extract_features_for_window(precomputed: dict, start_idx: int, window_size: int):
    """Slice a single window from the precomputed features (v14 schema).

    Returns 7-tuple (matches ICWDataset._extract_features v14):
        0 bt: [T]          delta_bt raw value
        1 bfac_x: [T]      FAC tangential component x
        2 bfac_y: [T]      FAC tangential component y
        4 ellip: [F', T]   signed ellipticity
        5 wna_deg: [F', T] WNA angle (deg), frequency-resolved, quasi-parallel→0 (v12)
        6 fcp_curve: [T]   local proton cyclotron frequency, Hz

    v14 removes coh / dop / fft_peak_transverse_ratio and the two zero-filled placeholder slots.
    """
    end_idx = start_idx + window_size
    bt = precomputed['delta_bt'][start_idx:end_idx]
    bfac_x = precomputed['bfac_x'][start_idx:end_idx]
    bfac_y = precomputed['bfac_y'][start_idx:end_idx]
    ellip = precomputed['ellip'][:, start_idx:end_idx]
    wna_deg = precomputed['wna_deg'][:, start_idx:end_idx]
    fcp_curve = precomputed['fcp_curve'][start_idx:end_idx]
    return (bt, bfac_x, bfac_y, ellip, wna_deg, fcp_curve)


def _build_ellip_with_fcp_distance(
    ellip_signed: np.ndarray,
    fcp_curve: np.ndarray,
    freq_pol: np.ndarray,
    fft_peak_freq: np.ndarray | None = None,
) -> np.ndarray:
    """Build [2, F', T] ellip: signed + min_ref_distance.

    Channel 1 takes whichever of the fcp and fft_peak_freq distances is closer (absolute
    value), so the true ICW frequency sits ≈ 0 whether or not it lies exactly on fcp.
    """
    ellip_signed = np.asarray(ellip_signed, dtype=np.float32)
    if ellip_signed.ndim != 2:
        raise ValueError(
            f"ellip_signed must be [F', T], got {ellip_signed.shape}"
        )
    F_prime, T = ellip_signed.shape
    freq_axis = np.asarray(freq_pol, dtype=np.float32)
    fcp = np.asarray(fcp_curve, dtype=np.float32)
    if freq_axis.size != F_prime or fcp.shape[0] != T:
        ref_distance = np.zeros((F_prime, T), dtype=np.float32)
    else:
        safe_fcp = np.where(np.isfinite(fcp) & (fcp > 0.0), fcp, np.nan)
        dist_fcp = np.abs(
            (freq_axis[:, None] - safe_fcp[None, :]) / safe_fcp[None, :]
        )

        if fft_peak_freq is not None:
            pf = np.asarray(fft_peak_freq, dtype=np.float32)
            if pf.shape[0] == T:
                safe_pf = np.where(np.isfinite(pf) & (pf > 0.0), pf, np.nan)
                dist_pf = np.abs(
                    (freq_axis[:, None] - safe_pf[None, :]) / safe_pf[None, :]
                )
                ref_distance = np.fmin(dist_fcp, dist_pf)
            else:
                ref_distance = dist_fcp
        else:
            ref_distance = dist_fcp

        ref_distance = np.nan_to_num(ref_distance, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.stack([ellip_signed, ref_distance], axis=0).astype(np.float32, copy=False)


def _compute_fcp_band_lh_fraction(
    ellip_signed: np.ndarray,
    fcp_curve: np.ndarray,
    freq_pol: np.ndarray,
    lh_thresh: float = -0.3,
    band_width: float = 0.3,
    fft_peak_freq: np.ndarray | None = None,
) -> float:
    """LH polarization coverage fraction in the band near f_cp or the FFT peak frequency.

    The band is the union of f_cp and fft_peak_freq; satisfying either puts a bin in band.
    """
    ellip_signed = np.asarray(ellip_signed, dtype=np.float32)
    fcp_curve = np.asarray(fcp_curve, dtype=np.float32)
    freq_pol = np.asarray(freq_pol, dtype=np.float32)

    if ellip_signed.size == 0 or freq_pol.size == 0:
        return 0.0

    if freq_pol.size != ellip_signed.shape[0]:
        return 0.0
    if fcp_curve.shape[0] != ellip_signed.shape[1]:
        return 0.0

    valid_fcp = np.isfinite(fcp_curve) & (fcp_curve > 0)
    if not np.any(valid_fcp):
        return 0.0

    safe_fcp = np.where(valid_fcp, fcp_curve, np.nan)
    rel_distance_fcp = np.abs((freq_pol[:, None] - safe_fcp[None, :]) / safe_fcp[None, :])
    band_mask = np.isfinite(rel_distance_fcp) & (rel_distance_fcp <= float(band_width))

    # FFT peak-frequency band (union)
    if fft_peak_freq is not None:
        fft_pf = np.asarray(fft_peak_freq, dtype=np.float32)
        if fft_pf.shape[0] == ellip_signed.shape[1]:
            valid_pf = np.isfinite(fft_pf) & (fft_pf > 0)
            safe_pf = np.where(valid_pf, fft_pf, np.nan)
            rel_distance_pf = np.abs((freq_pol[:, None] - safe_pf[None, :]) / safe_pf[None, :])
            pf_mask = np.isfinite(rel_distance_pf) & (rel_distance_pf <= float(band_width))
            band_mask = band_mask | pf_mask

    if not np.any(band_mask):
        return 0.0
    lh_mask = ellip_signed < lh_thresh
    return float(np.count_nonzero(lh_mask & band_mask) / np.count_nonzero(band_mask))


def _compute_fft_peak_fcp_distance(bt: np.ndarray, fcp_curve: np.ndarray, fs: float = 1.0) -> float:
    """Distance of the FFT main-peak frequency from the mean f_cp: |f_peak/f_cp - 1|."""
    values = np.asarray(bt, dtype=np.float32)
    fcp = np.asarray(fcp_curve, dtype=np.float32)
    if values.size < 4:
        return 0.0
    valid_fcp = fcp[np.isfinite(fcp) & (fcp > 0)]
    if valid_fcp.size == 0:
        return 0.0
    demeaned = np.nan_to_num(values - float(np.nanmean(values)), nan=0.0)
    spectrum = np.abs(np.fft.rfft(demeaned)) ** 2
    freqs = np.fft.rfftfreq(values.size, d=1.0 / fs)
    if spectrum.size <= 1:
        return 0.0
    spectrum[0] = 0.0
    peak_freq = float(freqs[int(np.argmax(spectrum))])
    fcp_avg = float(valid_fcp.mean())
    return float(abs((peak_freq / max(fcp_avg, 1e-6)) - 1.0))


def _build_log_peak_normalized_psd_profile(
    bfac_x: np.ndarray,
    bfac_y: np.ndarray,
    fcp_curve: np.ndarray,
    fs: float = 1.0,
    eps: float = 1e-8,
    pad_factor: int = 2,
    smooth_points: int = 4,
) -> np.ndarray:
    """Build a 2-channel PSD profile: [log10 peak-normalized PSD_perp, fcp_distance].

    Mirrors ICWDataset._build_log_peak_normalized_psd_profile exactly so the
    deployed model sees identical inputs to training. Channel 0 = peak-normalized
    log10 PSD_perp; channel 1 = signed (f - f_cp)/f_cp on the PSD's rFFT axis
    (0 at f_cp), giving the PSD CNN explicit f_cp registration. Returns [2, F].
    """
    x_values = np.asarray(bfac_x, dtype=np.float32).reshape(-1)
    y_values = np.asarray(bfac_y, dtype=np.float32).reshape(-1)
    n = min(x_values.size, y_values.size)
    if n < 4:
        return np.zeros((2, 1), dtype=np.float32)
    x_values = x_values[:n]
    y_values = y_values[:n]

    clean_x = np.nan_to_num(x_values - float(np.nanmean(x_values)), nan=0.0)
    clean_y = np.nan_to_num(y_values - float(np.nanmean(y_values)), nan=0.0)
    nfft = int(n * max(1, int(pad_factor)))

    def one_sided_psd(signal: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(signal, n=nfft)
        psd_values = (np.abs(spec) ** 2) / (float(fs) * float(n))
        if nfft % 2 == 0:
            if psd_values.size > 2:
                psd_values[1:-1] *= 2.0
        elif psd_values.size > 1:
            psd_values[1:] *= 2.0
        return psd_values.astype(np.float32, copy=False)

    psd = one_sided_psd(clean_x) + one_sided_psd(clean_y)
    if psd.size:
        psd[0] = 0.0
    w = max(1, min(int(smooth_points), psd.size))
    if w > 1:
        pad_left = w // 2
        pad_right = w - 1 - pad_left
        kernel = np.full(w, 1.0 / w, dtype=np.float32)
        padded = np.pad(psd, (pad_left, pad_right), mode="edge")
        psd = np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)

    # Channel 1: signed (f - f_cp) / f_cp on the PSD's own rFFT frequency axis.
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fs)).astype(np.float32)
    fcp = np.asarray(fcp_curve, dtype=np.float32)
    valid_fcp = fcp[np.isfinite(fcp) & (fcp > 0.0)]
    if valid_fcp.size and freqs.size == psd.size:
        fcp_mean = float(valid_fcp.mean())
        fcp_dist = ((freqs - fcp_mean) / max(fcp_mean, 1e-6)).astype(np.float32)
    else:
        fcp_dist = np.zeros_like(psd, dtype=np.float32)
    fcp_dist = np.nan_to_num(fcp_dist, nan=0.0, posinf=0.0, neginf=0.0)

    finite = psd[np.isfinite(psd)]
    peak_power = float(np.max(finite)) if finite.size else 0.0
    if peak_power <= 0.0:
        profile = np.zeros_like(psd, dtype=np.float32)
    else:
        relative = np.clip(psd / peak_power, float(eps), None)
        profile = np.nan_to_num(
            np.log10(relative), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
    return np.stack([profile, fcp_dist], axis=0).astype(np.float32, copy=False)


def _compute_valid_point_fraction(
    bt: np.ndarray,
    ellip_signed: np.ndarray,
    fcp_curve: np.ndarray,
) -> float:
    """Fraction of finite values among features currently used by the model."""
    parts = [
        np.asarray(bt).reshape(-1),
        np.asarray(ellip_signed).reshape(-1),
        np.asarray(fcp_curve).reshape(-1),
    ]
    total = sum(int(part.size) for part in parts)
    if total <= 0:
        return 0.0
    finite = sum(int(np.count_nonzero(np.isfinite(part))) for part in parts)
    return float(finite / total)


class CacheAlignmentError(RuntimeError):
    """Raised when cache features cannot be strictly aligned with raw analysis samples."""

    def __init__(self, message: str, suspect_dates: list[str] | None = None):
        super().__init__(message)
        self.suspect_dates = suspect_dates or []


def _infer_misaligned_dates(
    analysis_times_us: np.ndarray,
    cache_times_us: np.ndarray,
    max_dates: int = 8,
) -> list[str]:
    """Infer suspect UTC dates where cache/raw timestamps diverge."""
    if analysis_times_us.size == 0 or cache_times_us.size == 0:
        return []

    us_per_day = 86_400_000_000
    analysis_day_ids = np.floor_divide(analysis_times_us, us_per_day)
    cache_day_ids = np.floor_divide(cache_times_us, us_per_day)

    a_days, a_starts, a_counts = np.unique(analysis_day_ids, return_index=True, return_counts=True)
    c_days, c_starts, c_counts = np.unique(cache_day_ids, return_index=True, return_counts=True)

    analysis_map = {
        int(day_id): (int(start), int(count))
        for day_id, start, count in zip(a_days, a_starts, a_counts)
    }
    cache_map = {
        int(day_id): (int(start), int(count))
        for day_id, start, count in zip(c_days, c_starts, c_counts)
    }

    suspect_ids: list[int] = []
    for day_id in sorted(set(analysis_map.keys()) | set(cache_map.keys())):
        a_info = analysis_map.get(day_id)
        c_info = cache_map.get(day_id)
        if a_info is None or c_info is None:
            suspect_ids.append(day_id)
        else:
            a_start, a_count = a_info
            c_start, c_count = c_info
            if a_count != c_count:
                suspect_ids.append(day_id)
            else:
                if not np.array_equal(
                    analysis_times_us[a_start:a_start + a_count],
                    cache_times_us[c_start:c_start + c_count],
                ):
                    suspect_ids.append(day_id)
        if len(suspect_ids) >= max_dates:
            break

    if not suspect_ids:
        return []

    suspect_start_us = np.asarray(suspect_ids, dtype=np.int64) * us_per_day
    suspect_dt = pd.to_datetime(suspect_start_us, unit='us', utc=True)
    return [ts.strftime('%Y-%m-%d') for ts in suspect_dt]


def _validate_precomputed_features(precomputed: dict) -> tuple[int, np.ndarray]:
    """Validate cache tensor shapes and timestamp monotonicity."""
    feature_samples = precomputed['times_s'].shape[0]
    if not (
        precomputed['delta_bt'].shape[0] == feature_samples
        and precomputed['bfac_x'].shape[0] == feature_samples
        and precomputed['bfac_y'].shape[0] == feature_samples
        and precomputed['ellip'].ndim == 2
        and precomputed['ellip'].shape[1] == feature_samples
        and precomputed['fcp_curve'].shape[0] == feature_samples
        and precomputed['freq_pol'].ndim == 1
        and precomputed['freq_pol'].size == precomputed['ellip'].shape[0]
    ):
        raise ValueError(
            "Cached features are internally misaligned: "
            f"times={feature_samples}, "
            f"delta_bt={precomputed['delta_bt'].shape[0]}, "
            f"bfac_x={precomputed['bfac_x'].shape[0]}, "
            f"bfac_y={precomputed['bfac_y'].shape[0]}, "
            f"ellip={precomputed['ellip'].shape}, "
            f"fcp_curve={precomputed['fcp_curve'].shape}, "
            f"freq_pol={precomputed['freq_pol'].shape}"
        )

    cache_times_us = np.rint(precomputed['times_s'] * 1e6).astype(np.int64)
    if cache_times_us.size > 1 and np.any(cache_times_us[1:] < cache_times_us[:-1]):
        raise CacheAlignmentError(
            "Cached feature timestamps are not monotonically increasing. "
            "Please rebuild the daily cache and rerun."
        )

    return feature_samples, cache_times_us


def _trim_boundary_end_sample_if_needed(
    analysis: DataAnalysis,
    precomputed: dict,
    trange: list,
) -> bool:
    """
    Trim boundary spillover samples from analysis when cache is shorter.

    This handles leap-second style spillover where raw loading can include
    boundary samples (typically `00:00:00`) that are absent from day-cache
    concatenation.
    """
    analysis_times_us = pd.DatetimeIndex(analysis.dataframe['time_ut']).asi8.astype(np.int64, copy=False)
    cache_times_us = np.rint(precomputed['times_s'] * 1e6).astype(np.int64)

    if analysis_times_us.size <= cache_times_us.size:
        return False

    i = 0
    j = 0
    extra_indices: list[int] = []
    while i < analysis_times_us.size and j < cache_times_us.size:
        if analysis_times_us[i] == cache_times_us[j]:
            i += 1
            j += 1
            continue

        # Allow dropping one raw sample if the following sample re-aligns.
        if i + 1 < analysis_times_us.size and analysis_times_us[i + 1] == cache_times_us[j]:
            extra_indices.append(i)
            i += 1
            continue

        return False

    t_end = pd.to_datetime(trange[1], utc=True)
    end_us = int(t_end.value // 1000)

    if j < cache_times_us.size:
        return False

    # Any remaining raw samples are extras.
    if i < analysis_times_us.size:
        extra_indices.extend(range(i, analysis_times_us.size))

    if not extra_indices:
        return False

    # Only permit trimming safe boundary-style samples.
    for idx in extra_indices:
        ts = pd.to_datetime(int(analysis_times_us[idx]), unit='us', utc=True)
        is_midnight = (
            ts.hour == 0
            and ts.minute == 0
            and ts.second == 0
            and ts.microsecond == 0
        )
        is_trange_end = int(analysis_times_us[idx]) == end_us
        if not (is_midnight or is_trange_end):
            return False

    keep_mask = np.ones(analysis_times_us.size, dtype=bool)
    keep_mask[np.asarray(extra_indices, dtype=np.int64)] = False
    trimmed_times_us = analysis_times_us[keep_mask]
    if trimmed_times_us.shape[0] != cache_times_us.shape[0] or not np.array_equal(trimmed_times_us, cache_times_us):
        return False

    analysis.dataframe = analysis.dataframe.iloc[keep_mask].reset_index(drop=True)
    return True


def sliding_window_inference(
    model: torch.nn.Module,
    trange: list,
    data_dir: str = str(VEX_DATA_DIR),
    window_size: int = 80,
    step_size: int = 20,
    num_freqs: int = 64,
    device: torch.device = None,
    threshold: float = 0.6,
    feature_cache_dir: str = str(CACHE_DIR),
    inference_batch_size: int = 2,
    on_progress=None,
    on_cache_progress=None,
    feature_cache: ICWFeatureCache | None = None,
    freq_scale_factor: float = 1.0,
    vectorized_feature_build: bool = False,
    fcp_band_gate: bool = True,
    fcp_band: tuple = (0.8, 1.2),
    fcp_band_gate_softness: float = 0.15,
    ellip_gate: bool = True,
    ellip_lh_threshold: float = -0.5,
    ellip_gate_softness: float = 0.05,
    apply_pf_search_gate: bool = True,
    compute_window_fft_diagnostics: bool = True,
):
    """
    Sliding-window inference.

    Args:
        model: trained model
        trange: time range ['YYYY-MM-DD/HH:MM:SS', 'YYYY-MM-DD/HH:MM:SS']
        data_dir: data directory
        window_size: window size (seconds)
        step_size: step size (seconds)
        num_freqs: number of wavelet frequency bins
        device: compute device
        threshold: confidence threshold
        freq_scale_factor: ion cyclotron frequency scale factor (default 1.0 = proton).
            The model is trained on proton ICW (f_cp = qB/2πm_p); to zero-shot search
            other ions, multiply the f_cp anchor by this factor so the model peaks at
            the target ion's true cyclotron frequency. f_c ∝ Z/A, so factor = Z/A:
            proton=1.0, D⁺(A=2,Z=1)=0.5, He²⁺(A=4,Z=2)=0.5, He⁺(A=4,Z=1)=0.25,
            O⁺(A=16,Z=1)=0.0625. Heavy ions <1 (lower frequency).
        vectorized_feature_build: experimental batch feature builder. Default
            stays on the lower-memory per-window path; synthetic microbenchmarks
            show the vectorized path is not consistently faster for 80s windows.

    Returns:
        results: list of dict, per-window predictions
        candidates: list of dict, high-confidence candidates
    """
    if device is None:
        device = torch.device('cpu')
    inference_batch_size = max(1, int(inference_batch_size))

    trange_effective = normalize_trange_for_data_loading(trange)
    if trange_effective != trange:
        print(
            "[info] End boundary at 00:00:00 is treated as exclusive; "
            f"effective trange: {trange_effective}"
        )

    print("Loading/building daily cached features for this range...")
    precomputed = precompute_features(
        trange=trange_effective,
        data_dir=data_dir,
        num_freqs=num_freqs,
        cache_dir=feature_cache_dir,
        progress_callback=on_cache_progress,
        cache=feature_cache,
    )

    feature_samples, cache_times_us = _validate_precomputed_features(precomputed)
    if feature_samples < window_size:
        raise ValueError(
            f"Not enough aligned samples for inference: samples={feature_samples}, "
            f"window_size={window_size}"
        )

    # Non-proton ion search: scale the f_cp anchor in-place once, before the
    # window loop. Every downstream joint-profile channel that uses
    # precomputed['fcp_curve'] reads the same slice, so this single edit
    # retargets the frequency anchor.
    # An ion's cyclotron frequency is f_c = qB/(2πm) ∝ Z/A, so HEAVIER ions sit
    # at a LOWER frequency than the proton. freq_scale_factor is defined as Z/A
    # (proton=1, He²⁺=0.5, He⁺=0.25, O⁺=0.0625), i.e. the cyclotron-frequency
    # ratio itself. The effective anchor is therefore:
    #   effective_fcp = f_cp_proton * freq_scale_factor   (LOWER for heavy ions)
    # so the model — trained to expect the FFT peak near fcp_curve — looks for the
    # peak at the ion's real (lower) line. Only fcp_curve is scaled; ch0/ch3 are
    # rebuilt from it per window.
    scale = float(freq_scale_factor)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"freq_scale_factor must be finite > 0, got {freq_scale_factor}")
    if abs(scale - 1.0) > 1e-9:
        fcp = np.asarray(precomputed['fcp_curve'], dtype=np.float32)
        precomputed = dict(precomputed)
        precomputed['fcp_curve'] = (fcp * scale).astype(np.float32)
        print(f"[ion-search] f_cp scaled by {scale:g} (Z/A) "
              f"(median {np.nanmedian(fcp):.4f} -> {np.nanmedian(fcp)*scale:.4f} Hz)")
        # v13+ cache stores full-band ellipticity. New models consume it directly;
        # legacy checkpoints can re-mask around the scaled f_cp through the explicit
        # compatibility switch. The peak ellipticity gate always reads full-band data.

    times = pd.to_datetime(cache_times_us, unit='us', utc=True)

    print(f"Total samples: {feature_samples}")
    print(f"Window size: {window_size}s, Step size: {step_size}s")

    results = []
    n_windows = (feature_samples - window_size) // step_size + 1

    print(f"Processing {n_windows} windows...")

    # Phase 8: cache schema v8 — freq_pol is used to build the fcp_distance channel.
    freq_pol = np.asarray(precomputed['freq_pol'], dtype=np.float32)
    freq_pol_tensor = torch.from_numpy(freq_pol).to(device)

    window_indices = list(range(0, feature_samples - window_size + 1, step_size))
    pbar = tqdm(total=n_windows, desc="Sliding-window inference", leave=False, disable=not sys.stderr.isatty())
    for batch_start in range(0, n_windows, inference_batch_size):
        batch_indices = window_indices[batch_start:batch_start + inference_batch_size]

        if vectorized_feature_build:
            profile_batch, fcp_batch = _build_joint_profile_batch_inference(
                precomputed, batch_indices, window_size, freq_pol,
            )
        else:
            profile_items = []
            fcp_items = []
            for start_idx in batch_indices:
                (bt, bfac_x, bfac_y, ellip, wna_deg,
                 fcp_curve) = extract_features_for_window(
                    precomputed, start_idx, window_size,
                )
                profile = _build_joint_profile_inference(
                    bfac_x, bfac_y, ellip, wna_deg, fcp_curve, freq_pol,
                )
                profile_items.append(profile)
                fcp_items.append(np.asarray(fcp_curve, dtype=np.float32))
            profile_batch = np.stack(profile_items, axis=0)
            fcp_batch = np.stack(fcp_items, axis=0)
        profile_tensor = torch.from_numpy(profile_batch).to(device)
        fcp_tensor = torch.from_numpy(fcp_batch).to(device)

        with torch.inference_mode():
            raw_logits = model(
                profile_tensor, fcp_curve=fcp_tensor, freq_pol=freq_pol_tensor
            ).detach().cpu().numpy()
        uncalibrated_scores = 1.0 / (
            1.0 + np.exp(-np.clip(raw_logits, -60.0, 60.0))
        )
        raw_scores = apply_logit_calibration(
            raw_logits, getattr(model, "probability_calibration", None)
        )

        for local_idx, (start_idx, raw_score, uncalibrated_score) in enumerate(
            zip(batch_indices, raw_scores, uncalibrated_scores)
        ):
            normalized_score = normalize_score(float(raw_score))
            seg_bx = precomputed['bfac_x'][start_idx:start_idx + window_size]
            seg_by = precomputed['bfac_y'][start_idx:start_idx + window_size]
            tangential_rms_nT = float(np.sqrt(np.mean(seg_bx * seg_bx + seg_by * seg_by)))

            # Tier-0 physics gate: the 0.8-1.2 f_cp band requirement and the
            # left-hand polarization requirement (ellipticity ~ -1) are hard
            # criteria, so enforce them here on each window instead of trusting
            # the CNN to have learned them. fcp_curve is already divided by
            # freq_scale_factor, so non-proton f_cp gating stays consistent
            # (the ellip gate is auto-disabled for scaled searches above).
            fcp_peak_ratio = float('nan')
            gate = 1.0
            ellip_at_peak = float('nan')
            wna_at_peak = float('nan')
            ellip_gate_value = 1.0
            peak = float('nan')
            if compute_window_fft_diagnostics and (fcp_band_gate or ellip_gate):
                peak = compute_window_perp_peak_freq(seg_bx, seg_by, fs=1.0)
            if compute_window_fft_diagnostics and fcp_band_gate:
                seg_fcp = precomputed['fcp_curve'][start_idx:start_idx + window_size]
                valid_fcp = seg_fcp[np.isfinite(seg_fcp) & (seg_fcp > 0)]
                if valid_fcp.size and np.isfinite(peak) and peak > 0:
                    fcp_peak_ratio = float(peak / np.median(valid_fcp))
                gate = _fcp_band_gate_value(
                    fcp_peak_ratio, fcp_band, fcp_band_gate_softness
                )
            if compute_window_fft_diagnostics and ellip_gate:
                ellip_at_peak = _window_ellip_at_freq(
                    precomputed, start_idx, window_size, freq_pol, peak
                )
                ellip_gate_value = _ellip_lh_gate_value(
                    ellip_at_peak, ellip_lh_threshold, ellip_gate_softness
                )
            if compute_window_fft_diagnostics and np.isfinite(peak) and peak > 0:
                wna_at_peak = _window_wna_at_freq(
                    precomputed, start_idx, window_size, freq_pol, peak
                )
            gated_score = float(
                np.clip(normalized_score * gate * ellip_gate_value, 0.0, 1.0)
            )
            physics_compatibility = float(np.clip(gate * ellip_gate_value, 0.0, 1.0))

            # Added 2026-07-29 after the catalogue P/F study. This remains
            # separate from the sigmoid probability and filters only search
            # output sent to human review; confirmed training labels are not
            # re-evaluated by this path.
            pf_ratio = float("nan")
            pf_gate_pass = None
            pf_gate_status = "not_applicable_non_proton"
            if compute_window_fft_diagnostics and abs(scale - 1.0) <= 1e-9:
                seg_fcp = precomputed['fcp_curve'][start_idx:start_idx + window_size]
                pf_ratio = compute_pf_band_peak_ratio(seg_bx, seg_by, seg_fcp)
                if np.isfinite(pf_ratio):
                    pf_gate_pass = bool(pf_ratio >= PF_GATE_THRESHOLD)
                    pf_gate_status = "pass" if pf_gate_pass else "review"
                else:
                    pf_gate_status = "unavailable"

            model_diagnostics = {
                "pf_band_ratio": float(pf_ratio) if np.isfinite(pf_ratio) else None,
                "pf_band_ratio_window": float(pf_ratio) if np.isfinite(pf_ratio) else None,
                "pf_gate_aggregation": "window",
                "pf_gate_window_count": 1 if np.isfinite(pf_ratio) else 0,
                "pf_gate_threshold": PF_GATE_THRESHOLD,
                "pf_gate_pass": pf_gate_pass,
                "pf_gate_status": pf_gate_status,
                "pf_gate_applied_to_score": False,
                "pf_gate_applied_to_search_output": abs(scale - 1.0) <= 1e-9,
                "pf_gate_version": PF_GATE_VERSION,
                "pf_gate_reason": PF_GATE_REASON,
                "pf_gate_definition": {
                    "window_seconds": int(window_size),
                    "in_band_fcp": [0.8, 1.2],
                    "in_band_statistic": "max",
                    "out_band": "all positive frequencies outside in_band",
                    "out_band_statistic": "p75",
                },
            }

            window_start = times[start_idx]
            sample_end = times[start_idx + window_size - 1]
            min_window_end = window_start + pd.Timedelta(seconds=int(window_size))
            window_end = max(sample_end, min_window_end)

            results.append({
                'start': window_start.strftime('%Y-%m-%d/%H:%M:%S'),
                'end': window_end.strftime('%Y-%m-%d/%H:%M:%S'),
                'start_idx': start_idx,
                'raw_score': float(normalized_score),
                'uncalibrated_score': float(uncalibrated_score),
                'calibrated_probability': float(normalized_score),
                'physics_compatibility': physics_compatibility,
                'rms_tangential_nT': tangential_rms_nT,
                'score': gated_score,
                'fcp_peak_ratio': (
                    float(fcp_peak_ratio) if np.isfinite(fcp_peak_ratio) else None
                ),
                'fcp_band_gate': float(gate),
                'ellip_at_peak': (
                    float(ellip_at_peak) if np.isfinite(ellip_at_peak) else None
                ),
                'wna_at_peak': (
                    float(wna_at_peak) if np.isfinite(wna_at_peak) else None
                ),
                'ellip_gate': float(ellip_gate_value),
                'model_diagnostics': model_diagnostics,
            })
        pbar.update(len(batch_indices))
        if on_progress is not None:
            try:
                on_progress(min(batch_start + len(batch_indices), n_windows), n_windows)
            except Exception as cb_exc:
                print(f"[warn] on_progress callback failed: {cb_exc}")
    pbar.close()

    # Filter high-confidence candidates
    candidates = [r for r in results if r['score'] >= threshold]
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

    # Merge adjacent high-confidence windows
    merged_before_pf_gate = merge_adjacent_windows(candidates, step_size)
    if apply_pf_search_gate:
        merged_candidates, pf_rejected = filter_pf_search_candidates(
            merged_before_pf_gate
        )
    else:
        merged_candidates, pf_rejected = merged_before_pf_gate, []

    print(
        f"\nFound {len(merged_candidates)} candidate ICW events "
        f"(threshold={threshold}, P/F rejected={len(pf_rejected)})"
    )

    return results, merged_candidates, {
        'pf_gate_rejected': len(pf_rejected),
        'pf_gate_before': len(merged_before_pf_gate),
        'pf_gate_after': len(merged_candidates),
    }


def merge_adjacent_windows(
    candidates: list,
    step_size: int,
    max_gap_seconds_exclusive: float = EVENT_MERGE_GAP_SECONDS_EXCLUSIVE,
) -> list:
    """Merge candidate windows when the gap is strictly less than one minute."""
    if not candidates:
        return []

    def _window_ratios(candidate: dict) -> list[float]:
        existing = candidate.get('_pf_band_ratios')
        if isinstance(existing, list):
            return [float(v) for v in existing if np.isfinite(v)]
        diagnostics = candidate.get('model_diagnostics') or {}
        value = diagnostics.get('pf_band_ratio')
        try:
            value = float(value)
        except (TypeError, ValueError):
            return []
        return [value] if np.isfinite(value) else []

    def _finalize_pf_wmed(candidate: dict) -> dict:
        ratios = _window_ratios(candidate)
        candidate['_pf_band_ratios'] = ratios
        diagnostics = candidate.get('model_diagnostics')
        if not isinstance(diagnostics, dict) or not ratios:
            return candidate
        diagnostics = dict(diagnostics)
        ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
        passed = bool(ratio >= PF_GATE_THRESHOLD)
        diagnostics.update({
            'pf_band_ratio': ratio,
            'pf_gate_pass': passed,
            'pf_gate_status': 'pass' if passed else 'review',
            'pf_gate_aggregation': 'wmed',
            'pf_gate_window_count': len(ratios),
        })
        candidate['model_diagnostics'] = diagnostics
        return candidate

    # Sort by start time
    sorted_candidates = sorted(candidates, key=lambda x: x['start'])

    merged = []
    current = sorted_candidates[0].copy()
    current['_pf_band_ratios'] = _window_ratios(current)

    for next_window in sorted_candidates[1:]:
        # Keep the legacy step_size argument for call-site compatibility. Event
        # identity now follows the explicit production rule: gap < 60 seconds.
        current_end = pd.Timestamp(current['end'])
        next_start = pd.Timestamp(next_window['start'])

        time_gap = (next_start - current_end).total_seconds()

        if time_gap < float(max_gap_seconds_exclusive):
            # Merge
            current['_pf_band_ratios'].extend(_window_ratios(next_window))
            current['end'] = next_window['end']
            if next_window['score'] >= current['score']:
                current['score'] = next_window['score']
                for key in (
                    'raw_score',
                    'rms_tangential_nT',
                    'start_idx',
                    'fcp_peak_ratio',
                    'fcp_band_gate',
                    'ellip_at_peak',
                    'wna_at_peak',
                    'ellip_gate',
                    'model_explain',
                    'model_diagnostics',
                ):
                    if key in next_window:
                        current[key] = next_window[key]
        else:
            # Save the current window and start a new one
            merged.append(_finalize_pf_wmed(current))
            current = next_window.copy()
            current['_pf_band_ratios'] = _window_ratios(current)

    merged.append(_finalize_pf_wmed(current))
    return merged


def filter_pf_search_candidates(candidates: list) -> tuple[list, list]:
    """Split search results by the P/F gate, failing open when unavailable.

    Only an explicit ``pf_gate_pass is False`` is rejected. Missing diagnostics,
    invalid spectra, and non-proton searches remain in the output.
    """
    kept = []
    rejected = []
    for candidate in candidates:
        diagnostics = candidate.get('model_diagnostics') or {}
        if diagnostics.get('pf_gate_pass') is False:
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return kept, rejected


def _result_mid_times(results: list) -> pd.DatetimeIndex:
    mids = []
    for r in results:
        t0 = pd.Timestamp(r['start'])
        t1 = pd.Timestamp(r['end'])
        mids.append(t0 + (t1 - t0) / 2)
    return pd.DatetimeIndex(mids)


def _format_trange_timestamp(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    if ts.microsecond:
        return ts.strftime('%Y-%m-%d/%H:%M:%S.%f')
    return ts.strftime('%Y-%m-%d/%H:%M:%S')


def _effective_trange_end(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    """
    Treat end=00:00:00 as an exclusive boundary when end > start.

    Example: ['2010-04-09/00:00:00', '2011-01-01/00:00:00']
    effectively ends at 2010-12-31/23:59:59.999999.
    """
    is_midnight = (
        end.hour == 0
        and end.minute == 0
        and end.second == 0
        and end.microsecond == 0
        and getattr(end, 'nanosecond', 0) == 0
    )
    if end > start and is_midnight:
        return end - pd.Timedelta(microseconds=1)
    return end


def normalize_trange_for_data_loading(trange: list[str]) -> list[str]:
    """Normalize trange so midnight end-boundary behaves as right-open for loading/chunking."""
    start = pd.to_datetime(trange[0], utc=True)
    end = pd.to_datetime(trange[1], utc=True)
    if end < start:
        raise ValueError(f"Invalid trange: end earlier than start: {trange}")

    end_effective = _effective_trange_end(start, end)
    return [
        _format_trange_timestamp(start),
        _format_trange_timestamp(end_effective),
    ]


def split_trange_by_month(trange: list[str]) -> list[list[str]]:
    """Split automatically by calendar month; each chunk covers one calendar month.

    An end time of 00:00:00 is treated as excluding that day. precompute_features loads the
    whole trange's per-window [F', T] features (ellip/wna_deg etc.) at once and
    concatenates them into one contiguous buffer; at that instant the all-per-day source
    arrays and the equally sized target buffer are both resident in memory (about
    F'×T×channels×4B each), so a six-month chunk with T≈1.5e7 OOMs. Monthly chunks cap T at ~2.6e6 and the peak at single-digit GiB. (v14 dropped coh/dop, cutting spectral channels 5→2, which lowers the peak further, but monthly chunking remains necessary.)
    """
    normalized_trange = normalize_trange_for_data_loading(trange)
    start = pd.to_datetime(normalized_trange[0], utc=True)
    end = pd.to_datetime(normalized_trange[1], utc=True)

    chunks: list[list[str]] = []
    chunk_start = start
    while chunk_start <= end:
        # First day of next month 00:00:00 minus 1 microsecond = last microsecond of this month (year rollover explicit, avoiding offset-anchor ambiguity).
        if chunk_start.month == 12:
            next_month_first = pd.Timestamp(year=chunk_start.year + 1, month=1, day=1, tz='UTC')
        else:
            next_month_first = pd.Timestamp(year=chunk_start.year, month=chunk_start.month + 1, day=1, tz='UTC')
        month_end = next_month_first - pd.Timedelta(microseconds=1)
        chunk_end = min(month_end, end)
        chunks.append([
            _format_trange_timestamp(chunk_start),
            _format_trange_timestamp(chunk_end),
        ])
        chunk_start = chunk_end + pd.Timedelta(microseconds=1)

    return chunks


def _dates_covered_by_trange(trange: list[str]) -> set[str]:
    start = pd.to_datetime(trange[0], utc=True).normalize()
    end = pd.to_datetime(trange[1], utc=True).normalize()
    covered_dates: set[str] = set()
    current = start
    while current <= end:
        covered_dates.add(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return covered_dates


def _candidate_dates(candidate: dict) -> set[str]:
    return _dates_covered_by_trange(candidate['trange'])


def save_trange_score_json(candidates: list, output_path: str, processed_trange: list[str]):
    """Update candidates.json incrementally by date: days that were re-run are replaced whole, other days are kept."""
    new_payload = [
        {
            'trange': [c['start'], c['end']],
            'score': float(c['score'])
        }
        for c in candidates
    ]

    processed_dates = _dates_covered_by_trange(processed_trange)
    payload = []

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_payload = json.load(f)
            if isinstance(existing_payload, list):
                for item in existing_payload:
                    if not isinstance(item, dict):
                        continue
                    trange = item.get('trange')
                    score = item.get('score')
                    if (
                        not isinstance(trange, list)
                        or len(trange) != 2
                        or not isinstance(score, (int, float))
                    ):
                        continue
                    if _candidate_dates(item).isdisjoint(processed_dates):
                        payload.append({
                            'trange': [trange[0], trange[1]],
                            'score': float(score),
                        })
        except Exception as e:
            print(f"Warning: failed to read existing candidates file {output_path}: {e}")

    payload.extend(new_payload)
    payload.sort(key=lambda item: (
        pd.to_datetime(item['trange'][0], utc=True),
        pd.to_datetime(item['trange'][1], utc=True),
    ))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)


def _save_single_candidate_window_plot(
    candidate: dict,
    score_mid_times_ns: np.ndarray,
    score_values: np.ndarray,
    data_dir: str,
    output_dir: str,
    threshold: float,
    window_minutes: int,
):
    """Render the window plot for one candidate event; returns {'path': str|None, 'error': str|None, 'start': str}."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = None
    start_text = str(candidate.get('start', '--'))
    try:
        t_start = pd.Timestamp(candidate['start'])
        t_end = pd.Timestamp(candidate['end'])
        mid = t_start + (t_end - t_start) / 2
        half_window = timedelta(minutes=float(window_minutes) / 2)
        win_start = mid - half_window
        win_end = mid + half_window

        vis_trange = [
            win_start.strftime('%Y-%m-%d/%H:%M:%S'),
            win_end.strftime('%Y-%m-%d/%H:%M:%S'),
        ]

        vis_analysis = DataAnalysis(vis_trange, data_dir=data_dir)
        df = vis_analysis.dataframe
        if df is None or df.empty:
            return {'path': None, 'error': None, 'start': start_text}

        times = pd.to_datetime(df['time_ut'])
        bt = df['Bt'].values.astype(np.float32)
        _, delta_bt = compute_delta_bt_from_total(bt, cutoff_freq=0.01, fs=1.0)
        # Proton cyclotron frequency from local |B| (convert nT -> T before calling).
        f_cp = get_gyro_frequency(np.abs(bt).astype(np.float64) * 1e-9)

        bfac_x = df['B_fac_x'].values
        bfac_y = df['B_fac_y'].values

        freqs_cwt, power_cwt, _ = vis_analysis.wavelet_analysis(num_freqs=64, if_normalize=False)
        pol_result = vis_analysis.polarization_analysis_on_xy(step_size=10, workers=1)
        ellip_signed = pol_result["ellip_signed"]
        freq_pol = pol_result["freq_pol"]
        window_centers = pol_result["window_center_times"]

        score_mask = (
            (score_mid_times_ns >= win_start.value)
            & (score_mid_times_ns <= win_end.value)
        )
        score_t = pd.to_datetime(score_mid_times_ns[score_mask])
        score_y = score_values[score_mask]

        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        ax0, ax1, ax2, ax3, ax4 = axes

        # Match _plot.py: reserve space on the right of each panel for its colorbar so the frames align.
        caxes = []
        for ax in axes:
            divider = make_axes_locatable(ax)
            caxes.append(divider.append_axes("right", size="2%", pad=0.1))
        cax0, cax1, cax2, cax3, cax4 = caxes
        cax0.axis('off')
        cax1.axis('off')
        cax4.axis('off')

        ax0.plot(times, delta_bt, color='black', linewidth=1.0)
        ax0.set_ylabel(r'$\delta B_T$ [nT]')
        ax0.grid(True, alpha=0.25)

        ax1.plot(times, bfac_x, color='tab:blue', linewidth=1.0, label=r'$\delta B_{\perp 1}$')
        ax1.plot(times, bfac_y, color='tab:green', linewidth=1.0, label=r'$\delta B_{\perp 2}$')
        ax1.set_ylabel(r'$\delta B_{\perp}$ [nT]')
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc='upper right', fontsize=8)

        pcm2 = ax2.pcolormesh(
            times,
            freqs_cwt,
            np.log10(power_cwt + 1e-5),
            shading='auto',
            cmap='jet'
        )
        ax2.plot(times, f_cp, color='black', linewidth=1.8, alpha=0.9)
        ax2.plot(times, f_cp, color='white', linewidth=1.0, linestyle='--', label='f_cp')
        ax2.set_yscale('log')
        ax2.set_ylim(0.01, 0.5)
        ax2.set_ylabel('Wavelet\nFreq [Hz]')
        ax2.legend(loc='upper right', fontsize=8)
        cbar2 = fig.colorbar(pcm2, cax=cax2)
        cbar2.set_label('log10 Power')

        wc_times = pd.to_datetime(window_centers)
        time_epoch_s = pd.DatetimeIndex(times).asi8.astype(np.float64) / 1e9
        wc_epoch_s = pd.DatetimeIndex(wc_times).asi8.astype(np.float64) / 1e9
        f_cp_wc = np.interp(wc_epoch_s, time_epoch_s, f_cp, left=f_cp[0], right=f_cp[-1])
        pcm3 = ax3.pcolormesh(
            wc_times,
            freq_pol,
            np.asarray(ellip_signed).T,
            shading='auto',
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
        )
        ax3.plot(wc_times, f_cp_wc, color='black', linewidth=1.8, alpha=0.9)
        ax3.plot(wc_times, f_cp_wc, color='white', linewidth=1.0, linestyle='--', label='f_cp')
        ax3.set_ylim(0.0, 0.5)
        ax3.set_ylabel('Ellipticity\nFreq [Hz]')
        ax3.legend(loc='upper right', fontsize=8)
        cbar3 = fig.colorbar(pcm3, cax=cax3)
        cbar3.set_label('Ellipticity')

        ax4.plot(score_t, score_y, color='red', linewidth=1.4, label='Score')
        ax4.axhline(threshold, color='gray', linestyle='--', linewidth=1.0, label=f'Th={threshold:.2f}')
        ax4.set_ylabel(r'$P_{ICW}$')
        ax4.set_xlabel('UTC Time')
        ax4.set_ylim(0.0, 1.05)
        ax4.grid(True, alpha=0.25)
        ax4.legend(loc='upper right', fontsize=8)

        for ax in axes:
            ax.set_xlim(win_start, win_end)

        fig.suptitle(
            f"ICW Candidate Window | center={mid.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
            f"score={candidate['score']:.3f}",
            fontsize=11
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        os.makedirs(output_dir, exist_ok=True)
        out_name = f"{mid.strftime('%y-%m-%d-%H-%M-%S')}.jpg"
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=200, format='jpg')
        return {'path': out_path, 'error': None, 'start': start_text}
    except Exception as e:
        return {'path': None, 'error': str(e), 'start': start_text}
    finally:
        if fig is not None:
            plt.close(fig)


def save_candidate_window_plots(
    results: list,
    candidates: list,
    data_dir: str,
    output_dir: str,
    threshold: float,
    window_minutes: int = 30,
    max_plot_workers: int | None = None,
):
    """Render window plots in parallel around each candidate midpoint (delta Bt / FAC xy / wavelet / ellipticity / probability)."""
    if not candidates:
        return []

    os.makedirs(output_dir, exist_ok=True)
    score_mid_times_ns = _result_mid_times(results).asi8
    score_values = np.array([r['score'] for r in results], dtype=float)
    saved_paths = []

    if max_plot_workers is None:
        cpu_count = os.cpu_count() or 1
        max_plot_workers = min(8, max(1, cpu_count - 1))
    max_plot_workers = max(1, int(max_plot_workers))

    run_parallel = max_plot_workers > 1 and len(candidates) > 1
    if run_parallel:
        print(f"Saving candidate plots in parallel (workers={max_plot_workers})...")
        with ProcessPoolExecutor(max_workers=max_plot_workers) as executor:
            plot_results = executor.map(
                _save_single_candidate_window_plot,
                candidates,
                repeat(score_mid_times_ns),
                repeat(score_values),
                repeat(data_dir),
                repeat(output_dir),
                repeat(float(threshold)),
                repeat(int(window_minutes)),
            )
            for item in tqdm(plot_results, total=len(candidates), desc="Saving candidate plots", leave=False, disable=not sys.stderr.isatty()):
                if item['error']:
                    print(f"  Window visualization failed ({item['start']}): {item['error']}")
                elif item['path']:
                    saved_paths.append(item['path'])
    else:
        print("Saving candidate plots...")
        for c in tqdm(candidates, total=len(candidates), desc="Saving candidate plots", leave=False, disable=not sys.stderr.isatty()):
            item = _save_single_candidate_window_plot(
                c,
                score_mid_times_ns,
                score_values,
                data_dir,
                output_dir,
                float(threshold),
                int(window_minutes),
            )
            if item['error']:
                print(f"  Window visualization failed ({item['start']}): {item['error']}")
            elif item['path']:
                saved_paths.append(item['path'])

    return saved_paths
