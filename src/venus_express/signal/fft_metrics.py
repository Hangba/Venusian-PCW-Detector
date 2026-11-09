"""Canonical FFT metrics for ICW manual review and sample-pool scoring."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from venus_express.physics.gyro import get_gyro_frequency


def moving_average_1d(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    w = max(1, min(int(window_size), values.size))
    if w == 1:
        return values
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    kernel = np.full(w, 1.0 / w, dtype=np.float64)
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _moving_average_2d(values: np.ndarray, window_size: int) -> np.ndarray:
    """Row-wise uniform moving average, identical per-row to moving_average_1d.

    Uses an edge-padded cumulative sum so all rows are smoothed in one shot.
    out[:, i] = mean(padded[:, i:i+w]) matches np.convolve(..., 'valid') with a
    uniform kernel (the kernel is symmetric, so reversal is a no-op).
    """
    values = np.asarray(values, dtype=np.float64)
    n_cols = values.shape[1]
    w = max(1, min(int(window_size), n_cols))
    if w == 1:
        return values
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    padded = np.pad(values, ((0, 0), (pad_left, pad_right)), mode="edge")
    csum = np.cumsum(padded, axis=1)
    csum = np.concatenate([np.zeros((values.shape[0], 1)), csum], axis=1)
    return (csum[:, w:] - csum[:, :-w]) / float(w)


def compute_perp_fft_peak_freq_batched(
    bx_windows: np.ndarray,
    by_windows: np.ndarray,
    *,
    fs: float = 1.0,
    pad_factor: int = 2,
    smooth_points: int = 4,
) -> np.ndarray:
    """Vectorized transverse FFT peak frequency for many equal-length windows.

    ``bx_windows``/``by_windows`` are [n_windows, window_len] arrays (one row per
    sliding window). Returns a [n_windows] array of peak frequencies, each the
    global argmax of the smoothed P_perp = PSD(bx) + PSD(by) over freq > 0 —
    bit-for-bit the same definition as ``compute_window_perp_peak_freq`` /
    ``compute_perp_fft_metrics`` (no fcp_band, no bz/bt), just batched into one
    rfft instead of one FFT per window.
    """
    bx = np.asarray(bx_windows, dtype=np.float64)
    by = np.asarray(by_windows, dtype=np.float64)
    if bx.ndim != 2 or bx.shape != by.shape:
        raise ValueError("bx_windows/by_windows must be equal-shaped 2D arrays")
    n_win, n = bx.shape
    if n < 16:
        raise ValueError("Window has too few samples for FFT metrics (need >=16)")

    nfft = int(n * max(1, int(pad_factor)))
    freq = np.fft.rfftfreq(nfft, d=1.0 / float(fs))

    def _psd(sig: np.ndarray) -> np.ndarray:
        sig = sig - np.nanmean(sig, axis=1, keepdims=True)
        sig = np.nan_to_num(sig, nan=0.0)
        spec = np.fft.rfft(sig, n=nfft, axis=1)
        psd = (np.abs(spec) ** 2) / (float(fs) * float(n))
        if nfft % 2 == 0:
            if psd.shape[1] > 2:
                psd[:, 1:-1] *= 2.0
        elif psd.shape[1] > 1:
            psd[:, 1:] *= 2.0
        return psd

    psd_x = _psd(bx)
    psd_y = _psd(by)

    keep = np.isfinite(freq) & (freq > 0.0)
    if not np.any(keep):
        raise ValueError("No positive FFT frequency bins")
    freq_keep = freq[keep]
    psd_perp = np.maximum(psd_x[:, keep] + psd_y[:, keep], 1e-30)
    psd_perp = np.maximum(_moving_average_2d(psd_perp, smooth_points), 1e-30)

    peak_idx = np.argmax(psd_perp, axis=1)
    return freq_keep[peak_idx].astype(np.float64)


def one_sided_psd(signal: np.ndarray, fs: float, pad_factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size < 2:
        raise ValueError("Signal too short for FFT PSD")
    sig = sig - np.nanmean(sig)
    sig = np.nan_to_num(sig, nan=0.0)
    nfft = int(sig.size * max(1, int(pad_factor)))
    freq = np.fft.rfftfreq(nfft, d=1.0 / float(fs))
    spec = np.fft.rfft(sig, n=nfft)
    psd = (np.abs(spec) ** 2) / (float(fs) * float(sig.size))
    if nfft % 2 == 0:
        if psd.size > 2:
            psd[1:-1] *= 2.0
    elif psd.size > 1:
        psd[1:] *= 2.0
    return freq.astype(np.float64), psd.astype(np.float64)


def compute_pf_band_peak_ratio(
    bfac_x: np.ndarray,
    bfac_y: np.ndarray,
    fcp_curve: np.ndarray,
    *,
    fs: float = 1.0,
    in_band: tuple[float, float] = (0.8, 1.2),
    out_percentile: float = 75.0,
    pad_factor: int = 2,
    smooth_points: int = 4,
) -> float:
    """Return the P/F spectral contrast for one model window.

    Definition frozen on 2026-07-29 after the catalogue P/F study: maximum
    transverse PSD in ``in_band * median(f_cp)`` divided by the requested
    percentile of every positive-frequency PSD bin outside that band. Search
    records this independently from the model probability and uses it to keep
    broad-band ULF candidates out of the human-review search output.
    """
    bx = np.asarray(bfac_x, dtype=np.float64).ravel()
    by = np.asarray(bfac_y, dtype=np.float64).ravel()
    fcp = np.asarray(fcp_curve, dtype=np.float64).ravel()
    n = min(bx.size, by.size, fcp.size)
    if n < 16:
        return float("nan")

    finite_field = np.isfinite(bx[:n]) & np.isfinite(by[:n])
    if finite_field.sum() < 16 or not finite_field.all():
        return float("nan")
    if max(float(np.std(bx[:n])), float(np.std(by[:n]))) <= np.finfo(float).eps:
        return float("nan")

    valid_fcp = fcp[:n][np.isfinite(fcp[:n]) & (fcp[:n] > 0.0)]
    if not valid_fcp.size:
        return float("nan")
    fcp_ref = float(np.median(valid_fcp))

    freq, psd_x = one_sided_psd(bx[:n], fs, pad_factor)
    _, psd_y = one_sided_psd(by[:n], fs, pad_factor)
    positive = np.isfinite(freq) & (freq > 0.0)
    if not positive.any():
        return float("nan")
    freq = freq[positive]
    power = np.maximum(psd_x[positive] + psd_y[positive], 1e-30)
    power = np.maximum(moving_average_1d(power, smooth_points), 1e-30)

    lo, hi = float(in_band[0]), float(in_band[1])
    relative_freq = freq / fcp_ref
    inside = (relative_freq >= lo) & (relative_freq <= hi)
    outside = ~inside
    if not inside.any() or not outside.any():
        return float("nan")
    reference = float(np.percentile(power[outside], float(out_percentile)))
    if not np.isfinite(reference) or reference <= 0.0:
        return float("nan")
    return float(np.max(power[inside]) / reference)


def compute_perp_fft_metrics(
    bfac_x: np.ndarray,
    bfac_y: np.ndarray,
    bfac_z: Optional[np.ndarray] = None,
    bt: Optional[np.ndarray] = None,
    *,
    fs: float = 1.0,
    pad_factor: int = 2,
    smooth_points: int = 4,
    fcp_band: Optional[tuple] = None,
) -> Dict[str, Any]:
    """Compute canonical tangential FFT metrics over exactly one window.

    This is the manual-review/sample-pool definition:
    P_perp = PSD(B_fac_x) + PSD(B_fac_y), DC removed by using only freq > 0,
    optional zero-padding for display density, light moving-average smoothing.

    Peak selection:
    - ``fcp_band=None`` (default): unconstrained GLOBAL P_perp peak (legacy).
    - ``fcp_band=(lo, hi)`` with computable f_cp: peak is the argmax WITHIN
      [lo, hi]*f_cp. This is the physically-correct ICW f_max — only the
      cyclotron-band peak matters, so a stronger low-frequency (red-noise /
      background) peak cannot masquerade as f_max and drag ellip/WNA@f_max to
      the wrong frequency. Falls back to the global peak if f_cp is unavailable
      or the band is empty.
    """
    bx = np.asarray(bfac_x, dtype=np.float64).ravel()
    by = np.asarray(bfac_y, dtype=np.float64).ravel()
    n = int(min(bx.size, by.size))
    if bfac_z is not None:
        bz = np.asarray(bfac_z, dtype=np.float64).ravel()
        n = min(n, int(bz.size))
    else:
        bz = None
    if bt is not None:
        bt_arr = np.asarray(bt, dtype=np.float64).ravel()
        n = min(n, int(bt_arr.size))
    else:
        bt_arr = None
    if n < 16:
        raise ValueError("Window has too few samples for FFT metrics (need >=16)")

    bx = bx[:n]
    by = by[:n]
    freq, psd_x = one_sided_psd(bx, fs, pad_factor)
    _, psd_y = one_sided_psd(by, fs, pad_factor)

    keep = np.isfinite(freq) & (freq > 0.0)
    if not np.any(keep):
        raise ValueError("No positive FFT frequency bins")
    freq = freq[keep]
    psd_perp = np.maximum(psd_x[keep] + psd_y[keep], 1e-30)
    psd_perp = np.maximum(moving_average_1d(psd_perp, smooth_points), 1e-30)

    psd_parallel = None
    transverse_ratio_at_peak = None
    if bz is not None:
        _, psd_z = one_sided_psd(bz[:n], fs, pad_factor)
        psd_parallel = np.maximum(psd_z[keep], 1e-30)
        psd_parallel = np.maximum(moving_average_1d(psd_parallel, smooth_points), 1e-30)

    # f_cp first (needed for optional band-limited peak selection).
    f_cp_ref = None
    if bt_arr is not None:
        f_cp_local = get_gyro_frequency(bt_arr[:n] * 1e-9)
        valid_fcp = f_cp_local[np.isfinite(f_cp_local) & (f_cp_local > 0.0)]
        if valid_fcp.size:
            f_cp_ref = float(np.nanmedian(valid_fcp))

    # Peak: band-limited to [lo,hi]*f_cp when requested and f_cp is available,
    # otherwise the legacy global argmax.
    peak_idx = int(np.argmax(psd_perp))
    if fcp_band is not None and f_cp_ref is not None and f_cp_ref > 0:
        lo, hi = float(fcp_band[0]) * f_cp_ref, float(fcp_band[1]) * f_cp_ref
        band_mask = (freq >= lo) & (freq <= hi)
        if np.any(band_mask):
            band_idx = np.flatnonzero(band_mask)
            peak_idx = int(band_idx[int(np.argmax(psd_perp[band_mask]))])
    peak_freq = float(freq[peak_idx])
    peak_psd = float(psd_perp[peak_idx])
    if psd_parallel is not None:
        denom = peak_psd + float(psd_parallel[peak_idx])
        transverse_ratio_at_peak = float(peak_psd / denom) if denom > 0 else None

    peak_fcp_ratio = (
        peak_freq / f_cp_ref if (f_cp_ref is not None and f_cp_ref > 0) else None
    )

    # Band-dominance: in-band peak P_perp vs the strongest OUT-of-band peak.
    # A genuine narrowband ICW line should tower over background / red-noise /
    # harmonics, not merely be the global max by a hair. Band fixed at
    # [0.8,1.2]*f_cp (the ICW criterion band). ratio = in-band max / out max.
    band_dominance_ratio = None
    in_band_peak_psd = None
    out_of_band_peak_psd = None
    if f_cp_ref is not None and f_cp_ref > 0:
        blo, bhi = 0.8 * f_cp_ref, 1.2 * f_cp_ref
        in_mask = (freq >= blo) & (freq <= bhi)
        out_mask = ~in_mask  # freq already restricted to >0 (DC removed)
        if np.any(in_mask) and np.any(out_mask):
            in_band_peak_psd = float(np.max(psd_perp[in_mask]))
            out_of_band_peak_psd = float(np.max(psd_perp[out_mask]))
            band_dominance_ratio = in_band_peak_psd / max(out_of_band_peak_psd, 1e-30)

    return {
        "freq": freq,
        "psd_perp": psd_perp,
        "psd_parallel": psd_parallel,
        "peak_freq": peak_freq,
        "peak_psd": peak_psd,
        "f_cp_ref": f_cp_ref,
        "peak_fcp_ratio": peak_fcp_ratio,
        "transverse_ratio_at_peak": transverse_ratio_at_peak,
        "band_dominance_ratio": band_dominance_ratio,
        "in_band_peak_psd": in_band_peak_psd,
        "out_of_band_peak_psd": out_of_band_peak_psd,
    }


def best_window_band_dominance(
    bfac_x: np.ndarray,
    bfac_y: np.ndarray,
    bt: np.ndarray,
    *,
    fs: float = 1.0,
    window_size: int = 80,
    step_size: int = 20,
) -> float:
    """Max over 80s windows of (in-band peak P_perp / out-of-band peak P_perp).

    The band-dominance ratio is meaningful only on a single clean window: over a
    long merged segment, frequency drift / harmonics / multi-window averaging
    spread power and the ratio collapses toward 1 even for real ICW (measured
    ~2.6x), whereas a clean 80s window of a real event reaches ~17x. So an event
    qualifies if AT LEAST ONE of its windows has a strongly dominant in-band
    peak. Returns the best (max) window ratio; 0.0 if no window is computable.
    """
    bx = np.asarray(bfac_x, dtype=np.float64).ravel()
    by = np.asarray(bfac_y, dtype=np.float64).ravel()
    bta = np.asarray(bt, dtype=np.float64).ravel()
    n = min(bx.size, by.size, bta.size)
    if n < 16:
        return 0.0
    best = 0.0
    starts = list(range(0, n - window_size + 1, step_size)) or [0]
    for s in starts:
        e = min(s + window_size, n)
        if e - s < 16:
            continue
        try:
            m = compute_perp_fft_metrics(
                bx[s:e], by[s:e], None, bta[s:e],
                fs=fs, pad_factor=2, smooth_points=4,
            )
        except ValueError:
            continue
        d = m.get("band_dominance_ratio")
        if d is not None and d > best:
            best = float(d)
    return best
