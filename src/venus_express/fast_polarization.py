"""
Batched polarization analysis using fully-vectorized STFT.

Replaces the per-window scipy.welch + csd loop (4165 outer windows × 3 scipy
calls = 12495 calls/day) with a single vectorized rfft over all inner segments.
Expected speedup: 5-10× for polarization phase.

Matches DataAnalysis.polarization_analysis_on_xy output structure including
the _pad_window_outputs boundary padding (so ellip_signed shape == original).
"""
import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, rfftfreq


def _pad_window_outputs(arrays: list, window_center_idx: np.ndarray, n_times: int):
    """Mirror DataAnalysis._pad_window_outputs: replicate first/last window at edges."""
    if len(window_center_idx) == 0 or n_times == 0:
        return arrays, window_center_idx

    if window_center_idx[0] > 0:
        arrays = [np.concatenate([arr[:1], arr], axis=0) for arr in arrays]
        window_center_idx = np.insert(window_center_idx, 0, 0)

    if window_center_idx[-1] < n_times - 1:
        arrays = [np.concatenate([arr, arr[-1:]], axis=0) for arr in arrays]
        window_center_idx = np.append(window_center_idx, n_times - 1)

    return arrays, window_center_idx


def batched_polarization_analysis(
    bx: np.ndarray,
    by: np.ndarray,
    window_size: int,
    step_size: int,
    fs: float = 1.0,
) -> dict:
    """
    Fully-batched polarization for all sliding windows.

    Behavior mimics scipy.signal.welch/csd with:
      - nperseg = max(window_size // 1.5, 32), clamped to window_size
      - noverlap = nperseg // 2
      - window='hann' (fftbins=True, matches scipy default)
      - detrend='constant' (subtract mean per segment)
      - scaling='density'
      - one-sided spectrum (bins doubled except DC and Nyquist)

    Output matches DataAnalysis.polarization_analysis_on_xy:
      - shapes include _pad_window_outputs boundary replication
      - dict keys: pol_ratio, ellip_signed, coh, power, freq_pol,
                   window_center_indices, window_center_times
    """
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    if bx.shape != by.shape or bx.ndim != 1:
        raise ValueError(f"bx and by must be 1D arrays of same length, got {bx.shape}, {by.shape}")

    n_samples = len(bx)
    if n_samples < window_size:
        return {
            'pol_ratio': np.array([]),
            'ellip_signed': np.array([]),
            'coh': np.array([]),
            'power': np.array([]),
            'freq_pol': None,
            'window_center_indices': np.array([], dtype=np.int64),
        }

    # Outer window parameters (matches DataAnalysis._build_window_params_list)
    outer_starts = np.arange(0, n_samples - window_size + 1, step_size, dtype=np.int64)
    n_outer = len(outer_starts)
    outer_centers = outer_starts + window_size // 2  # matches "(start + end) // 2" with end=start+window_size

    # Inner segment parameters (matches scipy.welch internals)
    nperseg = int(min(window_size, max(window_size // 1.5, 32)))
    noverlap = nperseg // 2
    hop = nperseg - noverlap
    n_inner = (window_size - nperseg) // hop + 1

    # Frequency axis
    freq_pol = rfftfreq(nperseg, 1.0 / fs).astype(np.float32)
    n_freq = len(freq_pol)

    # Hann window (scipy default = fftbins=True, periodic)
    win = get_window('hann', nperseg, fftbins=True).astype(np.float64)
    win_norm = float(np.sum(win * win))  # scipy.welch density scale denominator

    # Build [n_outer, n_inner, nperseg] tensor of bx/by segments using fancy indexing
    # segment_starts[o, i] = outer_starts[o] + i * hop
    inner_offsets = (np.arange(n_inner) * hop).astype(np.int64)        # [n_inner]
    seg_starts = outer_starts[:, None] + inner_offsets[None, :]        # [n_outer, n_inner]
    # absolute sample indices: [n_outer, n_inner, nperseg]
    sample_idx = seg_starts[..., None] + np.arange(nperseg, dtype=np.int64)[None, None, :]

    bx_segs = bx[sample_idx]  # [n_outer, n_inner, nperseg]
    by_segs = by[sample_idx]

    # Detrend (subtract mean per segment) — scipy detrend='constant'
    bx_segs = bx_segs - bx_segs.mean(axis=-1, keepdims=True)
    by_segs = by_segs - by_segs.mean(axis=-1, keepdims=True)

    # Apply Hann window
    bx_segs *= win
    by_segs *= win

    # Single batched rfft (n_outer * n_inner FFTs in one call)
    Fx = rfft(bx_segs, axis=-1)  # [n_outer, n_inner, n_freq] complex
    Fy = rfft(by_segs, axis=-1)

    # Power / cross-spectra per inner segment
    pxx = (Fx * Fx.conj()).real     # |Fx|^2
    pyy = (Fy * Fy.conj()).real     # |Fy|^2
    # csd(seg_by, seg_bx) in scipy returns <By* Bx> = mean(Fy.conj() * Fx)
    pxy = Fy.conj() * Fx            # complex [n_outer, n_inner, n_freq]

    # Average over inner segments
    S_xx = pxx.mean(axis=1)         # [n_outer, n_freq]
    S_yy = pyy.mean(axis=1)
    S_xy = pxy.mean(axis=1)

    # Density scaling (scipy welch): divide by fs * sum(win^2)
    density_scale = 1.0 / (fs * win_norm)
    S_xx *= density_scale
    S_yy *= density_scale
    S_xy *= density_scale

    # One-sided spectrum: double all bins except DC and Nyquist (when nperseg even)
    if nperseg % 2 == 0:
        S_xx[:, 1:-1] *= 2.0
        S_yy[:, 1:-1] *= 2.0
        S_xy[:, 1:-1] *= 2.0
    else:
        S_xx[:, 1:] *= 2.0
        S_yy[:, 1:] *= 2.0
        S_xy[:, 1:] *= 2.0

    # Stokes parameters (vectorized over all windows)
    I = S_xx + S_yy
    Q = S_xx - S_yy
    U = 2.0 * np.real(S_xy)
    V = 2.0 * np.imag(S_xy)

    # Coherence γ² = |S_xy|² / (S_xx * S_yy)
    denom_coh = S_xx * S_yy
    coh_val = np.where(denom_coh > 1e-30, np.clip(np.abs(S_xy) ** 2 / denom_coh, 0.0, 1.0), 0.0)

    # Polarization degree
    polarized_power = np.sqrt(Q ** 2 + U ** 2 + V ** 2)
    p_val = np.where(I > 1e-30, np.clip(polarized_power / I, 0.0, 1.0), 0.0)

    # Ellipticity (signed)
    sin_2beta = np.where(polarized_power > 1e-30, np.clip(V / polarized_power, -1.0, 1.0), 0.0)
    beta = 0.5 * np.arcsin(sin_2beta)
    e_val = np.tan(beta)

    # log10(I) for power channel
    power_log = np.log10(I)

    # Apply boundary padding to match DataAnalysis.polarization_analysis_on_xy
    padded_arrays, window_center_idx = _pad_window_outputs(
        [p_val.astype(np.float32), e_val.astype(np.float32),
         coh_val.astype(np.float32), power_log.astype(np.float32)],
        outer_centers.copy(),
        n_samples,
    )
    p_padded, e_padded, c_padded, pow_padded = padded_arrays

    return {
        'pol_ratio': p_padded,
        'ellip_signed': e_padded,
        'coh': c_padded,
        'power': pow_padded,
        'freq_pol': freq_pol,
        'window_center_indices': window_center_idx,
    }


def batched_propagation_analysis(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    window_size: int,
    step_size: int,
    fs: float = 1.0,
) -> dict:
    """Fully-batched wave-normal-angle (Means method), vectorized like
    batched_polarization_analysis. Replaces the per-window scipy.csd loop
    (n_outer x 6 csd calls/day ~ 25k calls) with one batched rfft.

    Replicates DataAnalysis._compute_propagation_angle exactly:
      - nperseg = min(window_size, max(window_size // 2, 32)), noverlap = nperseg//2
      - window='hann' (fftbins), detrend='constant', one-sided
      - kp = (Im<S_yz>, Im<S_zx>, Im<S_xy>); k_mag = |kp| + 1e-30
      - wna_deg = degrees(arccos(|kp_z / k_mag|))
      - reliability = k_mag / (Re S_xx + Re S_yy + Re S_zz + 1e-30)
    Output dict mirrors propagation_angle_analysis (minus window_center_times):
      wna_deg [n_win_padded, n_freq], wna_reliability [..], freq_wna [n_freq],
      window_center_indices [n_win_padded].
    """
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bz = np.asarray(bz, dtype=np.float64)
    if not (bx.shape == by.shape == bz.shape) or bx.ndim != 1:
        raise ValueError(f"bx/by/bz must be equal-length 1D arrays, got {bx.shape},{by.shape},{bz.shape}")

    n_samples = len(bx)
    if n_samples < window_size:
        return {
            'wna_deg': np.array([]),
            'wna_reliability': np.array([]),
            'freq_wna': None,
            'window_center_indices': np.array([], dtype=np.int64),
        }

    outer_starts = np.arange(0, n_samples - window_size + 1, step_size, dtype=np.int64)
    outer_centers = outer_starts + window_size // 2

    nperseg = int(min(window_size, max(window_size // 2, 32)))
    noverlap = nperseg // 2
    hop = nperseg - noverlap
    n_inner = (window_size - nperseg) // hop + 1

    freq_wna = rfftfreq(nperseg, 1.0 / fs).astype(np.float32)

    win = get_window('hann', nperseg, fftbins=True).astype(np.float64)
    win_norm = float(np.sum(win * win))

    inner_offsets = (np.arange(n_inner) * hop).astype(np.int64)
    seg_starts = outer_starts[:, None] + inner_offsets[None, :]
    sample_idx = seg_starts[..., None] + np.arange(nperseg, dtype=np.int64)[None, None, :]

    def _segs(sig):
        s = sig[sample_idx]                      # [n_outer, n_inner, nperseg]
        s = s - s.mean(axis=-1, keepdims=True)   # detrend constant
        s = s * win                              # hann
        return rfft(s, axis=-1)                  # [n_outer, n_inner, n_freq]

    Fx, Fy, Fz = _segs(bx), _segs(by), _segs(bz)

    density_scale = 1.0 / (fs * win_norm)

    def _cross(Fa, Fb):
        # scipy csd(a,b) = <A* B>? Actually csd(x,y)=mean(X conj * Y)? -> match _compute:
        # _compute uses csd(seg_bx, seg_bx) etc.; scipy csd(x,y)=mean(conj(X)*Y).
        c = (Fa.conj() * Fb).mean(axis=1) * density_scale  # [n_outer, n_freq]
        if nperseg % 2 == 0:
            c[:, 1:-1] *= 2.0
        else:
            c[:, 1:] *= 2.0
        return c

    S_xx = _cross(Fx, Fx)
    S_yy = _cross(Fy, Fy)
    S_zz = _cross(Fz, Fz)
    S_yz = _cross(Fy, Fz)
    S_zx = _cross(Fz, Fx)
    S_xy = _cross(Fx, Fy)

    kp_x = np.imag(S_yz)
    kp_y = np.imag(S_zx)
    kp_z = np.imag(S_xy)
    k_mag = np.sqrt(kp_x ** 2 + kp_y ** 2 + kp_z ** 2) + 1e-30
    k_hat_z = kp_z / k_mag
    wna_deg = np.degrees(np.arccos(np.clip(np.abs(k_hat_z), 0.0, 1.0)))

    total_power = np.real(S_xx) + np.real(S_yy) + np.real(S_zz)
    reliability = k_mag / (total_power + 1e-30)

    padded_arrays, window_center_idx = _pad_window_outputs(
        [wna_deg.astype(np.float32), reliability.astype(np.float32)],
        outer_centers.copy(),
        n_samples,
    )
    wna_padded, rel_padded = padded_arrays
    return {
        'wna_deg': wna_padded,
        'wna_reliability': rel_padded,
        'freq_wna': freq_wna,
        'window_center_indices': window_center_idx,
    }


if __name__ == "__main__":
    # Sanity check
    rng = np.random.default_rng(42)
