"""Original mission-grid batched feature calculation and parent merging."""

import numpy as np
from scipy.fft import rfft, rfftfreq
from venus_express.icw_inference import band_normalized_psd_log


def profiles(f, indices):
    """Batched scipy FFT with original float32 arithmetic; verified vs scalar."""
    ix = np.asarray(indices)[:, None] + np.arange(80)[None, :]
    bx = np.asarray(f["bfac_x"])[ix]
    by = np.asarray(f["bfac_y"])[ix]
    fc = np.asarray(f["fcp_curve"], np.float32)[ix]
    el = np.asarray(f["ellip"], np.float32)[:, ix].transpose(1, 0, 2)
    wn = np.asarray(f["wna_deg"], np.float32)[:, ix].transpose(1, 0, 2)
    px = rfft(bx - bx.mean(1, keepdims=True), n=160, axis=1)
    py = rfft(by - by.mean(1, keepdims=True), n=160, axis=1)
    power = (abs(px) ** 2 + abs(py) ** 2).astype(np.float32)
    freq = rfftfreq(160, d=1.0).astype(np.float32)
    med = np.nanmedian(np.where(fc > 0, fc, np.nan), axis=1)
    band = (freq[None, :] >= 0.6 * med[:, None]) & (freq[None, :] <= 1.4 * med[:, None])
    peak = np.where(band, power, -np.inf).max(1)
    peak = np.where(band.any(1), peak, power.max(1))
    # Scalar helper's Python float is cast by NumPy to the same float32 dtype.
    lg = band_normalized_psd_log(power, np.maximum(peak, 1e-30))
    lg = np.stack([np.interp(f["freq_pol"], freq, row).astype(np.float32) for row in lg])
    dist = (f["freq_pol"][None, :, None] - fc[:, None, :]) / np.maximum(abs(fc[:, None, :]), 1e-6)
    full = np.stack(
        [
            np.broadcast_to(lg[:, :, None], el.shape),
            el,
            wn / np.float32(90),
            np.nan_to_num(dist, nan=0).astype(np.float32),
        ],
        axis=1,
    )
    return full.mean(-1), fc


def group_arrays(starts, ends, values, cut):
    ix = np.flatnonzero(values >= cut)
    groups = []
    for i in ix:
        a, b = int(starts[i]), int(ends[i])
        if not groups or a - groups[-1]["end_s"] >= 60:
            groups.append(dict(start_s=a, end_s=b, windows=1))
        else:
            groups[-1]["end_s"] = max(b, groups[-1]["end_s"])
            groups[-1]["windows"] += 1
    return groups
