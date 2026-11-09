"""Reconstruct the paper continuum subtraction from native FAC magnetic data."""

import numpy as np
from scipy.optimize import least_squares



def background(freq, psd, lo, hi):
    df = freq[1] - freq[0]
    left = (freq >= max(df, 0.5 * lo)) & (freq < lo)
    right = (freq > hi) & (freq <= min(0.5, 2 * hi))
    side = (left | right) & np.isfinite(psd) & (psd > 0)
    # If a guard removes a short spectrum's entire low side, retain it as
    # unresolved rather than silently changing the band or inventing samples.
    if side.sum() < 4:
        return None, dict(
            status="insufficient_sideband_bins",
            side_bins=int(side.sum()),
            left_bins=int(left.sum()),
            right_bins=int(right.sum()),
        )
    fref = np.sqrt(lo * hi)
    x = np.log(freq[side] / fref)
    y = np.log(psd[side])
    scale = float(np.median(psd[side]))
    weights = np.ones(len(x))
    # Balance left/right shoulders, so extra high-frequency bins do not dominate.
    if left.sum() and right.sum():
        weights = np.where(left[side], 1 / left.sum(), 1 / right.sum())
        weights /= weights.mean()
    beta = float(np.clip(-np.polyfit(x, y, 1)[0], 0, 4))
    start = np.log(scale)
    fun = lambda z: np.sqrt(weights) * (np.logaddexp(z[0] - z[1] * x, z[2]) - y)
    fits = []
    for frac in [0.01, 0.3, 0.8]:
        z0 = [start + np.log(1 - frac), beta, start + np.log(frac)]
        z = least_squares(
            fun,
            z0,
            bounds=([start - 25, 0, start - 25], [start + 20, 4, start + 20]),
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=180,
        )
        fits.append(z)
    result = min(fits, key=lambda z: z.cost)
    A = float(np.exp(result.x[0]))
    beta = float(result.x[1])
    N = float(np.exp(result.x[2]))
    pred = np.zeros_like(freq)
    positive = freq > 0
    pred[positive] = A * (freq[positive] / fref) ** (-beta) + N
    return pred, dict(
        status="ok",
        side_bins=int(side.sum()),
        left_bins=int(left.sum()),
        right_bins=int(right.sum()),
        background_amplitude_at_fref=A,
        background_slope_beta=beta,
        flat_psd_nT2_Hz=N,
        fref_hz=fref,
        fit_cost=float(result.cost),
        one_sided=not (left.sum() and right.sum()),
    )
