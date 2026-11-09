"""
Compact daily feature cache for ICW training and inference.

Cache file format:
- One file per day: YYYY-MM-DD.npz
- Stores only training/inference channels:
  - times_s
  - delta_bt
  - bfac_x
  - bfac_y
  - ellip
  - wna_deg   (v12: frequency-resolved WNA angle [F',T], quasi-parallel→0; on disk uint8 = angle×2)
  - fcp_curve
  - freq_pol
"""

import io
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import zstandard as zstd

from venus_express.analysis import DEFAULT_POL_WINDOW_SIZE, DataAnalysis
from venus_express.paths import CACHE_DIR, VEX_DATA_DIR
from venus_express.physics.gyro import get_gyro_frequency
from venus_express.signal.filters import compute_delta_bt_from_total

DEFAULT_NUM_FREQS = 100

# Lower bound of joint_profile ch0 = log10(PSD_perp / in-band peak), i.e. 60 dB
# below the f_cp-band peak. Measured over 2766 windows on four days spanning
# f_cp 0.060-0.147 Hz, the non-DC bins live in [-6.7, +2.8] with p1 = -3.07, so
# this floor clips 0.05 per mille of real bins while bounding the channel's
# dynamic range.
PSD_LOG_FLOOR = -6.0


def band_normalized_psd_log(psd: np.ndarray, peak_in_band) -> np.ndarray:
    """log10(PSD / in-band peak) for joint_profile ch0, floored and DC-repaired.

    The rfft is taken after removing the window mean, so PSD[f=0] is identically
    zero: the DC bin measures nothing. The previous ``log10(x + 1e-12)`` mapped
    it to a constant -12 while every other bin is O(1) -- an outlier ~5x outside
    the natural range, sitting on the first bin of the frequency axis. Because
    the conv stack has receptive-field radius 5 (kernels 5/5/3) and GroupNorm
    normalises across the whole frequency axis, that single bin pinned the
    attention-evidence maximum to bin 5 in 86% of windows regardless of f_cp,
    and perturbed the in-band evidence by ~1/3 of its between-window spread.
    Here the DC bin instead carries its neighbour's value.

    Args:
        psd: [..., F_fft] power spectrum, DC at index 0 of the last axis.
        peak_in_band: scalar, or one peak per row broadcastable to psd[..., 0].

    Returns:
        float32 array shaped like ``psd``.
    """
    psd = np.asarray(psd, dtype=np.float32)
    peak = np.asarray(peak_in_band, dtype=np.float32)
    if peak.ndim:
        peak = peak[..., None]
    out = np.log10(np.maximum(psd / peak, 10.0 ** PSD_LOG_FLOOR)).astype(np.float32)
    if out.shape[-1] > 1:
        out[..., 0] = out[..., 1]
    return out




# Daily feature cache schema version.
# v1: ellip is a single-channel 2D [F', T]
# v2: ellip upgraded to 3-channel 3D [3, F', T] (raw / |ellip| / relu(-ellip))
# v3: added STFT time-frequency maps (stft [2, F_stft, T], stft_freq [F_stft])
# v4: removed redundant wavelet_fcp_guide (time-independent, equals norm_freq_axis-1)
#     and the STFT branch (information covered by wavelet, -60 MB raw per file)
# v5: Phase 2 — ellip cut back from 3 channels to 1 (signed only);
#     added fcp_curve [T] (Hz, local proton gyrofrequency) and freq_pol [F'] (Hz, ellip frequency axis),
#     so the dataset can compute the fcp_distance(t, f) = (f - fcp(t)) / fcp(t) channel in place.
#     The abs/relu-neg channels are rebuilt on demand in dataset.__getitem__ (no longer cached).
# v7: removed wavelet. The Phase 6 model only uses the PSD profile, ellip and physical scalars;
#     the cache keeps only data that training/inference slice directly.
# v8: added bfac_x/bfac_y so the PSD profile can be computed directly from the cache;
#     inference/training no longer re-reads raw data after loading.
# v9: ICWNet rework — added polarization-criterion channels coh (coherence γ²)/dop (degree of polarization)/wna (wave normal angle)/
#     wna_reliability, all interpolated/broadcast to [F', T] with the same shape as ellip, for the
#     "polarization-spectrum joint profile". Also changed the window pol_window_size 80→160
#     (frequency resolution df 0.0063→0.0031 Hz, freq_pol bins doubled), giving the ICW line more bins around f_cp.
# v10: wrote the low-risk, bounded 2D model-input channels coh/dop/wna_cos/wna_reliability
#      to disk as float16; they are still converted back to float32 for training/inference after loading.
# v11: fft_peak_transverse_ratio / fft_perp_peak_freq now use an 80 s window +
#      band-limited argmax over [0.5,1.5]*f_cp (the old 160 s full-band bare argmax
#      latched onto low-frequency red noise, putting the ch6 transverse ratio at the wrong low frequency).
# v12: the WNA channel changed from a "fake 2D cos collapsed along frequency and broadcast"
#      to a true frequency-resolved angle wna_deg[F',T] (quasi-parallel→0°, perpendicular→90°).
#      Measured WNA at f_cp for real ICWs differs by 8-25° from the window-collapsed value; the old
#      broadcast convention kept the model from testing the "WNA≈0" criterion on the f_cp line. Also removed
#      the wna_reliability channel (8→7 channels). wna_deg is stored as uint8 = clip(angle*2,0,180)
#      and restored by ÷2 on load (0.5°/step). The native 41-bin WNA frequency axis is first
#      interpolated onto the 54-bin freq_pol axis, then to per-second.
# v13: ellip now stores the FULL band, and the full band is used for training and inference.
#      ellip also drops from float32 to float16 (range [-1,1] gives enough precision),
#      offsetting the size growth of full-band storage so the daily file does not grow.
#      coh/dop drop from float16 to uint8 (round(v*255), range [0,1], precision ~0.004),
#      saving another ~3.6 MB per day (same zstd level, unchanged rebuild speed).
# v14: removed the coh / dop / fft_peak_transverse_ratio fields (no longer computed or stored).
#      The model has not read them since the 4-channel joint_profile (see icw_dataset._build_joint_profile);
#      they only kept occupying disk: 14.5 MB→10.9 MB on-disk per day (−25%), 38 GB→~29 GB total.
#      ⚠️ All old v13 files are treated as old_format; _load_daily_features returns None on version
#      mismatch, and build_or_load rebuilds each day on demand (graceful degradation, not a hard
#      failure), so a full rebuild is not required first.
# Older-format files lack this key (or have a mismatched version) and are recognized as old_format
# in _load_daily_features and classify_existing_cache.
SCHEMA_VERSION = 14

# v9 canonical polarization window. The freq-bin count of every [F', T] array
# (ellip/coh/dop/wna/freq_pol) is a function of this window, NOT of
# SCHEMA_VERSION — so two files can share schema_version=9 yet have
# incompatible freq axes if they were built with different windows
# (80 -> 27 bins, 160 -> 54 bins). The loader validates the on-disk bin count
# against the configured window (see _load_daily_features) and any helper that
# builds cache must default to this value to avoid contaminating the shared
# cache dir with a stale layout.
# The canonical value now lives in venus_express.analysis (imported above and
# re-exported here, so `from venus_express.cache import DEFAULT_POL_WINDOW_SIZE`
# keeps working for cache_jobs.py et al.). Currently 160.

ZSTD_MAGIC = b"\x28\xB5\x2F\xFD"
DEFAULT_ZSTD_LEVEL = 3


def compute_window_perp_peak_freq(
    bfac_x: np.ndarray, bfac_y: np.ndarray, fs: float = 1.0, smooth: int = 3
) -> float:
    """Compatibility wrapper for the canonical manual-review FFT peak."""
    from venus_express.signal.fft_metrics import compute_perp_fft_metrics

    try:
        metrics = compute_perp_fft_metrics(
            bfac_x,
            bfac_y,
            fs=fs,
            pad_factor=2,
            smooth_points=4,
        )
        return float(metrics["peak_freq"])
    except Exception:
        return float("nan")


class ICWFeatureCache:
    """Daily/on-demand compact feature cache for model pipelines."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        num_freqs: int,
        pol_window_size: int = DEFAULT_POL_WINDOW_SIZE,
        pol_step_size: int = 20,
        bt_bg_cutoff_hz: float = 0.01,
        fs: float = 1.0,
        memory_cache_days: int = 64,
        preload_days_ahead: int = 10,
        daily_cache_workers: int | None = None,
        spectral_analysis_workers: int = 1,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ):
        self.data_dir = data_dir
        self.cache_dir_root = cache_dir
        self.num_freqs = num_freqs
        self.pol_window_size = pol_window_size
        self.pol_step_size = pol_step_size
        self.bt_bg_cutoff_hz = bt_bg_cutoff_hz
        self.fs = fs
        self.memory_cache_days = max(0, int(memory_cache_days))
        self.preload_days_ahead = max(0, int(preload_days_ahead))
        worker_default = min(4, os.cpu_count() or 1)
        self.daily_cache_workers = max(1, int(worker_default if daily_cache_workers is None else daily_cache_workers))
        self.spectral_analysis_workers = max(1, int(spectral_analysis_workers))
        self.progress_callback = progress_callback
        self.cache_dir = self.cache_dir_root
        self._daily_mem_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._daily_mem_cache_lock = threading.Lock()
        self._preload_inflight: set[str] = set()
        self._preload_inflight_lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

    def __getstate__(self):
        """
        Make the cache picklable for multiprocessing DataLoader workers.

        `threading.Lock` objects cannot be pickled under the spawn start method
        used on macOS / Python 3.13, so we drop the lock and recreate it after
        deserialization in each worker.
        """
        state = self.__dict__.copy()
        state["_daily_mem_cache_lock"] = None
        state["_preload_inflight_lock"] = None
        state["_preload_inflight"] = set()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._daily_mem_cache_lock = threading.Lock()
        self._preload_inflight_lock = threading.Lock()
        self._preload_inflight = set()

    def _remember_day(self, date_str: str, payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.memory_cache_days <= 0:
            return payload
        with self._daily_mem_cache_lock:
            self._daily_mem_cache[date_str] = payload
            self._daily_mem_cache.move_to_end(date_str)
            while len(self._daily_mem_cache) > self.memory_cache_days:
                self._daily_mem_cache.popitem(last=False)
        return payload

    def _get_remembered_day(self, date_str: str) -> Dict[str, np.ndarray] | None:
        if self.memory_cache_days <= 0:
            return None
        with self._daily_mem_cache_lock:
            payload = self._daily_mem_cache.get(date_str)
            if payload is not None:
                self._daily_mem_cache.move_to_end(date_str)
            return payload

    @staticmethod
    def _day_trange(date_str: str) -> List[str]:
        day_start = pd.Timestamp(f"{date_str} 00:00:00", tz="UTC")
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return [
            day_start.strftime("%Y-%m-%d/%H:%M:%S"),
            day_end.strftime("%Y-%m-%d/%H:%M:%S.%f"),
        ]

    def _date_cache_path(self, date_str: str) -> str:
        return os.path.join(self.cache_dir, f"{date_str}.npz")

    @staticmethod
    def _is_non_decreasing(values: np.ndarray) -> bool:
        values = np.asarray(values)
        return values.size < 2 or bool(np.all(values[:-1] <= values[1:]))

    @staticmethod
    def _detect_cache_format(cache_path: str) -> str:
        """Return 'zstd' | 'unknown' based on file header magic bytes."""
        try:
            with open(cache_path, "rb") as f:
                head = f.read(4)
        except Exception:
            return "unknown"
        if head.startswith(ZSTD_MAGIC):
            return "zstd"
        return "unknown"

    @staticmethod
    def _expected_ellip_freq_bins(window_size: int) -> int:
        """Match spectral polarization Welch frequency-bin count."""
        nperseg = int(max(window_size // 1.5, 32))
        if nperseg > window_size:
            nperseg = int(window_size)
        return int(nperseg // 2 + 1)

    @staticmethod
    def _to_unix_s(values) -> np.ndarray:
        dt_utc = pd.to_datetime(values, utc=True)
        ns = dt_utc.tz_convert("UTC").tz_localize(None).astype("datetime64[ns]").astype("int64")
        return np.asarray(ns, dtype=np.float64) / 1e9

    @staticmethod
    def _interp_ellipticity_to_samples(
        ellip_win: np.ndarray,
        window_centers_s: np.ndarray,
        sample_times_s: np.ndarray,
        fallback_freq_bins: int,
    ) -> np.ndarray:
        sample_times_s = np.asarray(sample_times_s, dtype=np.float64)
        if sample_times_s.size == 0:
            return np.zeros((fallback_freq_bins, 0), dtype=np.float32)

        if ellip_win is None or np.size(ellip_win) == 0 or len(window_centers_s) == 0:
            return np.zeros((fallback_freq_bins, sample_times_s.shape[0]), dtype=np.float32)

        ellip_win_t = np.asarray(ellip_win, dtype=np.float32).T  # [F, N_windows]
        center_ts = np.asarray(window_centers_s, dtype=np.float64)
        if center_ts.ndim != 1 or center_ts.size == 0:
            return np.zeros((ellip_win_t.shape[0], sample_times_s.shape[0]), dtype=np.float32)

        if not ICWFeatureCache._is_non_decreasing(center_ts):
            order = np.argsort(center_ts)
            center_ts = center_ts[order]
            ellip_win_t = ellip_win_t[:, order]

        ellip = np.zeros((ellip_win_t.shape[0], sample_times_s.shape[0]), dtype=np.float32)
        if center_ts.size == 1:
            ellip[:] = np.nan_to_num(ellip_win_t[:, :1], nan=0.0)
            return ellip

        for f_idx in range(ellip_win_t.shape[0]):
            row = np.nan_to_num(ellip_win_t[f_idx], nan=0.0)
            ellip[f_idx] = np.interp(
                sample_times_s,
                center_ts,
                row,
                left=row[0],
                right=row[-1],
            ).astype(np.float32)

        return ellip

    @staticmethod
    def _interp_wna_to_samples(
        wna_deg_win: np.ndarray,
        freq_wna: np.ndarray,
        freq_pol: np.ndarray,
        window_centers_s: np.ndarray,
        sample_times_s: np.ndarray,
    ) -> np.ndarray:
        """Frequency-resolved WNA angle → [len(freq_pol), T] in degrees.

        v12: replaces the old reliability-weighted collapse+broadcast. WNA is
        native [N_win, F_wna] on its OWN frequency axis (41 bins) which differs
        from freq_pol (54 bins), so we first interpolate each window's WNA
        profile onto freq_pol, then interpolate along time to per-second. NaNs
        (no reliable WNA) fill 90 deg = perpendicular = least ICW-like = safe.
        """
        sample_times_s = np.asarray(sample_times_s, dtype=np.float64)
        n_fp = int(np.size(freq_pol))
        if sample_times_s.size == 0 or n_fp == 0:
            return np.full((max(n_fp, 1), 0), 90.0, dtype=np.float32)

        wna = np.asarray(wna_deg_win, dtype=np.float32)
        centers = np.asarray(window_centers_s, dtype=np.float64)
        fw = np.asarray(freq_wna, dtype=np.float64)
        fp = np.asarray(freq_pol, dtype=np.float64)
        if wna.size == 0 or centers.size == 0 or fw.size == 0:
            return np.full((n_fp, sample_times_s.shape[0]), 90.0, dtype=np.float32)

        # Step 1: interpolate each window from freq_wna(41) onto freq_pol(54).
        # wna is [N_win, F_wna]; produce [N_win, F_pol].
        wna_on_fp = np.empty((wna.shape[0], n_fp), dtype=np.float32)
        for w_idx in range(wna.shape[0]):
            row = np.nan_to_num(wna[w_idx], nan=90.0)
            wna_on_fp[w_idx] = np.interp(fp, fw, row, left=row[0], right=row[-1])

        # Step 2: interpolate along time to per-second → [F_pol, T].
        wna_t = wna_on_fp.T  # [F_pol, N_win]
        if not ICWFeatureCache._is_non_decreasing(centers):
            order = np.argsort(centers)
            centers = centers[order]
            wna_t = wna_t[:, order]

        out = np.full((n_fp, sample_times_s.shape[0]), 90.0, dtype=np.float32)
        if centers.size == 1:
            out[:] = wna_t[:, :1]
            return np.clip(out, 0.0, 90.0)
        for f_idx in range(n_fp):
            row = wna_t[f_idx]
            out[f_idx] = np.interp(
                sample_times_s, centers, row, left=row[0], right=row[-1]
            ).astype(np.float32)
        return np.clip(out, 0.0, 90.0)

    @staticmethod
    def _moving_mean(values: np.ndarray, window_size: int) -> np.ndarray:
        if values.size == 0:
            return values.astype(np.float32, copy=False)

        window_size = max(1, min(int(window_size), values.size))
        pad_left = window_size // 2
        pad_right = window_size - 1 - pad_left
        kernel = np.full(window_size, 1.0 / window_size, dtype=np.float32)
        padded = np.pad(values.astype(np.float32, copy=False), (pad_left, pad_right), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _local_fcp_curve(df: pd.DataFrame) -> np.ndarray:
        bx = df["Bx"].to_numpy(dtype=np.float32, copy=False)
        by = df["By"].to_numpy(dtype=np.float32, copy=False)
        bz = df["Bz"].to_numpy(dtype=np.float32, copy=False)
        b_local = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        return get_gyro_frequency(b_local * 1e-9).astype(np.float32, copy=False)


    @staticmethod
    def _build_broadcast_curve_to_samples(
        per_window_values: np.ndarray,
        window_centers_s: np.ndarray,
        sample_times_s: np.ndarray,
        n_freq_bins: int,
        weights_per_window: np.ndarray | None = None,
    ) -> np.ndarray:
        """Collapse a [N_win, F] per-window field to one value per window, interp
        to per-second, then broadcast across all freq bins → [n_freq_bins, T].

        Used for WNA / WNA-reliability (user chose broadcast-to-all-freq): the
        per-window frequency profile is reduced to a single representative value
        via an optional weighted mean (reliability-weighted for WNA), so the
        model gets a frequency-flat WNA channel aligned to the ellip grid.
        """
        sample_times_s = np.asarray(sample_times_s, dtype=np.float64)
        if sample_times_s.size == 0:
            return np.zeros((n_freq_bins, 0), dtype=np.float32)
        vals = np.asarray(per_window_values, dtype=np.float32)
        centers = np.asarray(window_centers_s, dtype=np.float64)
        if vals.size == 0 or centers.size == 0:
            return np.zeros((n_freq_bins, sample_times_s.shape[0]), dtype=np.float32)
        if vals.ndim == 1:
            per_win_scalar = vals
        else:
            # reduce frequency axis to one representative value per window
            if weights_per_window is not None:
                w = np.asarray(weights_per_window, dtype=np.float32)
                w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
                num = np.nansum(np.nan_to_num(vals, nan=0.0) * w, axis=1)
                den = np.sum(w, axis=1)
                per_win_scalar = np.where(den > 1e-12, num / np.maximum(den, 1e-12),
                                          np.nanmean(np.nan_to_num(vals, nan=0.0), axis=1))
            else:
                per_win_scalar = np.nanmean(vals, axis=1)
        per_win_scalar = np.nan_to_num(per_win_scalar, nan=0.0).astype(np.float64)

        if centers.size == 1:
            row = np.full(sample_times_s.shape[0], per_win_scalar[0], dtype=np.float32)
        else:
            order = np.argsort(centers)
            c_sorted = centers[order]
            v_sorted = per_win_scalar[order]
            row = np.interp(sample_times_s, c_sorted, v_sorted,
                            left=v_sorted[0], right=v_sorted[-1]).astype(np.float32)
        return np.broadcast_to(row[np.newaxis, :], (n_freq_bins, row.shape[0])).astype(np.float32, copy=True)

    def _build_daily_features(self, date_str: str) -> Dict[str, np.ndarray]:
        analysis = DataAnalysis(self._day_trange(date_str), data_dir=self.data_dir)
        if analysis.dataframe is None or len(analysis.dataframe) == 0:
            raise ValueError(f"No data found for date {date_str} in {self.data_dir}")

        df = analysis.dataframe
        times = df["time_ut"].values
        times_s = self._to_unix_s(times)

        bt = df["Bt"].to_numpy(dtype=np.float32, copy=False)
        bx_fac = df["B_fac_x"].to_numpy(dtype=np.float32, copy=False)
        by_fac = df["B_fac_y"].to_numpy(dtype=np.float32, copy=False)
        # v14: B_fac_z is no longer read here -- its only consumer was the
        # fft_peak_transverse_ratio pass, removed with that field.
        f_cp_curve = self._local_fcp_curve(df)
        bt_bg, delta_bt = compute_delta_bt_from_total(
            bt,
            cutoff_freq=self.bt_bg_cutoff_hz,
            fs=self.fs,
        )
        bt_bg = bt_bg.astype(np.float32, copy=False)
        delta_bt = delta_bt.astype(np.float32, copy=False)

        spectral_result = analysis.polarization_analysis_on_xy(
            window_size=self.pol_window_size,
            step_size=self.pol_step_size,
            workers=self.spectral_analysis_workers,
        )
        expected_ellip_bins = self._expected_ellip_freq_bins(self.pol_window_size)
        center_idx = np.asarray(spectral_result["window_center_indices"], dtype=np.int64)
        freq_pol = np.asarray(spectral_result["freq_pol"], dtype=np.float32)
        center_times_s = (
            times_s[np.clip(center_idx, 0, max(len(times_s) - 1, 0))]
            if len(times_s)
            else np.array([], dtype=np.float64)
        )
        ellip = self._interp_ellipticity_to_samples(
            spectral_result["ellip_signed"],
            center_times_s,
            times_s,
            fallback_freq_bins=expected_ellip_bins,
        )
        # Store full-band ellipticity for all consumers.

        # v14: coherence (γ²) and degree-of-polarization are no longer cached.
        # The model dropped them from joint_profile, so computing and storing
        # them only cost disk (~3.6 MB/day on-disk). analysis.py still returns
        # them for callers that recompute polarization directly (e.g. export.py
        # paper figures), which never read them from this cache.

        # v12: wave-normal angle (WNA), frequency-resolved. Native [N_win, F_wna]
        # on its own 41-bin axis; interpolate onto freq_pol(54) then per-second,
        # keep DEGREES (quasi-parallel→0). Replaces the old reliability-weighted
        # collapse+broadcast that hid the f_cp-line WNA. reliability dropped.
        wna_result = analysis.propagation_angle_analysis(
            window_size=self.pol_window_size,
            step_size=self.pol_step_size,
            workers=self.spectral_analysis_workers,
        )
        wna_center_idx = np.asarray(wna_result.get("window_center_indices", center_idx), dtype=np.int64)
        wna_center_times_s = (
            times_s[np.clip(wna_center_idx, 0, max(len(times_s) - 1, 0))]
            if len(times_s) else np.array([], dtype=np.float64)
        )
        wna_deg_win = np.asarray(wna_result.get("wna_deg", np.array([])), dtype=np.float32)
        freq_wna = np.asarray(wna_result.get("freq_wna", np.array([])), dtype=np.float32)
        wna_deg = self._interp_wna_to_samples(
            wna_deg_win, freq_wna, freq_pol, wna_center_times_s, times_s,
        )

        # v14: fft_peak_transverse_ratio (old ch5) removed together with its
        # 80s-window FFT pass, so a rebuild no longer pays for it at all.

        return {
            "times_s": np.asarray(times_s, dtype=np.float64),
            "delta_bt": delta_bt.astype(np.float32, copy=False),
            "bfac_x": bx_fac.astype(np.float32, copy=False),
            "bfac_y": by_fac.astype(np.float32, copy=False),
            "ellip": ellip.astype(np.float32, copy=False),  # [F', T] signed
            "wna_deg": wna_deg.astype(np.float32, copy=False),       # [F', T] WNA angle (deg), frequency-resolved, quasi-parallel→0
            "fcp_curve": f_cp_curve.astype(np.float32, copy=False),  # [T] Hz
            "freq_pol": freq_pol.astype(np.float32, copy=False),     # [F'] Hz
        }

    def _save_daily_features(self, date_str: str, features: Dict[str, np.ndarray]) -> None:
        cache_path = self._date_cache_path(date_str)
        tmp_path = f"{cache_path}.tmp"
        arrays = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "times_s": np.asarray(features["times_s"], dtype=np.float64),
            "delta_bt": np.asarray(features["delta_bt"], dtype=np.float32),
            "bfac_x": np.asarray(features["bfac_x"], dtype=np.float32),
            "bfac_y": np.asarray(features["bfac_y"], dtype=np.float32),
            "ellip": np.asarray(features["ellip"], dtype=np.float16),
            # v12: WNA angle (deg) → uint8 = clip(angle*2, 0, 180); divide by 2 on load (0.5°/step)
            "wna_deg": np.clip(
                np.rint(np.asarray(features["wna_deg"], dtype=np.float32) * 2.0), 0, 180
            ).astype(np.uint8),
            "fcp_curve": np.asarray(features["fcp_curve"], dtype=np.float32),
            "freq_pol": np.asarray(features["freq_pol"], dtype=np.float32),
        }
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        payload = zstd.ZstdCompressor(level=DEFAULT_ZSTD_LEVEL).compress(buf.getvalue())
        with open(tmp_path, "wb") as f:
            f.write(payload)
        os.replace(tmp_path, cache_path)

    def _load_daily_features(self, date_str: str) -> Dict[str, np.ndarray] | None:
        cache_path = self._date_cache_path(date_str)
        if not os.path.exists(cache_path):
            return None

        required_keys = {
            "times_s",
            "delta_bt",
            "bfac_x",
            "bfac_y",
            "ellip",
            "wna_deg",
            "fcp_curve",
            "freq_pol",
        }

        try:
            with open(cache_path, "rb") as f:
                if f.read(4) != ZSTD_MAGIC:
                    return None
                f.seek(0)
                raw = zstd.ZstdDecompressor().decompress(f.read())
            cached_ctx = np.load(io.BytesIO(raw), allow_pickle=False)

            with cached_ctx as cached:
                if not required_keys.issubset(cached.files):
                    return None
                # Schema version check: reject if the key is absent or the version mismatches
                if "schema_version" not in cached.files:
                    return None
                try:
                    file_version = int(cached["schema_version"])
                except Exception:
                    return None
                if file_version != SCHEMA_VERSION:
                    return None
                features = {
                    "times_s": cached["times_s"].astype(np.float64, copy=False),
                    "delta_bt": cached["delta_bt"].astype(np.float32, copy=False),
                    "bfac_x": cached["bfac_x"].astype(np.float32, copy=False),
                    "bfac_y": cached["bfac_y"].astype(np.float32, copy=False),
                    "ellip": cached["ellip"].astype(np.float32, copy=False),
                    # v12: uint8 (angle×2) → angle (deg) via ÷2
                    "wna_deg": cached["wna_deg"].astype(np.float32) * 0.5,
                    "fcp_curve": cached["fcp_curve"].astype(np.float32, copy=False),
                    "freq_pol": cached["freq_pol"].astype(np.float32, copy=False),
                }
            times_s = features["times_s"]
            ellip_arr = features["ellip"]
            # The cached freq axis must match what THIS cache instance would
            # produce for its configured pol_window_size. A file built with a
            # different window (e.g. 80 -> 27 bins while we expect 160 -> 54)
            # is internally consistent but incompatible with freshly-computed
            # neighbours; loading it leads to a "(27,T) into (54,T)" broadcast
            # crash when load_features_for_trange stitches days into one buffer.
            # Reject it -> build_or_load rebuilds the day at the correct layout.
            expected_bins = self._expected_ellip_freq_bins(self.pol_window_size)
            if (
                features["delta_bt"].shape != times_s.shape
                or features["bfac_x"].shape != times_s.shape
                or features["bfac_y"].shape != times_s.shape
                or ellip_arr.ndim != 2  # v5+: single channel [F', T]
                or ellip_arr.shape[0] != expected_bins  # Frequency axis must match the current window
                or ellip_arr.shape[1] != times_s.size
                or features["fcp_curve"].shape != times_s.shape
                or features["freq_pol"].ndim != 1
                or features["freq_pol"].size != ellip_arr.shape[0]
            ):
                return None
            return features
        except Exception:
            return None

    def build_or_load_daily_features(self, date_str: str) -> Dict[str, np.ndarray]:
        remembered = self._get_remembered_day(date_str)
        if remembered is not None:
            return remembered

        features = self._load_daily_features(date_str)
        if features is not None:
            return self._remember_day(date_str, features)

        features = self._build_daily_features(date_str)
        self._save_daily_features(date_str, features)
        return self._remember_day(date_str, features)

    def _load_daily_features_or_missing(self, date_str: str) -> tuple[str, Dict[str, np.ndarray] | None, bool]:
        try:
            return date_str, self.build_or_load_daily_features(date_str=date_str), False
        except ValueError as e:
            if "No data found for date" in str(e):
                return date_str, None, True
            raise

    def _emit_progress(self, phase: str, done: int, total: int, date_str: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(phase, int(done), int(total), date_str)
        except Exception:
            pass

    def _preload_existing_daily_cache(self, date_str: str) -> tuple[str, bool]:
        if self._get_remembered_day(date_str) is not None:
            return date_str, True

        features = self._load_daily_features(date_str)
        if features is None:
            return date_str, False

        self._remember_day(date_str, features)
        return date_str, True

    def _schedule_preload_existing_cache(self, date_list: List[str]) -> None:
        """Warm future days in a daemon thread without delaying the caller."""
        if not date_list:
            return
        with self._preload_inflight_lock:
            pending = [
                date_str for date_str in date_list
                if date_str not in self._preload_inflight
                and self._get_remembered_day(date_str) is None
            ]
            self._preload_inflight.update(pending)
        if not pending:
            return

        def _run() -> None:
            try:
                max_workers = min(self.daily_cache_workers, len(pending))
                if max_workers > 1:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        list(executor.map(self._preload_existing_daily_cache, pending))
                else:
                    for date_str in pending:
                        self._preload_existing_daily_cache(date_str)
            finally:
                with self._preload_inflight_lock:
                    self._preload_inflight.difference_update(pending)

        threading.Thread(
            target=_run,
            name="icw-cache-preload",
            daemon=True,
        ).start()

    def load_features_for_trange(self, trange: List[str]) -> Dict[str, np.ndarray]:
        t_start = pd.to_datetime(trange[0], utc=True)
        t_end = pd.to_datetime(trange[1], utc=True)
        if t_end < t_start:
            raise ValueError("Invalid trange: end is earlier than start")

        missing_dates: List[str] = []
        date_list: List[str] = []
        current_date = t_start.normalize()
        end_date = t_end.normalize()
        while current_date <= end_date:
            date_list.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        if self.daily_cache_workers > 1 and len(date_list) > 1:
            max_workers = min(self.daily_cache_workers, len(date_list))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                daily_results = []
                for idx, result in enumerate(executor.map(self._load_daily_features_or_missing, date_list), start=1):
                    daily_results.append(result)
                    self._emit_progress("load", idx, len(date_list), result[0])
        else:
            daily_results = []
            for idx, date_str in enumerate(date_list, start=1):
                result = self._load_daily_features_or_missing(date_str)
                daily_results.append(result)
                self._emit_progress("load", idx, len(date_list), result[0])

        if self.preload_days_ahead > 0:
            preload_dates: List[str] = []
            preload_date = end_date + timedelta(days=1)
            preload_end_date = end_date + timedelta(days=self.preload_days_ahead)
            while preload_date <= preload_end_date:
                preload_dates.append(preload_date.strftime("%Y-%m-%d"))
                preload_date += timedelta(days=1)

            self._schedule_preload_existing_cache(preload_dates)

        chunks = []
        for date_str, features, is_missing in daily_results:
            if is_missing:
                missing_dates.append(date_str)
            elif features is not None:
                chunks.append(features)
        # `daily_results` also holds (date_str, features, is_missing) tuples;
        # chunks already holds only the features references, so drop
        # daily_results immediately to avoid a second strong reference to each
        # daily ndarray during multi-day inference. The LRU still holds features
        # (up to memory_cache_days), so no cache entry is lost.
        del daily_results

        if missing_dates:
            print("[ICWFeatureCache] Missing daily data skipped: " + ", ".join(missing_dates))

        if not chunks:
            raise ValueError(
                "No available daily data in trange "
                f"{trange}; missing dates: {', '.join(missing_dates) if missing_dates else 'none'}"
            )

        start_s = float(t_start.timestamp())
        end_s = float(t_end.timestamp())

        if len(chunks) == 1:
            # Single-day fast path: mask the daily arrays directly, avoiding a full-range buffer.
            daily = chunks[0]
            chunks.clear()
            times_s_full = daily["times_s"]
            mask = (times_s_full >= start_s) & (times_s_full <= end_s)
            if not np.any(mask):
                raise ValueError(f"No cached samples in trange {trange}")
            return {
                "times_s": times_s_full[mask],
                "delta_bt": daily["delta_bt"][mask],
                "bfac_x": daily["bfac_x"][mask],
                "bfac_y": daily["bfac_y"][mask],
                "ellip": daily["ellip"][:, mask],
                "wna_deg": daily["wna_deg"][:, mask],
                "fcp_curve": daily["fcp_curve"][mask],
                "freq_pol": np.asarray(daily["freq_pol"], dtype=np.float32),
            }

        # Multi-day path: multi-year/multi-month inference is prone to OOM.
        # Use preallocation + streaming copy + mask-while-copying so
        # `np.concatenate` does not hold all daily arrays before allocating the
        # new buffer. Only one final trange-sized array is kept now.
        F_ell = int(chunks[0]["ellip"].shape[0])
        freq_pol = np.asarray(chunks[0]["freq_pol"], dtype=np.float32)

        # Estimate the number of samples inside trange (assume monotonic chunk
        # times; most chunks lie fully inside, and the two boundary chunks are
        # cropped). Allocate an upper bound first, then slice to the actual length.
        upper_T = sum(int(c["delta_bt"].shape[0]) for c in chunks)

        times_s_buf = np.empty(upper_T, dtype=np.float64)
        delta_bt_buf = np.empty(upper_T, dtype=np.float32)
        bfac_x_buf = np.empty(upper_T, dtype=np.float32)
        bfac_y_buf = np.empty(upper_T, dtype=np.float32)
        ellip_buf = np.empty((F_ell, upper_T), dtype=np.float32)
        wna_deg_buf = np.empty((F_ell, upper_T), dtype=np.float32)
        fcp_buf = np.empty(upper_T, dtype=np.float32)

        pos = 0
        out_of_order = False
        last_t = -np.inf
        for c in chunks:
            day_times = c["times_s"]
            day_mask = (day_times >= start_s) & (day_times <= end_s)
            n_kept = int(np.count_nonzero(day_mask))
            if n_kept == 0:
                continue

            times_s_buf[pos:pos + n_kept] = day_times[day_mask]
            delta_bt_buf[pos:pos + n_kept] = c["delta_bt"][day_mask]
            bfac_x_buf[pos:pos + n_kept] = c["bfac_x"][day_mask]
            bfac_y_buf[pos:pos + n_kept] = c["bfac_y"][day_mask]
            ellip_buf[:, pos:pos + n_kept] = c["ellip"][:, day_mask]
            wna_deg_buf[:, pos:pos + n_kept] = c["wna_deg"][:, day_mask]
            fcp_buf[pos:pos + n_kept] = c["fcp_curve"][day_mask]

            first_t = float(times_s_buf[pos])
            if first_t < last_t:
                out_of_order = True
            last_t = float(times_s_buf[pos + n_kept - 1])

            pos += n_kept
        chunks.clear()

        if pos == 0:
            raise ValueError(f"No cached samples in trange {trange}")

        times_s = times_s_buf[:pos]
        delta_bt = delta_bt_buf[:pos]
        bfac_x = bfac_x_buf[:pos]
        bfac_y = bfac_y_buf[:pos]
        ellip = ellip_buf[:, :pos]
        wna_deg = wna_deg_buf[:, :pos]
        fcp_curve = fcp_buf[:pos]

        if out_of_order and not self._is_non_decreasing(times_s):
            order = np.argsort(times_s)
            times_s = times_s[order]
            delta_bt = delta_bt[order]
            bfac_x = bfac_x[order]
            bfac_y = bfac_y[order]
            ellip = ellip[:, order]
            wna_deg = wna_deg[:, order]
            fcp_curve = fcp_curve[order]

        return {
            "times_s": times_s,
            "delta_bt": delta_bt,
            "bfac_x": bfac_x,
            "bfac_y": bfac_y,
            "ellip": ellip,
            "wna_deg": wna_deg,
            "fcp_curve": fcp_curve,
            "freq_pol": freq_pol,
        }


def build_or_load_daily_features(
    date_str: str,
    data_dir: str,
    num_freqs: int,
    cache_dir: str,
    pol_window_size: int = DEFAULT_POL_WINDOW_SIZE,
    pol_step_size: int = 20,
    bt_bg_cutoff_hz: float = 0.01,
    fs: float = 1.0,
    daily_cache_workers: int | None = None,
    spectral_analysis_workers: int = 1,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> Dict[str, np.ndarray]:
    cache = ICWFeatureCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        num_freqs=num_freqs,
        pol_window_size=pol_window_size,
        pol_step_size=pol_step_size,
        bt_bg_cutoff_hz=bt_bg_cutoff_hz,
        fs=fs,
        daily_cache_workers=daily_cache_workers,
        spectral_analysis_workers=spectral_analysis_workers,
        progress_callback=progress_callback,
    )
    return cache.build_or_load_daily_features(date_str=date_str)


def load_features_for_trange(
    trange: List[str],
    data_dir: str,
    num_freqs: int,
    cache_dir: str,
    pol_window_size: int = DEFAULT_POL_WINDOW_SIZE,
    pol_step_size: int = 20,
    bt_bg_cutoff_hz: float = 0.01,
    fs: float = 1.0,
    daily_cache_workers: int | None = None,
    spectral_analysis_workers: int = 1,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> Dict[str, np.ndarray]:
    cache = ICWFeatureCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        num_freqs=num_freqs,
        pol_window_size=pol_window_size,
        pol_step_size=pol_step_size,
        bt_bg_cutoff_hz=bt_bg_cutoff_hz,
        fs=fs,
        daily_cache_workers=daily_cache_workers,
        spectral_analysis_workers=spectral_analysis_workers,
        progress_callback=progress_callback,
    )
    return cache.load_features_for_trange(trange=trange)


def main() -> None:
    trange = ["2006-04-24/00:00:00", "2006-11-25/00:00:00"]
    data_dir = str(VEX_DATA_DIR)
    cache_dir = str(CACHE_DIR)
    num_freqs = 64
    pol_window_size = DEFAULT_POL_WINDOW_SIZE
    pol_step_size = 20
    bt_bg_cutoff_hz = 0.01
    fs = 1.0
    daily_cache_workers = 10
    spectral_analysis_workers = 8

    cache = ICWFeatureCache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        num_freqs=num_freqs,
        pol_window_size=pol_window_size,
        pol_step_size=pol_step_size,
        bt_bg_cutoff_hz=bt_bg_cutoff_hz,
        fs=fs,
        daily_cache_workers=daily_cache_workers,
        spectral_analysis_workers=spectral_analysis_workers,
    )

    print(f"Building/loading cache for trange: {trange}")
    features = cache.load_features_for_trange(trange=trange)
    print(
        "Cache ready | "
        f"samples={features['times_s'].shape[0]} | "
        f"ellip={features['ellip'].shape}"
    )


if __name__ == "__main__":
    main()
