import numpy as np
import pandas as pd
import pywt

from venus_express.dataset import VenusExpressDataset
from venus_express.fast_polarization import batched_polarization_analysis
from venus_express.paths import VEX_DATA_DIR
from venus_express.physics.gyro import get_gyro_frequency

# Canonical polarization-analysis STFT window (seconds @ 1 Hz == samples).
# v9 raised this 80 -> 160 (frequency resolution df halved, freq_pol bins
# 27 -> 54). This is the SINGLE SOURCE OF TRUTH for every real-time
# polarization caller (UI heatmaps, FFT overlays, exports, GUI tools);
# cache.py re-exports it. NOTE: this is the polarization STFT window, NOT the
# ML sample window — the model time axis is fixed at 80 in icw_dataset /
# icw_inference / icw_model and must stay 80.
DEFAULT_POL_WINDOW_SIZE = 160

class DataAnalysis():
    def __init__(self, trange, data_dir = str(VEX_DATA_DIR), output_dir="output"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.trange = trange
        self.dataset = VenusExpressDataset(data_dir, trange)

        # Convert coordinates first (filtering must run on global data, then crop)
        self.dataset.tofac()

        # Crop the dataframe to the time range
        t_start = pd.to_datetime(trange[0], utc=True)
        t_end = pd.to_datetime(trange[1], utc=True)
        mask = (self.dataset.dataframe['time_ut'] >= t_start) & (self.dataset.dataframe['time_ut'] <= t_end)
        self.dataframe = self.dataset.dataframe[mask].reset_index(drop=True)

        self.fs = 1.0  # Sampling frequency in Hz
        self.dt = 1.0 / self.fs  # Sampling interval in seconds
        self._spectral_result_cache = {}

    @staticmethod
    def _pad_window_outputs(arrays, window_center_idx, n_times):
        if len(window_center_idx) == 0 or n_times == 0:
            return arrays, window_center_idx

        if window_center_idx[0] > 0:
            arrays = [np.concatenate([arr[:1], arr], axis=0) for arr in arrays]
            window_center_idx = np.insert(window_center_idx, 0, 0)

        if window_center_idx[-1] < n_times - 1:
            arrays = [np.concatenate([arr, arr[-1:]], axis=0) for arr in arrays]
            window_center_idx = np.append(window_center_idx, n_times - 1)

        return arrays, window_center_idx

    @staticmethod
    def _run_window_jobs(func, window_params_list, workers):
        # workers kept for API compatibility but unused: per-window work is too
        # small (~80 samples) for thread pool overhead to be worthwhile.
        # Sequential execution measured ~equal-or-faster than ThreadPool(4/8).
        return [func(params) for params in window_params_list]

    def _build_window_params_list(self, window_size, step_size):
        B_fac = self.dataframe[['B_fac_x', 'B_fac_y', 'B_fac_z']].to_numpy(copy=False).T
        starts = np.arange(0, len(B_fac[0]) - window_size + 1, step_size, dtype=np.int64)
        return [
            (
                start,
                start + window_size,
                B_fac[0][start:start + window_size],
                B_fac[1][start:start + window_size],
                B_fac[2][start:start + window_size],
            )
            for start in starts
        ]

    @staticmethod
    def _normalize_tf_by_local_fcp(spec, freq_axis, f_cp_local, num_bins=None, spacing="log"):
        spec = np.asarray(spec)
        freq_axis = np.asarray(freq_axis, dtype=np.float64)
        f_cp_local = np.asarray(f_cp_local, dtype=np.float64)

        if spec.ndim != 2 or freq_axis.ndim != 1 or spec.shape[0] != freq_axis.size or spec.shape[1] != f_cp_local.size:
            raise ValueError("spec/freq_axis/f_cp_local dimensions do not match for normalization")

        positive_mask = freq_axis > 0
        if not np.any(positive_mask):
            return freq_axis.copy(), spec.astype(np.float32, copy=True)
        if not np.all(positive_mask):
            freq_axis = freq_axis[positive_mask]
            spec = spec[positive_mask]

        valid_fcp = f_cp_local[f_cp_local > 0]
        if valid_fcp.size == 0 or freq_axis.size == 0:
            return freq_axis.copy(), spec.astype(np.float32, copy=True)

        num_bins = int(freq_axis.size if num_bins is None else max(1, num_bins))
        order = np.argsort(freq_axis)
        freq_sorted = freq_axis[order]
        spec_sorted = spec[order]

        f_norm_min = float(freq_sorted[0] / np.max(valid_fcp))
        f_norm_max = float(freq_sorted[-1] / np.min(valid_fcp))
        if f_norm_min <= 0 or f_norm_max <= 0 or not np.isfinite(f_norm_min) or not np.isfinite(f_norm_max):
            return freq_sorted.copy(), spec_sorted.astype(np.float32, copy=True)

        if spacing == "linear":
            f_norm_axis = np.linspace(f_norm_min, f_norm_max, num=num_bins, dtype=np.float64)
        else:
            f_norm_axis = np.geomspace(f_norm_min, f_norm_max, num=num_bins, dtype=np.float64)

        target_freqs = f_norm_axis[:, np.newaxis] * f_cp_local[np.newaxis, :]
        idx = np.searchsorted(freq_sorted, target_freqs)
        idx_clipped = np.clip(idx, 1, len(freq_sorted) - 1)

        x0 = freq_sorted[idx_clipped - 1]
        x1 = freq_sorted[idx_clipped]
        weight = np.divide(target_freqs - x0, x1 - x0, out=np.zeros_like(target_freqs), where=np.abs(x1 - x0) > 1e-30)

        time_idx = np.arange(spec.shape[1])
        y0 = spec_sorted[idx_clipped - 1, time_idx]
        y1 = spec_sorted[idx_clipped, time_idx]
        spec_norm = y0 + weight * (y1 - y0)

        out_of_bounds = (target_freqs < freq_sorted[0]) | (target_freqs > freq_sorted[-1]) | (f_cp_local[np.newaxis, :] <= 0)
        spec_norm[out_of_bounds] = np.nan

        return f_norm_axis, spec_norm.astype(np.float32, copy=False)

    def _compute_polarization(self, window_params):
        """Compute polarization quantities in the XY tangential plane."""
        from scipy.signal import csd, welch

        start, end, seg_bx, seg_by, seg_bz = window_params
        window_len = len(seg_bx)

        pol_nperseg = min(window_len, max(window_len // 1.5, 32))
        pol_noverlap = pol_nperseg // 2
        pol_kw = dict(fs=self.fs, window='hann', nperseg=pol_nperseg, noverlap=pol_noverlap, return_onesided=True)

        freq_pol, S_xx_avg = welch(seg_bx, **pol_kw)
        _, S_yy_avg = welch(seg_by, **pol_kw)
        # csd(seg_by, seg_bx) returns <By* Bx> = <Bx By*>
        _, S_xy = csd(seg_by, seg_bx, **pol_kw)

        # Compute Stokes parameters (vectorized)
        I = S_xx_avg + S_yy_avg  # Total intensity
        Q = S_xx_avg - S_yy_avg
        U = 2.0 * np.real(S_xy)
        V = 2.0 * np.imag(S_xy)

        # Coherence γ² = |S_xy|² / (S_xx_avg * S_yy_avg)
        denom_coh = S_xx_avg * S_yy_avg
        coh_val = np.where(denom_coh > 1e-30, np.clip((np.abs(S_xy)**2) / denom_coh, 0.0, 1.0), 0.0)

        # Degree of polarization
        polarized_power = np.sqrt(Q**2 + U**2 + V**2)
        p_val = np.where(I > 1e-30, np.clip(polarized_power / I, 0.0, 1.0), 0.0)

        # Ellipticity
        sin_2beta = np.where(polarized_power > 1e-30,
                            np.clip(V / polarized_power, -1.0, 1.0),
                            0.0)
        beta = 0.5 * np.arcsin(sin_2beta)
        e_val = np.tan(beta)

        window_center = start + (end - start) // 2
        return p_val, e_val, coh_val, I, freq_pol, window_center

    def _compute_propagation_angle(self, window_params):
        """Compute the wave normal angle (WNA) and reliability via the Means method."""
        from scipy.signal import csd

        start, end, seg_bx, seg_by, seg_bz = window_params
        window_len = len(seg_bx)

        # Improve frequency resolution: use a longer spectral segment by default.
        # E.g. for window_len=80, nperseg rises from 16 to 40 and df from 0.0625 Hz to 0.025 Hz.
        wna_nperseg = min(window_len, max(window_len // 2, 32))
        wna_noverlap = wna_nperseg // 2
        wna_kw = dict(fs=self.fs, window='hann', nperseg=wna_nperseg, noverlap=wna_noverlap, return_onesided=True)

        freq_wna, S_xx = csd(seg_bx, seg_bx, **wna_kw)
        _, S_yy = csd(seg_by, seg_by, **wna_kw)
        _, S_zz = csd(seg_bz, seg_bz, **wna_kw)
        _, S_yz = csd(seg_by, seg_bz, **wna_kw)
        _, S_zx = csd(seg_bz, seg_bx, **wna_kw)
        _, S_xy_wna = csd(seg_bx, seg_by, **wna_kw)

        kp_x = np.imag(S_yz)
        kp_y = np.imag(S_zx)
        kp_z = np.imag(S_xy_wna)

        k_mag = np.sqrt(kp_x**2 + kp_y**2 + kp_z**2) + 1e-30

        k_hat_z = kp_z / k_mag
        wna_rad = np.arccos(np.clip(np.abs(k_hat_z), 0.0, 1.0))
        wna_deg = np.degrees(wna_rad)

        total_power = np.real(S_xx) + np.real(S_yy) + np.real(S_zz)
        reliability = k_mag / (total_power + 1e-30)

        window_center = start + (end - start) // 2
        return wna_deg, reliability, freq_wna, window_center

    @staticmethod
    def _empty_polarization_result():
        return {
            'pol_ratio': np.array([]),
            'ellip_signed': np.array([]),
            'coh': np.array([]),
            'power': np.array([]),
            'freq_pol': None,
            'window_center_indices': np.array([], dtype=np.int64),
            'window_center_times': np.array([]),
        }

    @staticmethod
    def _empty_propagation_angle_result():
        return {
            'wna_deg': np.array([]),
            'wna_reliability': np.array([]),
            'freq_wna': None,
            'window_center_indices': np.array([], dtype=np.int64),
            'window_center_times': np.array([]),
        }

    def polarization_analysis_on_xy(self, window_size=DEFAULT_POL_WINDOW_SIZE, step_size=20, workers=4):
        """Sliding-window polarization analysis on the FAC XY tangential components.

        Phase 3 optimization: uses batched_polarization_analysis (fully-vectorized
        STFT over all inner segments) instead of per-window scipy.welch/csd loop.
        Speedup: ~5-10× for polarization phase, numerically identical (max diff ~3e-8).
        """
        cache_key = ("polarization", window_size, step_size, workers)
        cached = self._spectral_result_cache.get(cache_key)
        if cached is not None:
            return cached

        times = self.dataframe['time_ut'].values
        bx = self.dataframe['B_fac_x'].to_numpy(dtype=np.float64, copy=False)
        by = self.dataframe['B_fac_y'].to_numpy(dtype=np.float64, copy=False)

        # Batched computation (replaces _run_window_jobs + _compute_polarization loop)
        result = batched_polarization_analysis(bx, by, window_size, step_size, fs=self.fs)

        if result['freq_pol'] is None:
            empty = self._empty_polarization_result()
            self._spectral_result_cache[cache_key] = empty
            return empty

        # Add window_center_times (batched version only returns indices)
        result['window_center_times'] = times[result['window_center_indices']]

        self._spectral_result_cache[cache_key] = result
        return result

    def propagation_angle_analysis(self, window_size=DEFAULT_POL_WINDOW_SIZE, step_size=20, workers=4):
        """Sliding-window WNA analysis on the three FAC components (batched, vectorized)."""
        cache_key = ("propagation_angle", window_size, step_size, workers)
        cached = self._spectral_result_cache.get(cache_key)
        if cached is not None:
            return cached

        from venus_express.fast_polarization import batched_propagation_analysis
        times = self.dataframe['time_ut'].values
        bx = self.dataframe['B_fac_x'].to_numpy(dtype=np.float64, copy=False)
        by = self.dataframe['B_fac_y'].to_numpy(dtype=np.float64, copy=False)
        bz = self.dataframe['B_fac_z'].to_numpy(dtype=np.float64, copy=False)
        result = batched_propagation_analysis(bx, by, bz, window_size, step_size, fs=1.0)
        result['window_center_times'] = times[result['window_center_indices']]
        self._spectral_result_cache[cache_key] = result
        return result

    def spectral_analysis_on_xy(self, window_size=DEFAULT_POL_WINDOW_SIZE, step_size=20, workers=4):
        """
        Legacy-compatible interface: returns the merged polarization and propagation-angle results.
        """
        cache_key = ("spectral_combined", window_size, step_size, workers)
        cached = self._spectral_result_cache.get(cache_key)
        if cached is not None:
            return cached

        pol_result = self.polarization_analysis_on_xy(window_size=window_size, step_size=step_size, workers=workers)
        wna_result = self.propagation_angle_analysis(window_size=window_size, step_size=step_size, workers=workers)
        result = {
            'pol_ratio': pol_result['pol_ratio'],
            'ellip_signed': pol_result['ellip_signed'],
            'coh': pol_result['coh'],
            'power': pol_result['power'],
            'freq_pol': pol_result['freq_pol'],
            'wna_deg': wna_result['wna_deg'],
            'wna_reliability': wna_result['wna_reliability'],
            'freq_wna': wna_result['freq_wna'],
            'window_center_indices': pol_result['window_center_indices'],
            'window_center_times': pol_result['window_center_times'],
        }
        self._spectral_result_cache[cache_key] = result
        return result

    def wavelet_analysis(self, num_freqs=200, if_normalize=False):
        # Wavelet analysis
        # if_normalize optionally normalizes frequencies to multiples of the local proton gyrofrequency, easing comparison of wave properties across time

        B_x = self.dataframe['B_fac_x'].values
        B_y = self.dataframe['B_fac_y'].values
        Bt = self.dataframe['Bt'].values  # Get the total field magnitude for the local proton gyrofrequency

        dt = self.dt
        fs = self.fs

        # Period range: shortest 2*dt (Nyquist), longest 1/4 of the total duration
        periods = np.geomspace(2 * dt, len(B_x) * dt / 4, num=num_freqs)
        # Corresponding frequencies (Hz)
        frequencies = 1.0 / periods
        # pywt scale parameter (for cmor1.5-1.0, scale = 1.0 / (freq * dt))
        # Note: this fixes the argument order in the reference code; pywt.cwt takes scales, not frequencies, as its second argument
        scales_cwt = 1.0 / (frequencies * dt)

        # Continuous wavelet transform (CWT); method='fft' speeds it up substantially
        W_x, freqs = pywt.cwt(B_x, scales_cwt, 'cmor1.5-1.0', sampling_period=dt, method='fft')
        W_y, _     = pywt.cwt(B_y, scales_cwt, 'cmor1.5-1.0', sampling_period=dt, method='fft')

        # Total tangential magnetic field power spectrum (time × frequency)
        power_transverse = np.abs(W_x)**2 + np.abs(W_y)**2

        # Wavelet cross-spectrum (for polarization handedness)
        cross_spectrum = W_x * np.conj(W_y)   # <W_x W_y^*>
        # The sign of the imaginary part sets handedness: negative = left-hand LHP (EMIC waves), positive = right-hand RHP
        polarization_sign = np.imag(cross_spectrum) / (np.abs(cross_spectrum) + 1e-30)

        if if_normalize:
            # Normalize the frequency axis by the local proton gyrofrequency
            f_cp_local = get_gyro_frequency(Bt * 1e-9)
            spec_norm_axis, power_norm = self._normalize_tf_by_local_fcp(
                power_transverse,
                freqs,
                f_cp_local,
                num_bins=num_freqs,
                spacing="log",
            )
            _, pol_sign_norm = self._normalize_tf_by_local_fcp(
                polarization_sign,
                freqs,
                f_cp_local,
                num_bins=num_freqs,
                spacing="log",
            )
            return spec_norm_axis, power_norm, pol_sign_norm

        return freqs, power_transverse, polarization_sign

    def stft_analysis(self, nperseg: int = 128, hop: int = 1):
        """
        Sliding-window STFT on the three FAC components, giving perp/parallel time-frequency power spectra.

        Args:
            nperseg: STFT window length (samples). At fs=1 Hz this is also the window in seconds. Default 128.
            hop: Sliding step (samples). hop=1 gives as many output columns as input samples,
                aligning the time axis with other daily-cache features.

        Returns:
            freq_stft: [F] linear frequency axis (Hz), length nperseg//2 + 1 (default 65)
            psd_perp: [F, T] = |FFT(B_fac_x)|² + |FFT(B_fac_y)|², T == len(dataframe)
            psd_parallel: [F, T] = |FFT(B_fac_z)|²

        Implementation notes:
        - Use scipy.signal.stft's zero-boundary mode so output length matches the input samples
        - Detrend (subtract the mean) per window, apply a Hann window, then rfft
        - Reuse self._spectral_result_cache for in-memory caching (same as wavelet/polarization)
        """
        cache_key = ("stft", int(nperseg), int(hop))
        cached = self._spectral_result_cache.get(cache_key)
        if cached is not None:
            return cached["freq"], cached["perp"], cached["parallel"]

        from scipy.signal import stft as scipy_stft

        if self.dataframe is None or len(self.dataframe) == 0:
            empty_freq = np.zeros(int(nperseg) // 2 + 1, dtype=np.float32)
            empty_psd = np.zeros((empty_freq.size, 0), dtype=np.float32)
            self._spectral_result_cache[cache_key] = {
                "freq": empty_freq, "perp": empty_psd, "parallel": empty_psd,
            }
            return empty_freq, empty_psd, empty_psd

        bx = self.dataframe['B_fac_x'].to_numpy(dtype=np.float64, copy=False)
        by = self.dataframe['B_fac_y'].to_numpy(dtype=np.float64, copy=False)
        bz = self.dataframe['B_fac_z'].to_numpy(dtype=np.float64, copy=False)
        T_in = bx.size

        kwargs = dict(
            fs=self.fs,
            nperseg=int(nperseg),
            noverlap=int(nperseg) - int(hop),
            window='hann',
            boundary='zeros',
            padded=True,
            return_onesided=True,
        )

        # Three independent STFTs
        f_axis, _, Zxx_x = scipy_stft(bx, **kwargs)
        _, _, Zxx_y = scipy_stft(by, **kwargs)
        _, _, Zxx_z = scipy_stft(bz, **kwargs)

        # Take the power spectrum and trim to T_in columns (zero boundary usually yields T_in+1)
        psd_x = (np.abs(Zxx_x) ** 2)[:, :T_in].astype(np.float32, copy=False)
        psd_y = (np.abs(Zxx_y) ** 2)[:, :T_in].astype(np.float32, copy=False)
        psd_z = (np.abs(Zxx_z) ** 2)[:, :T_in].astype(np.float32, copy=False)

        psd_perp = psd_x + psd_y
        psd_parallel = psd_z

        # Pad a possibly short tail (rare, when T_in is tiny) with the edge column
        if psd_perp.shape[1] < T_in:
            pad = T_in - psd_perp.shape[1]
            psd_perp = np.concatenate(
                [psd_perp, np.repeat(psd_perp[:, -1:], pad, axis=1)], axis=1
            )
            psd_parallel = np.concatenate(
                [psd_parallel, np.repeat(psd_parallel[:, -1:], pad, axis=1)], axis=1
            )

        freq_stft = np.asarray(f_axis, dtype=np.float32)
        self._spectral_result_cache[cache_key] = {
            "freq": freq_stft,
            "perp": psd_perp,
            "parallel": psd_parallel,
        }
        return freq_stft, psd_perp, psd_parallel


def build_demo_plotters(
    trange,
    data_dir=str(VEX_DATA_DIR),
    show_fig=True,
    workers=4,
    include_tangential_panels=True,
):
    """
    Build and return the plotting objects needed by the demo.

    Returns:
        tuple: (plotter, plotter_wav, analysis)
    """
    from venus_express.export import build_demo_plotters as _build_demo_plotters

    return _build_demo_plotters(
        trange,
        data_dir=data_dir,
        show_fig=show_fig,
        workers=workers,
        include_tangential_panels=include_tangential_panels,
    )


def run_demo(trange, data_dir=str(VEX_DATA_DIR), show_fig=True, workers=4, render_plots=True):
    if render_plots:
        plotter, plotter_wav, analysis = build_demo_plotters(
            trange,
            data_dir=data_dir,
            show_fig=show_fig,
            workers=workers,
            include_tangential_panels=True,
        )
        plotter.save("polarization_analysis.png", dpi=300)
        plotter_wav.save("wavelet_analysis.png", dpi=300)
    else:
        analysis = DataAnalysis(trange, data_dir=data_dir)

if __name__ == "__main__":
    trange = ['2006-05-08/03:20:00', '2006-05-08/03:45:00']
    # trange = ['2006-05-08/03:20:00', '2006-05-08/03:30:00']
    # trange = ['2006-05-08/21:24:47', '2006-05-08/21:28:53']
    # trange = ["2006-05-08/03:23:47", "2006-05-08/03:34:53"]
    # trange = ['2006-06-22/03:00:00', '2006-06-22/03:50:00']
    # trange = ['2006-06-01/03:10:00', '2006-06-01/03:20:00']
    # trange = ['2006-06-01/03:00:00', '2006-06-01/03:30:00']
    # trange = ['2006-06-25/00:50:00', '2006-06-25/01:00:00']
    # trange = ['2006-06-23/02:30:00', '2006-06-23/03:00:00']
    # trange = ['2006-09-07/00:20:00', '2006-09-07/00:40:00']
    run_demo(trange, show_fig=True)
