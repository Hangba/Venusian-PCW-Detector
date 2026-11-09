#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot-export helpers shared by analysis and surf_venus."""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from venus_express._plot import BasePlotter
from venus_express.analysis import DataAnalysis
from venus_express.analysis_windows import (
    EXPORT_DISPLAY_SECONDS,
    EXPORT_POL_STEP_SECONDS,
    export_analysis_geometry,
)
from venus_express.cache import load_features_for_trange
from venus_express.constants import R_VENUS
from venus_express.paths import CACHE_DIR, VEX_DATA_DIR
from venus_express.physics.gyro import get_gyro_frequency
from venus_express.physics.field_connection import dataframe_field_connection
from venus_express.signal.fft_metrics import (
    compute_perp_fft_metrics,
    compute_pf_band_peak_ratio,
)

# Paper font for the secondary-label export figure (New Computer Modern Math).
_FONT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "assets", "fonts", "NewCMMath-Book.otf")
)
_EXPORT_FONT = None


def _export_font():
    """Lazily register + return the export FontProperties; None if file missing."""
    global _EXPORT_FONT
    if _EXPORT_FONT is None:
        if not os.path.exists(_FONT_FILE):
            return None
        try:
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(_FONT_FILE)
            _EXPORT_FONT = fm.FontProperties(fname=_FONT_FILE)
        except Exception:
            return None
    return _EXPORT_FONT


def export_figure_rc() -> dict:
    """rcParams to render the export figure in NewComputerModernMath. Wrap BOTH
    the figure build and the savefig in matplotlib.rc_context(export_figure_rc())
    so dynamically regenerated tick labels also pick up the font. `mathtext.fontset
    = 'cm'` makes $..$ subscripts render in Computer Modern (matches NewCM text).
    Empty if the font file is unavailable."""
    fp = _export_font()
    if fp is None:
        return {}
    return {
        "font.family": fp.get_name(),
        "mathtext.fontset": "cm",
        # Keep text as text in vector exports so the SVG remains editable in
        # Illustrator/Inkscape instead of turning every glyph into a path.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _main_axes_from_figure(fig):
    """Return data axes (exclude narrow colorbar axes)."""
    return [ax for ax in fig.axes if ax.get_position().width > 0.2]


def _move_titles_to_left(fig, variables: List[Dict]) -> None:
    """Move per-panel titles from top to the left side of each subplot."""
    for ax, var_info in zip(_main_axes_from_figure(fig), variables):
        panel_title = var_info.get("title", "")
        ax.set_title("")
        if panel_title:
            ax.text(
                -0.16,
                0.5,
                panel_title,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=10,
            )


def _build_overlay_fcp(analysis: DataAnalysis) -> List[Dict]:
    f_cp_array = get_gyro_frequency(analysis.dataframe["Bt"].values * 1e-9)
    f_cp_times = analysis.dataframe["time_ut"].values
    return [
        {
            "data": f_cp_array,
            "times": f_cp_times,
            "color": "white",
            "linewidth": 1.5,
            "linestyle": "-",
            "label": r"$f_{cp}$",
        }
    ]


def _moving_average_1d(values: np.ndarray, window_size: int = 4) -> np.ndarray:
    """Edge-padded moving average for PSD curve smoothing."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)

    window_size = max(1, min(int(window_size), arr.size))
    if window_size == 1:
        return arr.astype(np.float32, copy=False)

    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    kernel = np.full(window_size, 1.0 / window_size, dtype=np.float64)
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32, copy=False)


def _crop_spectral_result(result: Dict, t_start: pd.Timestamp, t_end: pd.Timestamp) -> Dict:
    """Crop spectral analysis result to requested time range.

    Used when polarization analysis was computed on an extended window
    to avoid edge artifacts. Crops all time-indexed arrays (2D spectrograms
    and 1D window_center_times) back to the requested range.
    """
    window_times = result.get("window_center_times")
    if window_times is None or len(window_times) == 0:
        return result

    # window_center_times come from df['time_ut'].values, which drops the tz
    # info (tz-naive numpy datetime64 holding UTC wall-clock), whereas
    # t_start/t_end are tz-aware (pd.to_datetime(..., utc=True)). Comparing the
    # two raises "Cannot compare tz-naive and tz-aware timestamps". Normalize
    # the window times to tz-aware UTC — works for both naive and aware inputs.
    window_times_utc = pd.to_datetime(np.asarray(window_times), utc=True)
    mask = np.asarray((window_times_utc >= t_start) & (window_times_utc <= t_end))
    if not mask.any():
        # No windows fall in range - return empty result
        cropped = result.copy()
        for key in ["pol_ratio", "ellip_signed", "coh", "power", "wna_deg", "wna_reliability"]:
            if key in cropped and cropped[key] is not None:
                cropped[key] = cropped[key][:0]  # Empty window/time dimension (axis 0)
        cropped["window_center_times"] = window_times[:0]
        cropped["window_center_indices"] = np.array([], dtype=int)
        return cropped

    # Spectral arrays are [n_windows, n_freq] (time on axis 0, freq on axis 1) —
    # window_center_times indexes axis 0. Crop the window/time axis (axis 0).
    cropped = result.copy()
    for key in ["pol_ratio", "ellip_signed", "coh", "power", "wna_deg", "wna_reliability"]:
        if key in cropped and cropped[key] is not None:
            cropped[key] = cropped[key][mask]

    cropped["window_center_times"] = window_times[mask]
    if "window_center_indices" in cropped:
        cropped["window_center_indices"] = cropped["window_center_indices"][mask]

    return cropped


def build_demo_plotters(
    trange,
    data_dir: str = str(VEX_DATA_DIR),
    show_fig: bool = True,
    workers: int = 4,
    include_tangential_panels: bool = True,
) -> Tuple[BasePlotter, BasePlotter, DataAnalysis]:
    """Build analysis demo plotters (polarization figure + wavelet figure)."""
    analysis = DataAnalysis(trange, data_dir=data_dir)

    pol_result = analysis.polarization_analysis_on_xy(step_size=10, workers=workers)
    wna_result = analysis.propagation_angle_analysis(step_size=10, workers=workers)

    pol_ratio = pol_result["pol_ratio"]
    ellip_signed = pol_result["ellip_signed"]
    coh = pol_result["coh"]
    power = pol_result["power"]
    freq = pol_result["freq_pol"]
    window_centers = pol_result["window_center_times"]

    wna_deg = wna_result["wna_deg"]
    wna_freq = wna_result["freq_wna"]
    wna_window_centers = wna_result["window_center_times"]

    f_cp_overlay = _build_overlay_fcp(analysis)

    variables: List[Dict] = [
        {
            "data": coh,
            "times": window_centers,
            "freqs": freq,
            "title": "Coherence",
            "zlabel": "Coherence",
            "cmap": "RdBu_r",
            "vmin": 0,
            "vmax": 1,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": pol_ratio,
            "times": window_centers,
            "freqs": freq,
            "title": "Polarization Ratio",
            "zlabel": r"Polarization Ratio $\mathcal{P}$",
            "cmap": "viridis",
            "vmin": 0,
            "vmax": 1,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": ellip_signed,
            "times": window_centers,
            "freqs": freq,
            "title": "Signed Ellipticity",
            "zlabel": r"Ellipticity $\epsilon$",
            "cmap": "RdBu_r",
            "vmin": -1,
            "vmax": 1,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": power,
            "times": window_centers,
            "freqs": freq,
            "title": "Power Spectrum",
            "zlabel": r"Power ($nT^2/Hz$)",
            "cmap": "RdBu_r",
            "vmin": 0,
            "vmax": 2,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": wna_deg,
            "times": wna_window_centers,
            "freqs": wna_freq,
            "title": "Wave Normal Angle",
            "zlabel": r"$\theta_{k}$ (deg)",
            "cmap": "jet",
            "vmin": 0,
            "vmax": 30,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
    ]

    if include_tangential_panels:
        variables.append(
            {
                "data": analysis.dataframe[["B_fac_x", "B_fac_y"]].values,
                "times": analysis.dataframe["time_ut"].values,
                "title": "X and Y Tangential Magnetic Field (FAC Coordinates)",
                "ylabel": "Magnetic Field (nT)",
                "labels": ["$\\delta B_{\\perp1}$", "$\\delta B_{\\perp2}$"],
                "colors": ["blue", "green"],
                "yscale": "log",
            }
        )

    freqs_cwt, power_cwt, _ = analysis.wavelet_analysis()
    variables_wavelet: List[Dict] = [
        {
            "data": analysis.dataframe[["B_fac_x", "B_fac_y"]].values,
            "times": analysis.dataframe["time_ut"].values,
            "title": "X and Y Tangential Magnetic Field (FAC)",
            "ylabel": "Magnetic Field (nT)",
            "labels": ["$\\delta B_{\\perp1}$", "$\\delta B_{\\perp2}$"],
            "colors": ["blue", "green"],
        },
        {
            "data": np.log10(power_cwt + 1e-5),
            "times": analysis.dataframe["time_ut"].values,
            "freqs": freqs_cwt,
            "title": "Tangential Wavelet Power",
            "zlabel": "$\\log_{10}$ Power",
            "cmap": "jet",
            "ylabel": "Frequency (Hz)",
            "yscale": "log",
            "ylim": [0.01, 0.5],
            "overlay_lines": f_cp_overlay,
        },
    ]

    plotter = BasePlotter(trange, fs=analysis.fs)
    plotter.tplot(
        variables,
        title="Polarization Analysis on FAC Tangential Plane",
        figsize=(14, 14),
        show_fig=show_fig,
    )

    plotter_wav = BasePlotter(trange, fs=analysis.fs)
    plotter_wav.tplot(
        variables_wavelet,
        title="Wavelet Power Analysis",
        figsize=(14, 8),
        show_fig=show_fig,
    )

    return plotter, plotter_wav, analysis


# --- Polarization "paper figure" display geometry ----------------------------
# The polarization spectrograms (coherence / ellipticity / power / WNA) are a
# sliding STFT with window DEFAULT_POL_WINDOW_SIZE (160 s) stepped by
# _POL_STEP_SIZE. A short selection therefore yields only a handful of window
# centers, collapsing each spectrogram into a few vertical stripes (severe
# distortion). To fix this we render the analysis over a longer, event-centered
# window so there are ~MIN_POL_DISPLAY_SEC / _POL_STEP_SIZE columns. Bump
# MIN_POL_DISPLAY_SEC if you want an even longer span per figure.
_POL_STEP_SIZE = EXPORT_POL_STEP_SECONDS
MIN_POL_DISPLAY_SEC = EXPORT_DISPLAY_SECONDS


def build_selected_region_plotters(
    trange,
    data_dir: str = str(VEX_DATA_DIR),
    show_fig: bool = False,
    workers: int = 1,
    include_tangential_panels: bool = True,
    min_display_sec: int = MIN_POL_DISPLAY_SEC,
) -> Tuple[BasePlotter, DataAnalysis]:
    """Build one combined matplotlib figure for selected region analysis.

    Short selections are widened (keeping the event centered) to at least
    ``min_display_sec`` so the sliding-window polarization spectrograms have
    enough STFT window centers to render continuously instead of collapsing into
    a few vertical stripes. The analysis itself runs on a still-wider window
    (padded by one STFT window on each side) so the displayed edges are fully
    covered, then every result is cropped back to the display window.
    """
    geometry = export_analysis_geometry(tuple(trange), min_display_sec=min_display_sec)
    disp_start, disp_end = geometry.display_start, geometry.display_end
    analysis_trange = list(geometry.analysis_trange)
    analysis = DataAnalysis(analysis_trange, data_dir=data_dir)

    pol_result = analysis.polarization_analysis_on_xy(step_size=_POL_STEP_SIZE, workers=workers)
    wna_result = analysis.propagation_angle_analysis(step_size=_POL_STEP_SIZE, workers=workers)

    # 3) Crop spectral results + dataframe (wavelet / tangential / f_cp) back to
    #    the display window.
    pol_result = _crop_spectral_result(pol_result, disp_start, disp_end)
    wna_result = _crop_spectral_result(wna_result, disp_start, disp_end)
    df_mask = (analysis.dataframe["time_ut"] >= disp_start) & (analysis.dataframe["time_ut"] <= disp_end)
    analysis.dataframe = analysis.dataframe[df_mask].reset_index(drop=True)

    pol_ratio = pol_result["pol_ratio"]
    ellip_signed = pol_result["ellip_signed"]
    coh = pol_result["coh"]
    power = pol_result["power"]
    freq = pol_result["freq_pol"]
    window_centers = pol_result["window_center_times"]

    wna_deg = wna_result["wna_deg"]
    wna_freq = wna_result["freq_wna"]
    wna_window_centers = wna_result["window_center_times"]

    # f_cp_overlay will use the cropped dataframe
    f_cp_overlay = _build_overlay_fcp(analysis)

    spectral_panels: List[Dict] = [
        {
            "data": coh,
            "times": window_centers,
            "freqs": freq,
            "title": "Coherence",
            "zlabel": "Coherence",
            "cmap": "RdBu_r",
            "vmin": 0,
            "vmax": 1,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": pol_ratio,
            "times": window_centers,
            "freqs": freq,
            "title": "Polarization Ratio",
            "zlabel": r"Polarization Ratio $\nu$",
            "cmap": "viridis",
            "vmin": 0,
            "vmax": 1,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": ellip_signed,
            "times": window_centers,
            "freqs": freq,
            "title": "Signed Ellipticity",
            "zlabel": "Signed Ellipticity",
            "cmap": "RdBu_r",
            "vmin": -1,
            "vmax": 1,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": power,
            "times": window_centers,
            "freqs": freq,
            "title": "Power",
            "zlabel": "Power",
            "cmap": "RdBu_r",
            "vmin": 0,
            "vmax": 2,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
        {
            "data": wna_deg,
            "times": wna_window_centers,
            "freqs": wna_freq,
            "title": "Propagation Angle (Means)",
            "zlabel": "WNA [deg]",
            "cmap": "RdBu_r",
            "vmin": 0,
            "vmax": 30,
            "ylabel": "Frequency (Hz)",
            "ylim": [0.0, 0.5],
            "overlay_lines": f_cp_overlay,
        },
    ]

    freqs_cwt, power_cwt, _ = analysis.wavelet_analysis()
    tangential_panel = {
        "data": analysis.dataframe[["B_fac_x", "B_fac_y"]].values,
        "times": analysis.dataframe["time_ut"].values,
        "title": r"X and Y Tangential Magnetic Field ($B_{FAC}$)",
        "ylabel": "Magnetic Field (nT)",
        "labels": [r"$\delta B_{\perp 1}$", r"$\delta B_{\perp 2}$"],
        "colors": ["blue", "green"],
    }
    wavelet_panel = {
        "data": np.log10(power_cwt + 1e-5),
        "times": analysis.dataframe["time_ut"].values,
        "freqs": freqs_cwt,
        "title": "Tangential Wavelet Power Spectrum",
        "zlabel": r"$\log_{10}$ Power",
        "cmap": "jet",
        "ylabel": "Frequency (Hz)",
        "yscale": "log",
        "ylim": [0.01, 0.5],
        "overlay_lines": f_cp_overlay,
    }

    variables: List[Dict] = []
    if include_tangential_panels:
        variables.append(tangential_panel)
    variables.append(wavelet_panel)
    variables.extend(spectral_panels)

    figure_height = max(12.0, 2.2 * len(variables) + 2.0)
    # x-axis must span the (possibly widened) display window, not the short
    # requested trange — BasePlotter hard-sets xlim from the trange it is given.
    display_trange = [
        disp_start.strftime("%Y-%m-%d/%H:%M:%S"),
        disp_end.strftime("%Y-%m-%d/%H:%M:%S"),
    ]
    plotter = BasePlotter(display_trange, fs=analysis.fs)
    plotter.tplot(
        variables,
        title="Selected Region Analysis",
        figsize=(14, figure_height),
        show_fig=show_fig,
    )
    _move_titles_to_left(plotter.fig, variables)

    return plotter, analysis


def build_selected_region_training_channels_plotter(
    trange,
    data_dir: str = str(VEX_DATA_DIR),
    show_fig: bool = False,
    num_freqs: int = 64,
    feature_cache_dir: str = str(CACHE_DIR),
) -> Tuple[BasePlotter, Dict[str, np.ndarray]]:
    """Build a figure showing cached channels used by the current ICW model."""
    features = load_features_for_trange(
        trange=trange,
        data_dir=data_dir,
        num_freqs=num_freqs,
        cache_dir=feature_cache_dir,
        pol_window_size=160,  # v9 canonical (54 freq bins) — must match training/inference
        pol_step_size=20,
    )

    times = pd.to_datetime(features["times_s"], unit="s", utc=True)
    delta_bt = np.asarray(features["delta_bt"], dtype=np.float32)
    ellip = np.asarray(features["ellip"], dtype=np.float32)
    fcp_curve = np.asarray(features["fcp_curve"], dtype=np.float32)

    n_samples = times.shape[0]
    if (
        delta_bt.shape[0] != n_samples
        or ellip.ndim != 2
        or ellip.shape[1] != n_samples
        or fcp_curve.shape[0] != n_samples
    ):
        raise ValueError(
            "Training-channel feature alignment failed for selected trange: "
            f"ellip={ellip.shape}, fcp_curve={fcp_curve.shape}."
        )

    bt_channels = delta_bt[:, None]
    ellip_bins = np.arange(ellip.shape[0], dtype=np.float32)

    variables: List[Dict] = [
        {
            "data": bt_channels,
            "times": times,
            "title": "Scalar Input Channels",
            "ylabel": "Input Value",
            "labels": [r"$\delta B_T$"],
            "colors": ["black"],
        },
        {
            "data": ellip,
            "times": times,
            "freqs": ellip_bins,
            "title": "Ellipticity Input",
            "zlabel": r"Ellipticity $\epsilon$",
            "cmap": "RdBu_r",
            "vmin": -1.0,
            "vmax": 1.0,
            "ylabel": "Ellip Bin",
        },
    ]

    plotter = BasePlotter(trange, fs=1.0)
    plotter.tplot(
        variables,
        title="AI Cached Channels (Current Model)",
        figsize=(14, 10),
        show_fig=show_fig,
    )
    _move_titles_to_left(plotter.fig, variables)
    return plotter, features


def build_selected_region_fft_tangential_plotter(
    trange,
    data_dir: str = str(VEX_DATA_DIR),
    show_fig: bool = False,
    fft_window_sec: int = 256,
    smooth_points: int = 4,
    fft_pad_factor: int = 2,
) -> Tuple[BasePlotter, DataAnalysis]:
    """
    Build a Delva-fig2-like PSD line plot on FAC components.

    Panel style:
    - perpendicular PSD curve: P_perp = P_x + P_y
    - parallel PSD curve: P_parallel = P_z
    - 4-point moving average on PSD curves
    - zero-padding for denser frequency sampling display
    - log-log axes with fixed limits
    """
    analysis = DataAnalysis(trange, data_dir=data_dir)
    if analysis.dataframe is None or len(analysis.dataframe) == 0:
        raise ValueError("No data available for selected trange")

    bx = analysis.dataframe["B_fac_x"].to_numpy(dtype=np.float64, copy=False)
    by = analysis.dataframe["B_fac_y"].to_numpy(dtype=np.float64, copy=False)
    bz = analysis.dataframe["B_fac_z"].to_numpy(dtype=np.float64, copy=False)
    bt = analysis.dataframe["Bt"].to_numpy(dtype=np.float64, copy=False)

    n_samples = bx.size
    if n_samples < 16:
        raise ValueError("Selected region is too short for FFT analysis (need >=16 samples)")

    fs = float(analysis.fs)
    n_fft = int(max(16, min(int(fft_window_sec), n_samples)))

    if n_samples > n_fft:
        start = (n_samples - n_fft) // 2
        end = start + n_fft
        bx = bx[start:end]
        by = by[start:end]
        bz = bz[start:end]
    else:
        n_fft = n_samples

    metrics = compute_perp_fft_metrics(
        bx,
        by,
        bz,
        bt,
        fs=fs,
        pad_factor=fft_pad_factor,
        smooth_points=smooth_points,
    )
    freq = metrics["freq"]
    psd_perp = metrics["psd_perp"]
    psd_para = metrics["psd_parallel"]
    peak_freq = float(metrics["peak_freq"])
    peak_psd = float(metrics["peak_psd"])
    f_cp_ref = (
        float(metrics["f_cp_ref"])
        if metrics.get("f_cp_ref") is not None
        else np.nan
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(freq, psd_perp, color="tab:blue", linewidth=1.6, label=r"$P_{\perp}$")
    ax.plot(freq, psd_para, color="tab:red", linewidth=1.6, label=r"$P_{\parallel}$")
    ax.scatter(
        [peak_freq],
        [peak_psd],
        color="tab:blue",
        s=36,
        zorder=5,
        label=rf"$P_{{\perp}}$ peak={peak_freq:.3f} Hz",
    )
    ax.annotate(
        f"{peak_freq:.3f} Hz",
        xy=(peak_freq, peak_psd),
        xytext=(10, 10),
        textcoords="offset points",
        color="tab:blue",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:blue", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.0),
    )
    if np.isfinite(f_cp_ref) and f_cp_ref > 0.0:
        ax.axvline(
            f_cp_ref,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=rf"$f_{{cp}}={f_cp_ref:.3f}\,Hz$",
        )
        ax.axvline(
            0.5 * f_cp_ref,
            color="#666666",
            linestyle="--",
            linewidth=1.1,
            label=rf"$0.5 f_{{cp}}={0.5 * f_cp_ref:.3f}\,Hz$",
        )
        ax.axvline(
            1.5 * f_cp_ref,
            color="#999999",
            linestyle="--",
            linewidth=1.1,
            label=rf"$1.5 f_{{cp}}={1.5 * f_cp_ref:.3f}\,Hz$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-2, 1e3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD (nT$^2$/Hz)")
    ax.set_title(f"FFT PSD ({max(1, int(smooth_points))}-point average)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    plotter = BasePlotter(trange, fs=analysis.fs)
    plotter.fig = fig
    if show_fig:
        plt.show()
    else:
        plt.close(fig)
    return plotter, analysis


def _add_cbar_spacer(ax) -> None:
    """Reserve a right-side colorbar-width slot on a line panel so its time axis
    lines up with the spectrogram panels (which have real colorbars)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.1)
    cax.axis("off")


def _heatmap_panel(
    fig, ax, x_times, freqs, c2d, label, cmap, vmin, vmax,
    ylim=(0.0, 0.5), yscale=None, overlay_t=None, overlay_y=None,
) -> None:
    """Render one spectrogram panel. ``c2d`` must already be [n_freq, n_x]."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.1)
    x = np.asarray(x_times)
    f = np.asarray(freqs, dtype=float)
    C = np.asarray(c2d)
    if C.size == 0 or x.size == 0 or f.size == 0:
        cax.axis("off")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(label)
        return
    mesh = ax.pcolormesh(x, f, C, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(label)
    if overlay_t is not None and overlay_y is not None:
        ax.plot(overlay_t, overlay_y, color="white", lw=1.2, ls="-")
    if yscale == "log":
        ax.set_yscale("log")
    ax.set_ylim(ylim)
    ax.set_ylabel("Freq (Hz)")


def build_secondary_label_export_figure(
    trange,
    data_dir: str = str(VEX_DATA_DIR),
    show_fig: bool = False,
    fcp_scale_factor: float = 1.0,
) -> Tuple["object", Dict]:
    """Build a single composite paper-quality export figure for one ICW event,
    plus a dict of physics parameters rendered in the right-hand sidebar.

    Layout (custom GridSpec — NOT BasePlotter.tplot, which only stacks equal-height
    time-aligned panels and cannot host the FFT's frequency x-axis or a text panel):
      left column  : FFT PSD (main, tallest) on its own frequency x-axis, then 5
                     time-aligned panels — raw Bt, FAC Bx/By, ellipticity, WNA,
                     wavelet — sharing the UT x-axis.
      right column : text panel with 8 physics parameters.

    Scalar params and the FFT are computed over the ORIGINAL event window
    (``trange``); the spectrograms are computed on an event-centered widened
    window (mirroring ``build_selected_region_plotters``) so they render
    continuously. Position is reported in the native VSO frame in R_Venus units.

    Returns ``(fig, params)``. The caller owns ``fig`` (close it after use).
    """
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    geometry = export_analysis_geometry(tuple(trange))
    t_start, t_end = geometry.event_start, geometry.event_end
    disp_start, disp_end = geometry.display_start, geometry.display_end
    analysis_trange = list(geometry.analysis_trange)
    analysis = DataAnalysis(analysis_trange, data_dir=data_dir)
    if analysis.dataframe is None or len(analysis.dataframe) == 0:
        raise ValueError("No data available for selected trange")

    pol_result = analysis.polarization_analysis_on_xy(step_size=_POL_STEP_SIZE, workers=1)
    wna_result = analysis.propagation_angle_analysis(step_size=_POL_STEP_SIZE, workers=1)
    pol_result = _crop_spectral_result(pol_result, disp_start, disp_end)
    wna_result = _crop_spectral_result(wna_result, disp_start, disp_end)
    df_mask = (analysis.dataframe["time_ut"] >= disp_start) & (analysis.dataframe["time_ut"] <= disp_end)
    analysis.dataframe = analysis.dataframe[df_mask].reset_index(drop=True)
    df = analysis.dataframe

    ellip = np.asarray(pol_result["ellip_signed"])          # [n_win, n_freq]
    freq_pol = np.asarray(pol_result["freq_pol"], dtype=float) if pol_result["freq_pol"] is not None else np.array([])
    pol_centers = pd.to_datetime(np.asarray(pol_result["window_center_times"]), utc=True)
    wna_deg = np.asarray(wna_result["wna_deg"])              # [n_win, n_freq]
    freq_wna = np.asarray(wna_result["freq_wna"], dtype=float) if wna_result["freq_wna"] is not None else np.array([])
    wna_centers = pd.to_datetime(np.asarray(wna_result["window_center_times"]), utc=True)

    # ---- Event-window slice (original trange) for scalar params + FFT ----
    times_utc = pd.to_datetime(df["time_ut"].values, utc=True)
    ev_mask = np.asarray((times_utc >= t_start) & (times_utc <= t_end))
    ev = df[ev_mask] if int(np.count_nonzero(ev_mask)) >= 16 else df
    bx = ev["B_fac_x"].to_numpy(dtype=np.float64)
    by = ev["B_fac_y"].to_numpy(dtype=np.float64)
    bz = ev["B_fac_z"].to_numpy(dtype=np.float64)
    bt = ev["Bt"].to_numpy(dtype=np.float64)
    fs = float(analysis.fs)

    # f_cp, f_max, and PSD curves use the canonical manual-review FFT metrics.
    # f_max is the TRUE GLOBAL peak (not band-limited): the export figure must
    # show the real spectral peak so a reviewer can verify it actually falls in
    # [0.8,1.2]*f_cp. Events whose global peak is out of band are filtered out
    # upstream by the audit (merged_band_ok), so every exported figure's global
    # peak is genuinely in-band — same condition, single source of truth.
    fft_metrics = compute_perp_fft_metrics(
        bx,
        by,
        bz,
        bt,
        fs=fs,
        pad_factor=2,
        smooth_points=4,
    )
    f_cp = (
        float(fft_metrics["f_cp_ref"])
        if fft_metrics.get("f_cp_ref") is not None
        else float("nan")
    )
    # Ion search: scale the proton f_cp reference to the target ion's cyclotron
    # frequency (f_c ∝ Z/A; fcp_scale_factor = Z/A). For protons it stays 1.0.
    _scale = float(fcp_scale_factor) if fcp_scale_factor and fcp_scale_factor > 0 else 1.0
    if _scale != 1.0 and np.isfinite(f_cp):
        f_cp = f_cp * _scale
    f_max = float(fft_metrics["peak_freq"])
    fft_freq = fft_metrics["freq"]
    psd_perp = fft_metrics["psd_perp"]
    psd_para = fft_metrics["psd_parallel"]
    event_pf_band_ratio = compute_pf_band_peak_ratio(
        bx, by, np.full(min(bx.size, by.size), f_cp), fs=fs
    )

    # Best-window band dominance: max over 80s windows of in-band/out-of-band
    # peak P_perp. Measured per-window (not on the whole event span) because the
    # ratio collapses toward 1 on long segments; a clean ICW window reaches ~17x.
    from venus_express.signal.fft_metrics import best_window_band_dominance
    best_band_dominance = best_window_band_dominance(bx, by, bt, fs=fs)
    event_band_dominance = fft_metrics.get("band_dominance_ratio")

    # ---- mean ellipticity / WNA at f_max and f_cp (avg over event-window columns) ----
    def _mean_at(arr2d, freq_axis, centers, f_target):
        arr2d = np.asarray(arr2d)
        if arr2d.size == 0 or freq_axis.size == 0 or not np.isfinite(f_target):
            return float("nan")
        bin_i = int(np.argmin(np.abs(freq_axis - f_target)))
        col = arr2d[:, bin_i].astype(np.float64)
        if centers.size == col.size:
            wm = np.asarray((centers >= t_start) & (centers <= t_end))
            if wm.any():
                col = col[wm]
        finite = col[np.isfinite(col)]
        return float(np.mean(finite)) if finite.size else float("nan")

    ellip_at_fmax = _mean_at(ellip, freq_pol, pol_centers, f_max)
    ellip_at_fcp = _mean_at(ellip, freq_pol, pol_centers, f_cp)
    wna_at_fmax = _mean_at(wna_deg, freq_wna, wna_centers, f_max)
    wna_at_fcp = _mean_at(wna_deg, freq_wna, wna_centers, f_cp)

    # ---- mean spacecraft position (VSO, R_Venus) over event window ----
    def _mean_pos(col):
        if col in ev.columns:
            v = ev[col].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                return float(np.mean(v) / R_VENUS)
        return float("nan")

    pos_x, pos_y, pos_z = _mean_pos("pos_x"), _mean_pos("pos_y"), _mean_pos("pos_z")
    connection = dataframe_field_connection(df, t_start, t_end)

    def _num(v):
        return float(v) if isinstance(v, float) and np.isfinite(v) else None

    params = {
        "time_start": t_start.strftime("%Y-%m-%d/%H:%M:%S"),
        "time_end": t_end.strftime("%Y-%m-%d/%H:%M:%S"),
        "pos_vso_rv": [_num(pos_x), _num(pos_y), _num(pos_z)],
        "background_bow_shock_connection": connection,
        "f_cp_hz": _num(f_cp),
        "f_max_hz": _num(f_max),
        "fmax_over_fcp": _num((f_max / f_cp) if (np.isfinite(f_max) and np.isfinite(f_cp) and f_cp > 0) else float("nan")),
        "band_dominance_ratio": _num(best_band_dominance if isinstance(best_band_dominance, float) else float("nan")),
        "event_band_dominance_ratio": _num(
            float(event_band_dominance)
            if event_band_dominance is not None
            else float("nan")
        ),
        "event_pf_band_ratio": _num(float(event_pf_band_ratio)),
        "ellip_at_fmax": _num(ellip_at_fmax),
        "ellip_at_fcp": _num(ellip_at_fcp),
        "wna_at_fmax_deg": _num(wna_at_fmax),
        "wna_at_fcp_deg": _num(wna_at_fcp),
    }

    # ===================== Figure (paper layout) =====================
    #   row0:  B_VSO | B_FAC      (two waveform panels, side by side)
    #   row1:  FFT               (main body, spans full width)
    #   row2:  Ellip | WNA | wavelet   (three spectrograms, spread out)
    #   right column: centered physics-parameter panel
    fp = _export_font()
    fig = plt.figure(figsize=(17, 10.5))
    outer = GridSpec(
        1, 2, width_ratios=[5, 1], wspace=0.04, figure=fig,
        top=0.91, bottom=0.07, left=0.055, right=0.975,
    )
    left_col = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[0, 0], height_ratios=[1.0, 2.3, 1.4], hspace=0.55,
    )
    gs_row0 = GridSpecFromSubplotSpec(1, 2, subplot_spec=left_col[0], wspace=0.22)
    gs_row2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=left_col[2], wspace=0.62)
    ax_bvso = fig.add_subplot(gs_row0[0, 0])
    ax_bfac = fig.add_subplot(gs_row0[0, 1])
    ax_fft = fig.add_subplot(left_col[1])
    ax_ell = fig.add_subplot(gs_row2[0, 0])
    ax_wna = fig.add_subplot(gs_row2[0, 1])
    ax_wav = fig.add_subplot(gs_row2[0, 2])
    ax_txt = fig.add_subplot(outer[0, 1])
    ax_txt.axis("off")

    disp_times = pd.to_datetime(df["time_ut"].values, utc=True)
    f_cp_overlay = get_gyro_frequency(df["Bt"].values * 1e-9) * _scale  # ion: scale to Z/A line

    def _fmt_time(ax, fmt="%H:%M:%S"):
        ax.set_xlim(disp_start, disp_end)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        ax.tick_params(axis="x", labelrotation=18, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    # --- row0 left: B in VSO frame (raw components) ---
    ax_bvso.plot(disp_times, df["Bx"].values, color="tab:red", lw=0.7, label=r"$B_x$")
    ax_bvso.plot(disp_times, df["By"].values, color="tab:green", lw=0.7, label=r"$B_y$")
    ax_bvso.plot(disp_times, df["Bz"].values, color="tab:blue", lw=0.7, label=r"$B_z$")
    ax_bvso.axvspan(t_start, t_end, color="orange", alpha=0.08)
    ax_bvso.set_title(r"$B_{\mathrm{VSO}}$ (nT)")
    ax_bvso.legend(loc="upper right", fontsize=7, ncol=3)
    ax_bvso.grid(True, alpha=0.3)
    _fmt_time(ax_bvso)

    # --- row0 right: B in field-aligned coordinates ---
    ax_bfac.plot(disp_times, df["B_fac_x"].values, color="tab:blue", lw=0.7, label=r"$B_{\perp 1}$")
    ax_bfac.plot(disp_times, df["B_fac_y"].values, color="tab:green", lw=0.7, label=r"$B_{\perp 2}$")
    ax_bfac.plot(disp_times, df["B_fac_z"].values, color="tab:red", lw=0.7, label=r"$B_\parallel$")
    ax_bfac.axvspan(t_start, t_end, color="orange", alpha=0.08)
    ax_bfac.set_title(r"$B_{\mathrm{FAC}}$ (nT)")
    ax_bfac.legend(loc="upper right", fontsize=7, ncol=3)
    ax_bfac.grid(True, alpha=0.3)
    _fmt_time(ax_bfac)

    # --- row1: FFT power spectrum (main body, frequency x-axis) ---
    ax_fft.plot(fft_freq, psd_perp, color="tab:blue", lw=1.6, label=r"$P_\perp$")
    ax_fft.plot(fft_freq, psd_para, color="tab:red", lw=1.1, label=r"$P_\parallel$")
    if np.isfinite(f_max) and f_max > 0:
        ax_fft.axvline(f_max, color="tab:blue", ls="-", lw=1.2, label=f"$f_\\mathrm{{max}}$ = {f_max:.4f} Hz")
    if np.isfinite(f_cp) and f_cp > 0:
        ax_fft.axvline(f_cp, color="black", ls="--", lw=1.4, label=f"$f_\\mathrm{{cp}}$ = {f_cp:.4f} Hz")
    ax_fft.set_xscale("log")
    ax_fft.set_yscale("log")
    ax_fft.set_xlim(1e-3, 0.5)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel(r"PSD (nT$^2$/Hz)")
    ax_fft.set_title("FFT power spectrum (tangential)")
    ax_fft.grid(True, which="both", alpha=0.3)
    ax_fft.legend(loc="upper right", fontsize=9)

    # --- row2: ellipticity / WNA / wavelet spectrograms ---
    _heatmap_panel(fig, ax_ell, pol_centers, freq_pol, ellip.T if ellip.size else ellip,
                   "", "RdBu_r", -1, 1, overlay_t=disp_times, overlay_y=f_cp_overlay)
    ax_ell.set_title("Ellipticity")
    _fmt_time(ax_ell, "%H:%M")
    _heatmap_panel(fig, ax_wna, wna_centers, freq_wna, wna_deg.T if wna_deg.size else wna_deg,
                   "", "RdBu_r", 0, 30, overlay_t=disp_times, overlay_y=f_cp_overlay)
    ax_wna.set_title("Wave normal angle (WNA)")
    _fmt_time(ax_wna, "%H:%M")
    freqs_cwt, power_cwt, _ = analysis.wavelet_analysis()
    _heatmap_panel(fig, ax_wav, disp_times, freqs_cwt, np.log10(np.asarray(power_cwt) + 1e-5),
                   "", "viridis", None, None, ylim=(0.01, 0.5), yscale="log",
                   overlay_t=disp_times, overlay_y=f_cp_overlay)
    ax_wav.set_title("Wavelet power")
    _fmt_time(ax_wav, "%H:%M")

    # --- right column: centered physics-parameter panel (no time interval) ---
    def _fmt(v, nd=4):
        return "n/a" if v is None else f"{v:.{nd}f}"

    connection_label = {'intersects': 'Yes', 'no_intersection': 'No', 'unavailable': 'n/a'}[connection['status']]
    txt = "\n".join([
        "Parameters", "",
        r"Position VSO ($R_V$):",
        f"X = {_fmt(params['pos_vso_rv'][0], 3)}",
        f"Y = {_fmt(params['pos_vso_rv'][1], 3)}",
        f"Z = {_fmt(params['pos_vso_rv'][2], 3)}", "",
        f"B0 line / shock: {connection_label}",
        f"Distance = {_fmt(connection['distance_rv'], 2)} $R_V$", "",
        f"$f_\\mathrm{{cp}}$  = {_fmt(params['f_cp_hz'])} Hz",
        f"$f_\\mathrm{{max}}$ = {_fmt(params['f_max_hz'])} Hz",
        f"$f_\\mathrm{{max}}/f_\\mathrm{{cp}}$ = {_fmt(params['fmax_over_fcp'], 3)}", "",
        f"band PSD dominance (event) = {_fmt(params['event_band_dominance_ratio'], 2)}", "",
        f"P/F ratio p75 (event) = {_fmt(params['event_pf_band_ratio'], 2)}", "",
        f"band PSD dominance (best win) = {_fmt(params['band_dominance_ratio'], 2)}", "",
        f"Ellip @ $f_\\mathrm{{max}}$ = {_fmt(params['ellip_at_fmax'], 3)}",
        f"Ellip @ $f_\\mathrm{{cp}}$  = {_fmt(params['ellip_at_fcp'], 3)}", "",
        f"WNA @ $f_\\mathrm{{max}}$ = {_fmt(params['wna_at_fmax_deg'], 2)} deg",
        f"WNA @ $f_\\mathrm{{cp}}$  = {_fmt(params['wna_at_fcp_deg'], 2)} deg",
    ])
    _txt_kw = {"fontproperties": fp} if fp is not None else {}
    ax_txt.text(0.5, 0.5, txt, transform=ax_txt.transAxes, va="center", ha="center",
                fontsize=12, **_txt_kw)

    _sup_kw = {"fontproperties": fp} if fp is not None else {}
    fig.suptitle(f"ICW analysis export — {params['time_start']} -> {params['time_end']} UT",
                 y=0.955, fontsize=13, **_sup_kw)

    if show_fig:
        plt.show()
    return fig, params
