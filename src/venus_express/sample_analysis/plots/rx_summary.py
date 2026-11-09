"""Publication R-X summary and tangential band-energy figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from venus_express.physics.bow_shock import BOW_SHOCK_SOLAR_MINIMUM

from ..style import analysis_figure_style
from .rx_common import (
    add_rx_references,
    format_rx_axis,
    prepare_rx_coordinates,
    rx_event_bounds,
)

RX_CMAP = "cividis"
ENERGY_RX_CMAP = "turbo"
PROPAGATION_COLOR = "#2d6ca2"
MAGNETOSHEATH_COLOR = "#e69f00"


def _grid_array(
    grid: pd.DataFrame,
    column: str,
    x_edges: np.ndarray,
    r_edges: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    values = np.full((len(r_edges) - 1, len(x_edges) - 1), np.nan)
    for row in grid.itertuples(index=False):
        x_index = int(round((row.x_bin_min_rv - x_edges[0]) / bin_size))
        r_index = int(round((row.r_bin_min_rv - r_edges[0]) / bin_size))
        if 0 <= x_index < values.shape[1] and 0 <= r_index < values.shape[0]:
            value = float(getattr(row, column))
            if np.isfinite(value):
                values[r_index, x_index] = value
    return values


def _bounds_from_grid(grid: pd.DataFrame) -> tuple[float, float, float, float]:
    bin_size = float((grid["x_bin_max_rv"] - grid["x_bin_min_rv"]).iloc[0])
    x_min = np.floor(min(float(grid["x_bin_min_rv"].min()), -2.5) / bin_size) * bin_size
    x_max = np.ceil(
        max(
            float(grid["x_bin_max_rv"].max()),
            BOW_SHOCK_SOLAR_MINIMUM.subsolar_distance_rv + 0.5,
        )
        / bin_size
    ) * bin_size
    r_max = np.ceil(max(float(grid["r_bin_max_rv"].max()), 4.0) / bin_size) * bin_size
    return x_min, x_max, r_max, bin_size


def _format_rx(
    ax: plt.Axes,
    *,
    x_min: float,
    x_max: float,
    r_max: float,
    title: str,
    panel: str | None = None,
    light_references: bool = False,
) -> None:
    add_rx_references(
        ax, x_min, x_max, r_max, light=light_references
    )
    format_rx_axis(ax, x_min, x_max, r_max)
    ax.set_title(title)
    if panel:
        ax.text(
            0.025,
            0.98,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
            zorder=7,
        )


def plot_rx_spatial_summary(
    events: pd.DataFrame,
    occurrence: pd.DataFrame,
    output: str | Path,
    *,
    min_coverage_days: float = 2.0,
    rate_min_per_day: float = 0.1,
    rate_max_per_day: float = 4.0,
) -> Path:
    """Plot event locations, orbit coverage, and thresholded linear rates."""

    prepared = prepare_rx_coordinates(events)
    required = {"plasma_region", "shock_class"}
    missing = sorted(required.difference(prepared.columns))
    if missing:
        raise ValueError(f"Missing event classification columns: {', '.join(missing)}")
    x_min, x_max, r_max, bin_size = _bounds_from_grid(occurrence)
    x_edges = np.arange(x_min, x_max + 0.5 * bin_size, bin_size)
    r_edges = np.arange(0.0, r_max + 0.5 * bin_size, bin_size)
    coverage = _grid_array(occurrence, "observed_days", x_edges, r_edges, bin_size)
    rates = _grid_array(occurrence, "event_rate_per_day", x_edges, r_edges, bin_size)
    covered = np.isfinite(coverage) & (coverage > 0.0)
    rates[covered] = np.maximum(
        np.nan_to_num(rates[covered], nan=0.0), rate_min_per_day
    )
    rates[~covered] = np.nan
    rate_min = rate_min_per_day
    rate_max = rate_max_per_day
    displayed_rates = np.ma.masked_invalid(rates)
    rate_values = displayed_rates.compressed()
    has_underflow = bool(np.any(rate_values < rate_min))
    has_overflow = bool(np.any(rate_values > rate_max))
    rate_extend = "both" if has_underflow and has_overflow else "min" if has_underflow else "max" if has_overflow else "neither"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with analysis_figure_style(font_size=11.0):
        fig, axes = plt.subplots(1, 3, figsize=(20.5, 8.2), constrained_layout=True)
        solar_wind = prepared[prepared["plasma_region"].eq("solar_wind")]
        parallel = solar_wind[solar_wind["shock_class"].eq("quasi_parallel")]
        perpendicular = solar_wind[solar_wind["shock_class"].eq("quasi_perpendicular")]
        sheath = prepared[prepared["plasma_region"].eq("magnetosheath")]
        axes[0].scatter(
            parallel["RX_X_Rv"], parallel["RX_R_Rv"], s=22, color=PROPAGATION_COLOR,
            linewidth=0.65, alpha=0.82, zorder=3,
        )
        axes[0].scatter(
            perpendicular["RX_X_Rv"], perpendicular["RX_R_Rv"], s=22,
            facecolor="none", edgecolor=PROPAGATION_COLOR, linewidth=0.9, alpha=0.82, zorder=3,
        )
        axes[0].scatter(
            sheath["RX_X_Rv"], sheath["RX_R_Rv"], s=27, marker="x",
            color=MAGNETOSHEATH_COLOR, linewidth=1.0, alpha=0.9, zorder=4,
        )
        axes[0].grid(alpha=0.16)
        _format_rx(
            axes[0], x_min=x_min, x_max=x_max, r_max=r_max,
            title=f"Event Distribution (n={len(prepared)})", panel="(a)",
        )
        axes[0].legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="none", color=PROPAGATION_COLOR,
                       markerfacecolor=PROPAGATION_COLOR, label=f"Quasi-parallel (n={len(parallel)})"),
                Line2D([0], [0], marker="o", linestyle="none", color=PROPAGATION_COLOR,
                       markerfacecolor="none", label=f"Quasi-perpendicular (n={len(perpendicular)})"),
                Line2D([0], [0], marker="x", linestyle="none", color=MAGNETOSHEATH_COLOR,
                       label=f"Magnetosheath (n={len(sheath)})"),
            ],
            loc="lower left", frameon=False, fontsize=8.5,
        )

        coverage_mesh = axes[1].pcolormesh(
            x_edges, r_edges, np.ma.masked_invalid(coverage), cmap=RX_CMAP,
            norm=Normalize(vmin=0.0, vmax=20.0), shading="flat",
            edgecolors=(1.0, 1.0, 1.0, 0.18), linewidth=0.15, zorder=1,
        )
        coverage_bar = fig.colorbar(
            coverage_mesh, ax=axes[1], shrink=0.82, pad=0.025, extend="max"
        )
        coverage_ticks = sorted({0.0, min_coverage_days, 5.0, 10.0, 15.0, 20.0})
        coverage_bar.set_ticks(coverage_ticks)
        coverage_bar.ax.axhline(min_coverage_days, color="white", lw=1.0, ls="--")
        coverage_bar.set_label("Observation time (days)")
        _format_rx(
            axes[1], x_min=x_min, x_max=x_max, r_max=r_max,
            title="Orbit Coverage", panel="(b)", light_references=True,
        )

        rate_mesh = axes[2].pcolormesh(
            x_edges, r_edges, displayed_rates, cmap=RX_CMAP,
            norm=Normalize(vmin=rate_min, vmax=rate_max), shading="flat",
            edgecolors=(1.0, 1.0, 1.0, 0.18), linewidth=0.15, zorder=1,
        )
        rate_bar = fig.colorbar(
            rate_mesh, ax=axes[2], shrink=0.82, pad=0.025, extend=rate_extend,
            ticks=[rate_min, 1.0, 2.0, 3.0, rate_max],
        )
        rate_bar.set_label(r"Occurrence rate (day$^{-1}$)")
        _format_rx(
            axes[2], x_min=x_min, x_max=x_max, r_max=r_max,
            title=rf"Event Occurrence Rate ($t_{{\mathrm{{obs}}}}>{min_coverage_days:g}$ days)",
            panel="(c)", light_references=True,
        )
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    return output


def plot_rx_band_energy_scatter(
    events: pd.DataFrame,
    output: str | Path,
    *,
    color_min: float | None = None,
    color_max: float | None = None,
) -> Path:
    """Plot R-X events with a logarithmic tangential band-peak energy scale."""

    prepared = prepare_rx_coordinates(events)
    column = (
        "wave_energy_pJ_m3"
        if "wave_energy_pJ_m3" in prepared.columns
        else "peak_band_energy_pJ_m3_per_Hz"
    )
    valid = prepared[np.isfinite(prepared[column]) & prepared[column].gt(0)].copy()
    if valid.empty:
        raise ValueError("No positive tangential band-energy values to plot")
    x_min, x_max, r_max = rx_event_bounds(
        valid, padding_rv=0.5, bin_size_rv=0.5
    )
    energy = valid[column].to_numpy(dtype=float)
    if color_min is None:
        color_min = float(energy.min())
    if color_max is None:
        color_max = float(np.percentile(energy, 99.0))
        if color_max <= color_min:
            color_max = color_min * 10.0
    if (
        not np.isfinite(color_min)
        or not np.isfinite(color_max)
        or color_min <= 0
        or color_max <= color_min
    ):
        raise ValueError("Log energy color bounds must satisfy 0 < min < max")
    has_underflow = bool(energy.min() < color_min)
    has_overflow = bool(energy.max() > color_max)
    color_extend = (
        "both" if has_underflow and has_overflow
        else "min" if has_underflow
        else "max" if has_overflow
        else "neither"
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with analysis_figure_style(font_size=11.0):
        fig, ax = plt.subplots(figsize=(8.6, 8.0), constrained_layout=True)
        scatter = ax.scatter(
            valid["RX_X_Rv"], valid["RX_R_Rv"], c=energy, cmap=ENERGY_RX_CMAP,
            norm=LogNorm(vmin=color_min, vmax=color_max), s=26,
            edgecolors=(1.0, 1.0, 1.0, 0.55), linewidths=0.35, alpha=0.9, zorder=3,
        )
        colorbar = fig.colorbar(
            scatter, ax=ax, shrink=0.82, pad=0.025,
            extend=color_extend,
        )
        colorbar.set_label(
            r"Wave energy (pJ m$^{-3}$)"
        )
        _format_rx(
            ax, x_min=x_min, x_max=x_max, r_max=r_max,
            title="Wave energy",
        )
        ax.grid(alpha=0.16)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    return output


def plot_rx_parallel_summary(
    events: pd.DataFrame,
    occurrence: pd.DataFrame,
    output: str | Path,
    *,
    min_coverage_days: float = 2.0,
    rate_min_per_day: float = 0.1,
    rate_max_per_day: float = 4.0,
) -> Path:
    """Plot event locations, occurrence rate, and peak energy side by side."""

    prepared = prepare_rx_coordinates(events)
    required = {"plasma_region", "shock_class"}
    missing = sorted(required.difference(prepared.columns))
    if missing:
        raise ValueError(f"Missing event columns for R-X summary: {', '.join(missing)}")
    if occurrence.empty:
        raise ValueError("No R-X occurrence-rate bins are available")

    x_min, x_max, r_max, bin_size = _bounds_from_grid(occurrence)
    x_edges = np.arange(x_min, x_max + 0.5 * bin_size, bin_size)
    r_edges = np.arange(0.0, r_max + 0.5 * bin_size, bin_size)
    coverage = _grid_array(occurrence, "observed_days", x_edges, r_edges, bin_size)
    rates = _grid_array(occurrence, "event_rate_per_day", x_edges, r_edges, bin_size)
    covered = np.isfinite(coverage) & (coverage > 0.0)
    rates[covered] = np.maximum(
        np.nan_to_num(rates[covered], nan=0.0), rate_min_per_day
    )
    rates[~covered] = np.nan
    rate_min = rate_min_per_day
    rate_max = rate_max_per_day
    displayed_rates = np.ma.masked_invalid(rates)
    rate_values = displayed_rates.compressed()
    has_underflow = bool(np.any(rate_values < rate_min))
    has_overflow = bool(np.any(rate_values > rate_max))
    rate_extend = (
        "both" if has_underflow and has_overflow
        else "min" if has_underflow
        else "max" if has_overflow
        else "neither"
    )

    energy_column = (
        "wave_energy_pJ_m3"
        if "wave_energy_pJ_m3" in prepared.columns
        else "peak_band_energy_pJ_m3_per_Hz"
    )
    valid_energy = prepared[
        np.isfinite(prepared[energy_column]) & prepared[energy_column].gt(0)
    ].copy()
    if valid_energy.empty:
        raise ValueError("No positive tangential band-energy values to plot")
    energy = valid_energy[energy_column].to_numpy(dtype=float)
    energy_min = float(energy.min())
    energy_max = float(np.percentile(energy, 99.0))
    if energy_max <= energy_min:
        energy_max = energy_min * 10.0
    energy_extend = "max" if energy.max() > energy_max else "neither"
    energy_powers = np.arange(
        np.ceil(np.log10(energy_min)),
        np.floor(np.log10(energy_max)) + 1.0,
    )
    energy_ticks = np.power(10.0, energy_powers)
    if energy_ticks.size == 0:
        energy_ticks = np.asarray([energy_min, energy_max])

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with analysis_figure_style(font_size=11.0):
        fig = plt.figure(figsize=(14.4, 5.1))
        grid = fig.add_gridspec(
            1,
            5,
            width_ratios=(1.0, 1.0, 0.045, 1.0, 0.045),
            left=0.065,
            right=0.975,
            bottom=0.20,
            top=0.90,
            wspace=0.12,
        )
        axes = np.asarray(
            [
                fig.add_subplot(grid[0, 0]),
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[0, 3]),
            ]
        )
        rate_cax = fig.add_subplot(grid[0, 2])
        energy_cax = fig.add_subplot(grid[0, 4])
        rate_position = rate_cax.get_position()
        rate_cax.set_position(
            (
                rate_position.x0 - 0.015,
                rate_position.y0,
                rate_position.width,
                rate_position.height,
            )
        )
        energy_position = energy_cax.get_position()
        energy_cax.set_position(
            (
                energy_position.x0 - 0.015,
                energy_position.y0,
                energy_position.width,
                energy_position.height,
            )
        )

        solar_wind = prepared[prepared["plasma_region"].eq("solar_wind")]
        parallel = solar_wind[solar_wind["shock_class"].eq("quasi_parallel")]
        perpendicular = solar_wind[
            solar_wind["shock_class"].eq("quasi_perpendicular")
        ]
        sheath = prepared[prepared["plasma_region"].eq("magnetosheath")]
        axes[0].scatter(
            parallel["RX_X_Rv"], parallel["RX_R_Rv"], s=18,
            color=PROPAGATION_COLOR, linewidth=0.55, alpha=0.82, zorder=3,
        )
        axes[0].scatter(
            perpendicular["RX_X_Rv"], perpendicular["RX_R_Rv"], s=18,
            facecolor="none", edgecolor=PROPAGATION_COLOR, linewidth=0.8,
            alpha=0.82, zorder=3,
        )
        axes[0].scatter(
            sheath["RX_X_Rv"], sheath["RX_R_Rv"], s=22, marker="x",
            color=MAGNETOSHEATH_COLOR, linewidth=0.9, alpha=0.9, zorder=4,
        )
        axes[0].legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="none", color=PROPAGATION_COLOR,
                       markerfacecolor=PROPAGATION_COLOR, label="Quasi-parallel"),
                Line2D([0], [0], marker="o", linestyle="none", color=PROPAGATION_COLOR,
                       markerfacecolor="none", label="Quasi-perpendicular"),
                Line2D([0], [0], marker="x", linestyle="none", color=MAGNETOSHEATH_COLOR,
                       label="Magnetosheath"),
            ], loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=3, frameon=False,
        )
        axes[0].grid(alpha=0.16)
        _format_rx(
            axes[0], x_min=x_min, x_max=x_max, r_max=r_max,
            title="", light_references=False,
        )
        axes[0].set_title(
            f"(a) Event distribution ($n={len(prepared)}$)", loc="left", pad=8
        )

        rate_mesh = axes[1].pcolormesh(
            x_edges, r_edges, displayed_rates, cmap=RX_CMAP,
            norm=Normalize(vmin=rate_min, vmax=rate_max),
            shading="flat", edgecolors=(1.0, 1.0, 1.0, 0.18), linewidth=0.15,
            rasterized=True, zorder=1,
        )
        rate_bar = fig.colorbar(
            rate_mesh, cax=rate_cax, extend=rate_extend,
            ticks=[rate_min, 1.0, 2.0, 3.0, rate_max],
        )
        rate_bar.set_label(r"Occurrence rate (day$^{-1}$)", labelpad=12)
        _format_rx(
            axes[1], x_min=x_min, x_max=x_max, r_max=r_max,
            title="", light_references=True,
        )
        axes[1].set_title("(b) Occurrence rate", loc="left", pad=8)

        energy_scatter = axes[2].scatter(
            valid_energy["RX_X_Rv"], valid_energy["RX_R_Rv"], c=energy,
            cmap=ENERGY_RX_CMAP, norm=LogNorm(vmin=energy_min, vmax=energy_max),
            s=19, edgecolors=(1.0, 1.0, 1.0, 0.55), linewidths=0.3,
            alpha=0.9, zorder=3,
        )
        energy_bar = fig.colorbar(
            energy_scatter,
            cax=energy_cax,
            extend=energy_extend,
            ticks=energy_ticks,
        )
        energy_bar.set_label(r"Wave energy (pJ m$^{-3}$)")
        _format_rx(
            axes[2], x_min=x_min, x_max=x_max, r_max=r_max,
            title="", light_references=False,
        )
        axes[2].set_title("(c) Wave energy", loc="left", pad=8)
        axes[2].grid(alpha=0.16)
        for ax in axes[1:]:
            ax.set_ylabel("")
        axes[2].tick_params(axis="y", labelleft=False)
        fig.align_ylabels([axes[0]])
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return output
