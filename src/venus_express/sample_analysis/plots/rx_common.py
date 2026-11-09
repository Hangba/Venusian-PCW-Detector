"""Shared coordinate and reference helpers for cylindrical R-X plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd

from venus_express.physics.bow_shock import (
    BOW_SHOCK_SOLAR_MINIMUM,
    BowShockModel,
    vso_to_aberrated_vso,
)


def prepare_rx_coordinates(events: pd.DataFrame) -> pd.DataFrame:
    """Return an event table with canonical ``RX_X_Rv``/``RX_R_Rv`` columns."""

    result = events.copy()
    if {"AVSO_X_Rv", "AVSO_R_cyl_Rv"}.issubset(result.columns):
        result["RX_X_Rv"] = result["AVSO_X_Rv"]
        result["RX_R_Rv"] = result["AVSO_R_cyl_Rv"]
    elif {"VSO_X_Rv", "VSO_Y_Rv", "VSO_Z_Rv"}.issubset(result.columns):
        avso = vso_to_aberrated_vso(
            result[["VSO_X_Rv", "VSO_Y_Rv", "VSO_Z_Rv"]].to_numpy(dtype=float)
        )
        result["RX_X_Rv"] = avso[:, 0]
        result["RX_R_Rv"] = np.hypot(avso[:, 1], avso[:, 2])
    else:
        raise ValueError("Events need AVSO coordinates or VSO X/Y/Z coordinates")
    return result


def rx_event_bounds(
    events: pd.DataFrame,
    *,
    model: BowShockModel = BOW_SHOCK_SOLAR_MINIMUM,
    padding_rv: float = 0.6,
    bin_size_rv: float | None = None,
) -> tuple[float, float, float]:
    """Return shared R-X bounds, optionally rounded to a fixed grid."""

    if events.empty:
        return -5.0, 5.0, 5.0
    x = events["RX_X_Rv"].to_numpy(dtype=float)
    r = events["RX_R_Rv"].to_numpy(dtype=float)
    x_min = min(float(np.nanmin(x)) - padding_rv, -2.5)
    x_max = max(float(np.nanmax(x)) + padding_rv, model.subsolar_distance_rv + 0.5)
    r_max = max(float(np.nanmax(r)) + padding_rv, 4.0)
    if bin_size_rv is not None:
        x_min = np.floor(x_min / bin_size_rv) * bin_size_rv
        x_max = np.ceil(x_max / bin_size_rv) * bin_size_rv
        r_max = np.ceil(r_max / bin_size_rv) * bin_size_rv
    return x_min, x_max, r_max


def add_rx_references(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    r_max: float,
    *,
    model: BowShockModel = BOW_SHOCK_SOLAR_MINIMUM,
    light: bool = False,
    zorder: float = 4,
) -> None:
    """Draw the bow shock, wake boundary, R-X Venus, and zero axes."""

    line_color = "white" if light else "0.18"
    shock_x, shock_r = model.curve(points=900)
    visible = (
        np.isfinite(shock_x)
        & np.isfinite(shock_r)
        & (shock_x >= x_min)
        & (shock_x <= x_max)
        & (shock_r <= r_max)
    )
    ax.plot(shock_x[visible], shock_r[visible], color=line_color, lw=1.4, zorder=zorder)
    ax.plot([0.0, x_min], [1.0, 1.0], color=line_color, lw=1.0, ls="--", zorder=zorder)
    ax.add_patch(
        Wedge(
            (0, 0), 1, 0, 90, facecolor="#ffffff", edgecolor="0.2", lw=0.8,
            zorder=zorder + 1,
        )
    )
    ax.add_patch(
        Wedge(
            (0, 0), 1, 90, 180, facecolor="#171717", edgecolor="0.2", lw=0.8,
            zorder=zorder + 1,
        )
    )
    ax.axhline(0.0, color="0.55", lw=0.55, zorder=zorder - 1)
    ax.axvline(0.0, color="0.55", lw=0.55, zorder=zorder - 1)


def format_rx_axis(ax: plt.Axes, x_min: float, x_max: float, r_max: float) -> None:
    """Apply the shared cylindrical R-X axes and viewing direction."""

    ax.set_xlabel(r"$X_{\mathrm{AVSO}}$ ($R_{\mathrm{V}}$)")
    ax.set_ylabel(r"$R_{\mathrm{AVSO}}=\sqrt{Y_{\mathrm{AVSO}}^2+Z_{\mathrm{AVSO}}^2}$ ($R_{\mathrm{V}}$)")
    ax.set_xlim(x_max, x_min)
    ax.set_ylim(0.0, r_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
