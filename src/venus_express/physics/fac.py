"""Field-aligned coordinate transforms."""

from typing import Callable, Dict

import numpy as np

from venus_express.signal.filters import butterworth_filter


def transform_to_fac(Bx_vso, By_vso, Bz_vso, Bx_bg, By_bg, Bz_bg):
    """
    Vectorized transformation from VSO to FAC (Field-Aligned Coordinates).

    FAC system:
      z: along background field
      x: tangential axis defined by B_bg x [1, 0, 0]
      y: completes right-hand system
    """
    B_bg = np.stack([Bx_bg, By_bg, Bz_bg], axis=1)
    B_bg_mag = np.linalg.norm(B_bg, axis=1, keepdims=True)

    valid = B_bg_mag.squeeze() > 1e-10

    e_z = np.zeros_like(B_bg)
    e_z[valid] = B_bg[valid] / B_bg_mag[valid]

    cross = np.stack([np.zeros_like(Bz_bg), -Bz_bg, By_bg], axis=1)
    cross_mag = np.linalg.norm(cross, axis=1, keepdims=True)

    e_x = np.zeros_like(cross)
    valid_cross = cross_mag.squeeze() > 1e-10
    e_x[valid_cross] = cross[valid_cross] / cross_mag[valid_cross]

    e_y = np.cross(e_z, e_x)

    B_vso = np.stack([Bx_vso, By_vso, Bz_vso], axis=1)

    Bx_fac = np.sum(e_x * B_vso, axis=1)
    By_fac = np.sum(e_y * B_vso, axis=1)
    Bz_fac = np.sum(e_z * B_vso, axis=1)

    return Bx_fac, By_fac, Bz_fac


def convert_to_fac_coordinates(
    vex_data: Dict[str, np.ndarray],
    cutoff_freq: float = 0.01,
    fs: float = 1.0,
    filter_fn: Callable = butterworth_filter,
):
    """Convert VEX magnetic-field data from VSO into FAC coordinates."""
    vex_data_with_fac = vex_data.copy()

    Bx_bg = filter_fn(vex_data["Bx"], cutoff_freq, fs)
    By_bg = filter_fn(vex_data["By"], cutoff_freq, fs)
    Bz_bg = filter_fn(vex_data["Bz"], cutoff_freq, fs)

    Bx_fluct = vex_data["Bx"] - Bx_bg
    By_fluct = vex_data["By"] - By_bg
    Bz_fluct = vex_data["Bz"] - Bz_bg

    Bx_fac, By_fac, Bz_fac = transform_to_fac(
        Bx_fluct,
        By_fluct,
        Bz_fluct,
        Bx_bg,
        By_bg,
        Bz_bg,
    )

    vex_data_with_fac["Bx_bg"] = Bx_bg
    vex_data_with_fac["By_bg"] = By_bg
    vex_data_with_fac["Bz_bg"] = Bz_bg
    vex_data_with_fac["Bx_fluct"] = Bx_fluct
    vex_data_with_fac["By_fluct"] = By_fluct
    vex_data_with_fac["Bz_fluct"] = Bz_fluct
    vex_data_with_fac["Bx_fac"] = Bx_fac
    vex_data_with_fac["By_fac"] = By_fac
    vex_data_with_fac["Bz_fac"] = Bz_fac
    vex_data_with_fac["B_fac_x"] = Bx_fac
    vex_data_with_fac["B_fac_y"] = By_fac
    vex_data_with_fac["B_fac_z"] = Bz_fac

    return vex_data_with_fac
