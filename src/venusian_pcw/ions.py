"""Integrate the appendix's IMA moments on calibrated instrument bins."""

import numpy as np
import pandas as pd
from scipy.constants import atomic_mass, elementary_charge

from venus_express.repro.settings import BUILD


def integrate(
    counts, geometric_factor, mass_to_charge, energy_eV, width_eV, solid_angle_sr, unit_vector
):
    """Integrate over sector, energy and pitch for each instrument scan."""
    if counts.ndim != 5:
        raise ValueError("counts must be scan × ring × sector × energy × pitch")
    energy = np.asarray(energy_eV)[None, None, :, None]
    width = np.asarray(width_eV)[None, None, :, None]
    omega = np.asarray(solid_angle_sr)[None, :, None, :]
    directions = np.asarray(unit_vector)[None, :, None, :, :]
    if (
        not np.isfinite(energy).all()
        or not np.isfinite(width).all()
        or (energy <= 0).any()
        or (width <= 0).any()
        or not np.isfinite(omega).all()
        or (omega <= 0).any()
    ):
        raise ValueError("Energy centers, widths, and solid angles must be finite and positive")
    if not np.allclose(np.linalg.norm(directions, axis=-1), 1.0):
        raise ValueError("IMAS look-direction vectors must be unit length")
    valid = (
        np.isfinite(counts) & (counts >= 0) & np.isfinite(geometric_factor) & (geometric_factor > 0)
    )
    j = np.divide(
        counts * 1e4,
        geometric_factor * energy[None, ...] * 0.1209,
        out=np.zeros_like(counts, dtype=float),
        where=valid,
    )
    output = {}
    for label, mass, low, high in [("H", 1, 1.0, 1.55), ("O", 16, 14.0, 20.0)]:
        selected = (mass_to_charge >= low) & (mass_to_charge <= high)
        flux = (j * selected).sum(axis=1)
        speed = np.sqrt(2 * elementary_charge * energy / (mass * atomic_mass))
        bin_flux = flux * width * omega
        density = (bin_flux / speed).sum(axis=(1, 2, 3))
        first = (bin_flux[..., None] * directions).sum(axis=(1, 2, 3))
        velocity = np.divide(
            first, density[:, None], out=np.full_like(first, np.nan), where=density[:, None] > 0
        )
        has_support = (valid & selected).any(axis=(1, 2, 3, 4))
        output[f"{label}_density_cm3"] = np.where(has_support, density * 1e-6, np.nan)
        for i, axis in enumerate("xyz"):
            output[f"{label}_velocity_{axis}_km_s"] = velocity[:, i] * 1e-3
    return output


def compute(bundle):
    with np.load(bundle, allow_pickle=False) as file:
        keys = (
            "counts",
            "geometric_factor",
            "mass_to_charge",
            "energy_eV",
            "width_eV",
            "solid_angle_sr",
            "unit_vector",
        )
        result = integrate(*(file[key] for key in keys))
        result["utc"] = file["utc"].astype(str)
    BUILD.mkdir(exist_ok=True)
    output = BUILD / "ima_moments.csv"
    pd.DataFrame(result).to_csv(output, index=False)
    print(output)
