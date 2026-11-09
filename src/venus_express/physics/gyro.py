"""Gyro-frequency calculations."""

import numpy as np

from venus_express.constants import E_CHARGE, M_PROTON


def get_gyro_frequency(B_t, charge_number=1, mass_number=1) -> np.ndarray:
    """Compute ion gyro-frequency from magnetic-field strength in Tesla."""
    return (charge_number * E_CHARGE * B_t) / (2 * np.pi * M_PROTON * mass_number)
