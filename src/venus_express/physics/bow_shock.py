"""Axisymmetric Venus bow-shock models for solar minimum and maximum.

Coordinates are expressed in Venus radii in an aberrated VSO frame.  The
models use the conic representation

    r = L / (1 + epsilon * cos(theta))

with the focus at ``(x0, 0, 0)``.  Positive X points from Venus toward the
Sun.  A point has a positive signed distance when it is upstream/outside the
model surface and a negative value when it is planetward/inside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOW_SHOCK_ABERRATION_DEG = 5.0


def vso_to_aberrated_vso(
    vectors: np.ndarray,
    angle_deg: float = BOW_SHOCK_ABERRATION_DEG,
) -> np.ndarray:
    """Rotate VSO vectors into the bow-shock model's aberrated VSO frame.

    The model +X axis is rotated toward VSO -Y by the published 5-degree
    solar-wind aberration. The transformation accepts a single 3-vector or an
    array whose final dimension is three.
    """

    values = np.asarray(vectors, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("VSO vectors must have a final dimension of three")
    angle = np.radians(angle_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotated = values.copy()
    rotated[..., 0] = cosine * values[..., 0] - sine * values[..., 1]
    rotated[..., 1] = sine * values[..., 0] + cosine * values[..., 1]
    return rotated


@dataclass(frozen=True)
class BowShockModel:
    """Conic bow-shock surface with cylindrical symmetry around the X axis."""

    name: str
    semi_latus_rectum_rv: float
    eccentricity: float
    focus_x_rv: float = 0.0
    solar_activity: str = "unspecified"
    reference: str = ""
    note: str = ""

    @property
    def subsolar_distance_rv(self) -> float:
        """Distance of the shock nose from the planet center."""

        return self.focus_x_rv + self.semi_latus_rectum_rv / (1.0 + self.eccentricity)

    @property
    def terminator_distance_rv(self) -> float:
        """Cylindrical shock distance at the planetary terminator (X=0)."""

        l_value = self.semi_latus_rectum_rv
        epsilon = self.eccentricity
        x0 = self.focus_x_rv
        value = l_value * (l_value + 2.0 * epsilon * x0) + x0**2 * (epsilon**2 - 1.0)
        return float(np.sqrt(value))

    def cylindrical_radius_rv(self, x_rv: float | np.ndarray) -> np.ndarray:
        """Return the model cylindrical radius ``sqrt(Y**2 + Z**2)`` at X.

        The positive branch of the conic is returned.  Locations for which the
        conic has no real cylindrical radius are returned as ``nan``.
        """

        x = np.asarray(x_rv, dtype=np.float64)
        dx = x - self.focus_x_rv
        rho_squared = (
            self.semi_latus_rectum_rv**2
            - 2.0 * self.semi_latus_rectum_rv * self.eccentricity * dx
            + (self.eccentricity**2 - 1.0) * dx**2
        )
        tolerance = 1e-12
        return np.sqrt(
            np.where(rho_squared >= -tolerance, np.clip(rho_squared, 0.0, None), np.nan)
        )

    def signed_implicit_distance_rv(
        self,
        x_rv: float | np.ndarray,
        y_rv: float | np.ndarray,
        z_rv: float | np.ndarray,
    ) -> np.ndarray:
        """Evaluate the conic implicit function at one or more positions.

        This quantity is zero on the surface.  It is positive on the solar-wind
        side of the shock and negative on the planetward side.  It has units of
        Venus radii but is not the exact Euclidean shortest distance.
        """

        x = np.asarray(x_rv, dtype=np.float64)
        y = np.asarray(y_rv, dtype=np.float64)
        z = np.asarray(z_rv, dtype=np.float64)
        dx = x - self.focus_x_rv
        radius_from_focus = np.sqrt(dx * dx + y * y + z * z)
        return radius_from_focus + self.eccentricity * dx - self.semi_latus_rectum_rv

    def is_outside(
        self,
        x_rv: float | np.ndarray,
        y_rv: float | np.ndarray,
        z_rv: float | np.ndarray,
    ) -> np.ndarray:
        """Return true for positions upstream/outside the bow shock."""

        return self.signed_implicit_distance_rv(x_rv, y_rv, z_rv) > 0.0

    def curve(self, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """Return the dayside/nightside conic branch for X-rho plotting."""

        theta_limit = np.pi
        if self.eccentricity > 1.0:
            theta_limit = float(np.arccos(-1.0 / self.eccentricity))
        theta = np.linspace(0.0, theta_limit * (1.0 - 1e-6), points)
        denominator = 1.0 + self.eccentricity * np.cos(theta)
        radius = self.semi_latus_rectum_rv / denominator
        return self.focus_x_rv + radius * np.cos(theta), radius * np.sin(theta)


# Martinecz et al. (2008), Table 1: first five months of VEX ASPERA-4
# observations under solar-minimum conditions.  Its quoted terminator distance
# is 1.984 Rv.
BOW_SHOCK_SOLAR_MINIMUM = BowShockModel(
    name="martinecz_2008_solar_minimum",
    semi_latus_rectum_rv=1.303,
    eccentricity=1.056,
    focus_x_rv=0.788,
    solar_activity="minimum",
    reference="Martinecz et al. (2008), doi:10.1016/j.pss.2007.07.007",
    note="Three-parameter VEX/ASPERA-4 fit in 5-degree aberrated VSO coordinates.",
)


# Martinecz et al. (2008), Fig. 2, uses the Russell et al. (1988) PVO solar-
# maximum curve and concludes that the terminator shock moves from about 2.14
# Rv at solar minimum to 2.40 Rv at solar maximum.  Russell's curve is a
# planet-centered two-parameter ellipse.  Martinecz reproduces epsilon=0.609
# for the Russell minimum fit but does not reprint a separate maximum epsilon;
# this project therefore retains that published mean shape and sets L to the
# explicitly reported solar-maximum terminator distance.  The approximation is
# documented here so it is not mistaken for a new independent fit.
BOW_SHOCK_SOLAR_MAXIMUM = BowShockModel(
    name="russell_1988_solar_maximum",
    semi_latus_rectum_rv=2.400,
    eccentricity=0.609,
    focus_x_rv=0.0,
    solar_activity="maximum",
    reference=(
        "Martinecz et al. (2008), Fig. 2; Russell et al. (1988), "
        "doi:10.1029/JA093iA06p05461"
    ),
    note=(
        "Planet-centered PVO solar-maximum proxy: published 2.40 Rv terminator "
        "distance with the Russell solar-cycle mean eccentricity reproduced by Martinecz."
    ),
)


BOW_SHOCK_MODELS = {
    "solar_minimum": BOW_SHOCK_SOLAR_MINIMUM,
    "solar_maximum": BOW_SHOCK_SOLAR_MAXIMUM,
}


def get_bow_shock_model(solar_activity: str) -> BowShockModel:
    """Return a named solar-minimum or solar-maximum model."""

    key = solar_activity.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "minimum": "solar_minimum",
        "min": "solar_minimum",
        "valley": "solar_minimum",
        "maximum": "solar_maximum",
        "max": "solar_maximum",
        "peak": "solar_maximum",
    }
    key = aliases.get(key, key)
    try:
        return BOW_SHOCK_MODELS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(BOW_SHOCK_MODELS))
        message = f"Unknown solar activity {solar_activity!r}; choose one of: {choices}"
        raise ValueError(message) from exc
