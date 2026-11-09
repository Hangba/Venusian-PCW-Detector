"""Physical and mission constants used across the project.

The Venus/solar-wind defaults follow the definitions used by Weichbold et al.
(2025), *First Detection of Deuterium in Venus's Extended Exosphere*.  The
paper uses event-resolved ASPERA-4 solar-wind measurements; values marked as
fallbacks below are only used when those measurements are unavailable.
"""

E_CHARGE = 1.602176634e-19
M_PROTON = 1.67262192369e-27

# The paper explicitly uses 1 R_V = 6050 km.
R_VENUS = 6050.0

# VSO definition: solar wind travels approximately in -X, while Venus's
# orbital velocity contributes approximately -35 km/s in Y.  Therefore
# v_inj = -v_sw + v_Venus = (+V_sw, -35, 0) km/s.
DEFAULT_SOLAR_WIND_SPEED_KM_S = 350.0
VENUS_ORBITAL_VELOCITY_Y_KM_S = -35.0

# The paper reports a particle-to-wave transfer fraction of 0.2--0.4.  Use the
# interval midpoint as the fallback, while keeping it user-configurable.
DEFAULT_ICW_ENERGY_TRANSFER_EFFICIENCY = 0.30

# No universal n_SW is prescribed by the paper: it uses event-resolved
# VEX/ASPERA-4 observations.  This is an explicit fallback only.
DEFAULT_SOLAR_WIND_DENSITY_CM3 = 10.0
