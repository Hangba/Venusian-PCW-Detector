"""Straight event-background field connection to the plotted bow-shock model."""

from dataclasses import asdict

import numpy as np

from venus_express.constants import R_VENUS
from venus_express.physics.bow_shock import (
    BOW_SHOCK_ABERRATION_DEG,
    BOW_SHOCK_SOLAR_MINIMUM,
    vso_to_aberrated_vso,
)


def dataframe_field_connection(frame, start, end, cutoff_hz=0.01):
    """Evaluate only declared event samples from an already filtered frame."""
    import pandas as pd

    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    times = pd.to_datetime(frame['time_ut'], utc=True)
    event = frame.loc[(times >= start) & (times <= end)]
    result = event_field_connection(
        event[['pos_x', 'pos_y', 'pos_z']].to_numpy(float),
        event[['Bx_bg', 'By_bg', 'Bz_bg']].to_numpy(float),
    )
    result.update(background_cutoff_hz=cutoff_hz, event_start_utc=start.isoformat(),
                  event_end_utc=end.isoformat())
    return result


CONNECTION_CSV_COLUMNS = [
    'B0ShockConnection', 'B0ShockDistanceRv', 'B0ShockDistanceKm',
    'B0ShockSignedDistanceRv', 'B0ShockIntersectionAVSOx',
    'B0ShockIntersectionAVSOy', 'B0ShockIntersectionAVSOz',
    'B0ShockValidSamples',
]


def connection_csv_values(connection):
    """Flatten the same connection metadata for both parameter export APIs."""
    point = connection.get('intersection_avso_rv') or [None, None, None]
    return dict(zip(CONNECTION_CSV_COLUMNS, [
        connection.get('status'), connection.get('distance_rv'),
        connection.get('distance_km'), connection.get('signed_distance_rv'),
        *point, connection.get('valid_samples'),
    ]))


def event_field_connection(position_vso_km, background_vso_nt):
    """Use paired finite event samples; extend the mean B0 in both directions.

    Input B0 must be the ion-scale filtered background used by the figure.
    The nearest valid intersection is selected in full 3-D AVSO, before any
    X-rho projection. This is a static model geometry, not field-line tracing.
    """
    from venus_express.sample_analysis.calculations.physics import (
        field_line_bow_shock_intersection,
    )

    position = np.asarray(position_vso_km, dtype=float)
    field = np.asarray(background_vso_nt, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3 or field.shape != position.shape:
        raise ValueError("Event positions and background must be aligned Nx3 arrays")
    valid = np.isfinite(position).all(axis=1) & np.isfinite(field).all(axis=1)
    model = BOW_SHOCK_SOLAR_MINIMUM
    result = dict(
        status="unavailable", intersects=None, valid_samples=int(valid.sum()),
        total_samples=len(position), model=asdict(model),
        aberration_deg=BOW_SHOCK_ABERRATION_DEG, coordinate_frame="AVSO",
        definition="event sample-mean position + t * unit(event sample-mean filtered B0); t in Rv, both signs",
        position_avso_rv=None, background_vso_nt=None, background_avso_nt=None,
        intersection_avso_rv=None, signed_distance_rv=None, distance_rv=None,
        distance_km=None,
    )
    if not valid.any():
        result["reason"] = "No paired finite position/background samples inside event"
        return result
    origin = vso_to_aberrated_vso(position[valid].mean(axis=0) / R_VENUS)
    mean_field = field[valid].mean(axis=0)
    avso_field = vso_to_aberrated_vso(mean_field)
    result.update(position_avso_rv=origin.tolist(), background_vso_nt=mean_field.tolist(),
                  background_avso_nt=avso_field.tolist())
    if np.linalg.norm(avso_field) <= 1e-10:
        result["reason"] = "Event mean background is zero or cancels; direction undefined"
        return result
    try:
        point, distance = field_line_bow_shock_intersection(origin, avso_field, model)
    except ValueError as exc:
        result.update(status="no_intersection", intersects=False, reason=str(exc))
        return result
    result.update(status="intersects", intersects=True, intersection_avso_rv=point.tolist(),
                  signed_distance_rv=float(distance), distance_rv=abs(float(distance)),
                  distance_km=abs(float(distance)) * R_VENUS)
    return result
