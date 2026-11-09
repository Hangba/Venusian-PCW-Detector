"""Per-event position, free energy, density inversion, and shock geometry."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from venus_express.constants import (
    DEFAULT_ICW_ENERGY_TRANSFER_EFFICIENCY,
    DEFAULT_SOLAR_WIND_DENSITY_CM3,
    DEFAULT_SOLAR_WIND_SPEED_KM_S,
    M_PROTON,
    R_VENUS,
    VENUS_ORBITAL_VELOCITY_Y_KM_S,
)
from venus_express.io.vex_mag import read_dat_files_df
from venus_express.physics.bow_shock import (
    BOW_SHOCK_SOLAR_MINIMUM,
    BowShockModel,
    vso_to_aberrated_vso,
)
from venus_express.signal.filters import butterworth_filter, compute_delta_bt_from_total

from .events import SampleEvent

MU0 = 4.0 * np.pi * 1e-7


def bow_shock_normal(
    position_rv: np.ndarray,
    model: BowShockModel = BOW_SHOCK_SOLAR_MINIMUM,
) -> np.ndarray:
    """Unit surface normal from the gradient of the bow-shock conic."""

    x, y, z = np.asarray(position_rv, dtype=np.float64)
    dx = x - model.focus_x_rv
    radius = float(np.linalg.norm([dx, y, z]))
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("Bow-shock normal is undefined at the conic focus")
    gradient = np.array(
        [dx / radius + model.eccentricity, y / radius, z / radius],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(gradient))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Bow-shock gradient is not finite")
    return gradient / norm


def field_line_bow_shock_intersection(
    position_rv: np.ndarray,
    background_b: np.ndarray,
    model: BowShockModel = BOW_SHOCK_SOLAR_MINIMUM,
) -> tuple[np.ndarray, float]:
    """Intersect the local background-field line with the conic bow shock.

    The field line is ``position + t * B_hat``. Both magnetic-field directions
    are considered and the valid intersection with the smallest ``abs(t)`` is
    returned. The signed ``t`` is expressed in Venus radii.
    """

    position = np.asarray(position_rv, dtype=np.float64)
    field = np.asarray(background_b, dtype=np.float64)
    field_norm = float(np.linalg.norm(field))
    if (
        position.shape != (3,)
        or field.shape != (3,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(field))
        or field_norm <= 0
    ):
        raise ValueError("Position and background field must be finite 3-vectors")
    direction = field / field_norm

    q = position.copy()
    q[0] -= model.focus_x_rv
    dx = float(direction[0])
    qx = float(q[0])
    l_value = float(model.semi_latus_rectum_rv)
    epsilon = float(model.eccentricity)
    rhs0 = l_value - epsilon * qx
    coefficients = np.array(
        [
            1.0 - epsilon**2 * dx**2,
            2.0 * (float(np.dot(q, direction)) + epsilon * dx * rhs0),
            float(np.dot(q, q)) - rhs0**2,
        ],
        dtype=np.float64,
    )
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    tolerance = 1e-12 * scale
    if abs(coefficients[0]) <= tolerance:
        if abs(coefficients[1]) <= tolerance:
            raise ValueError("Background-field line does not define a bow-shock crossing")
        roots = np.array([-coefficients[2] / coefficients[1]])
    else:
        roots = np.roots(coefficients)

    candidates: list[tuple[float, np.ndarray]] = []
    for root in roots:
        if abs(float(np.imag(root))) > 1e-8:
            continue
        distance = float(np.real(root))
        point = position + distance * direction
        relative_x = point[0] - model.focus_x_rv
        conic_rhs = l_value - epsilon * relative_x
        residual = float(model.signed_implicit_distance_rv(*point))
        if conic_rhs >= -1e-8 and abs(residual) <= 1e-6:
            candidates.append((distance, point))
    if not candidates:
        raise ValueError("Local background-field line does not intersect the bow-shock model")
    nearest_distance, nearest_point = min(candidates, key=lambda item: abs(item[0]))
    return nearest_point, nearest_distance


def field_normal_angle_deg(background_b: np.ndarray, normal: np.ndarray) -> float:
    """Directed 0..180 degree angle between background field and shock normal."""

    b = np.asarray(background_b, dtype=float)
    n = np.asarray(normal, dtype=float)
    denominator = float(np.linalg.norm(b) * np.linalg.norm(n))
    if denominator == 0 or not np.isfinite(denominator):
        raise ValueError("Background field and normal must be finite and non-zero")
    cosine = float(np.clip(np.dot(b, n) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def injection_velocity_vso_km_s(
    solar_wind_speed_km_s: float,
    venus_orbital_velocity_y_km_s: float = VENUS_ORBITAL_VELOCITY_Y_KM_S,
) -> np.ndarray:
    """Return the aberrated injection velocity in VSO coordinates.

    The paper defines ``v_inj = -v_sw + v_Venus``.  With the solar wind
    travelling in VSO ``-X`` and Venus's orbital velocity in ``-Y``, this is
    ``(+V_sw, -35, 0) km/s`` for the project fallback values.
    """

    vector = np.array(
        [solar_wind_speed_km_s, venus_orbital_velocity_y_km_s, 0.0], dtype=float
    )
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 0:
        raise ValueError("Injection velocity must be a finite non-zero vector")
    return vector


def injection_angle_deg(background_b: np.ndarray, injection_velocity: np.ndarray) -> float:
    """Directed 0..180 degree angle between event B_bg and v_inj."""

    b = np.asarray(background_b, dtype=float)
    velocity = np.asarray(injection_velocity, dtype=float)
    denominator = float(np.linalg.norm(b) * np.linalg.norm(velocity))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Background field and injection velocity must be finite and non-zero")
    cosine = float(np.clip(np.dot(b, velocity) / denominator, -1.0, 1.0))
    if np.isclose(cosine, 1.0, rtol=0.0, atol=1e-15):
        cosine = 1.0
    elif np.isclose(cosine, -1.0, rtol=0.0, atol=1e-15):
        cosine = -1.0
    return float(np.degrees(np.arccos(cosine)))


def cone_angle_deg(background_b: np.ndarray) -> float:
    """Return the 0..180 degree angle between B_bg and solar-wind flow (-X)."""

    b = np.asarray(background_b, dtype=float)
    if b.shape != (3,) or not np.all(np.isfinite(b)):
        raise ValueError("Background field must be a finite 3-vector")
    norm = float(np.linalg.norm(b))
    if norm <= 0:
        raise ValueError("Background field must be non-zero")
    cosine = float(np.clip(-b[0] / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def clock_angle_deg(background_b: np.ndarray) -> float:
    """Return B_bg clock angle in the Y-Z plane, from +Y toward +Z (0..360)."""

    b = np.asarray(background_b, dtype=float)
    if b.shape != (3,) or not np.all(np.isfinite(b)):
        raise ValueError("Background field must be a finite 3-vector")
    if float(np.hypot(b[1], b[2])) <= 0:
        return float("nan")
    return float(np.degrees(np.arctan2(b[2], b[1])) % 360.0)


def classify_shock_angle(theta_bn_deg: float) -> str:
    """Classify by the latest user-defined field-line/bow-shock angle rule."""

    if not np.isfinite(theta_bn_deg):
        return "quasi_perpendicular"
    if 45.0 < theta_bn_deg < 135.0:
        return "quasi_perpendicular"
    return "quasi_parallel"


def _load_day(day: str, data_dir: str | Path, cutoff_hz: float) -> pd.DataFrame:
    df, _ = read_dat_files_df([f"{day}/00:00:00", f"{day}/23:59:59"], str(data_dir))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["time", "Bx", "By", "Bz"]).sort_values("time").copy()
    for component in ("Bx", "By", "Bz"):
        df[f"{component}_bg"] = butterworth_filter(
            df[component].to_numpy(dtype=float), cutoff_hz, 1.0
        )
    bt = df["Bt"].to_numpy(dtype=float, copy=True)
    fallback = np.linalg.norm(df[["Bx", "By", "Bz"]].to_numpy(dtype=float), axis=1)
    invalid = ~np.isfinite(bt) | (np.abs(bt) > 1e4)
    bt[invalid] = fallback[invalid]
    _, df["delta_bt"] = compute_delta_bt_from_total(bt, cutoff_freq=cutoff_hz, fs=1.0)
    return df


def _event_physics_row(
    event: SampleEvent,
    samples: pd.DataFrame,
    sample_times_ns: np.ndarray,
    *,
    phi: float,
    v_sw_km_s: float,
    n_sw_cm3: float,
    shock_model: BowShockModel,
) -> dict | None:
    """Calculate one event using searchsorted instead of scanning a full day."""

    start_ns = event.start.value
    end_ns = event.end.value
    left = int(np.searchsorted(sample_times_ns, start_ns, side="left"))
    right = int(np.searchsorted(sample_times_ns, end_ns, side="right"))
    if right <= left:
        return None
    window = samples.iloc[left:right]
    delta = window["delta_bt"].to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return None
    delta_rms_nt = float(np.sqrt(np.mean(delta**2)))
    free_energy_j_m3 = float((delta_rms_nt * 1e-9) ** 2 / (2.0 * MU0))
    icw_energy_j_m3 = phi * free_energy_j_m3
    b_bg = np.nanmean(window[["Bx_bg", "By_bg", "Bz_bg"]].to_numpy(dtype=float), axis=0)
    b_norm_nt = float(np.linalg.norm(b_bg))
    if not np.isfinite(b_norm_nt) or b_norm_nt <= 0:
        return None

    midpoint_ns = event.midpoint.value
    local_times = sample_times_ns[left:right]
    midpoint_index = left + int(np.argmin(np.abs(local_times - midpoint_ns)))
    mid = samples.iloc[midpoint_index]
    pos_km = np.array([mid["pos_x"], mid["pos_y"], mid["pos_z"]], dtype=float)
    pos_rv = pos_km / R_VENUS
    pos_model_rv = vso_to_aberrated_vso(pos_rv)
    b_bg_model = vso_to_aberrated_vso(b_bg)
    try:
        shock_intersection, field_line_distance = field_line_bow_shock_intersection(
            pos_model_rv, b_bg_model, shock_model
        )
        shock_normal = bow_shock_normal(shock_intersection, shock_model)
        theta_bn = field_normal_angle_deg(b_bg_model, shock_normal)
        shock_class = classify_shock_angle(theta_bn)
    except ValueError:
        shock_intersection = np.full(3, np.nan)
        shock_normal = np.full(3, np.nan)
        field_line_distance = np.nan
        theta_bn = np.nan
        # Latest definition: an event without a modeled field-line/bow-shock
        # intersection is included in the quasi-perpendicular population.
        shock_class = "quasi_perpendicular"
    injection_velocity = injection_velocity_vso_km_s(v_sw_km_s)
    injection_speed_km_s = float(np.linalg.norm(injection_velocity))
    alpha_deg = injection_angle_deg(b_bg, injection_velocity)
    cone_deg = cone_angle_deg(b_bg)
    clock_deg = clock_angle_deg(b_bg)
    cos_alpha = float(np.cos(np.radians(alpha_deg)))
    angle_factor = (1.0 + cos_alpha) ** 2 + (1.0 - cos_alpha) ** 2
    va = (b_norm_nt * 1e-9) / np.sqrt(MU0 * (n_sw_cm3 * 1e6) * M_PROTON)
    denominator = M_PROTON * (injection_speed_km_s * 1e3) * va * angle_factor
    density_m3 = 4.0 * icw_energy_j_m3 / denominator
    return {
        "start": event.start,
        "end": event.end,
        "midpoint": event.midpoint,
        "duration_s": event.duration_s,
        "score": event.score,
        "VSO_X_Rv": pos_rv[0],
        "VSO_Y_Rv": pos_rv[1],
        "VSO_Z_Rv": pos_rv[2],
        "VSO_R_Rv": float(np.linalg.norm(pos_rv)),
        "AVSO_X_Rv": pos_model_rv[0],
        "AVSO_Y_Rv": pos_model_rv[1],
        "AVSO_Z_Rv": pos_model_rv[2],
        "AVSO_R_cyl_Rv": float(np.hypot(pos_model_rv[1], pos_model_rv[2])),
        "B_bg_x_nT": b_bg[0],
        "B_bg_y_nT": b_bg[1],
        "B_bg_z_nT": b_bg[2],
        "B_bg_mean_nT": b_norm_nt,
        "cone_angle_deg": cone_deg,
        "clock_angle_deg": clock_deg,
        "injection_velocity_x_km_s": injection_velocity[0],
        "injection_velocity_y_km_s": injection_velocity[1],
        "injection_velocity_z_km_s": injection_velocity[2],
        "injection_speed_km_s": injection_speed_km_s,
        "injection_angle_deg": alpha_deg,
        "injection_angle_70_110": bool(70.0 < alpha_deg < 110.0),
        "shock_intersection_X_Rv": shock_intersection[0],
        "shock_intersection_Y_Rv": shock_intersection[1],
        "shock_intersection_Z_Rv": shock_intersection[2],
        "field_line_distance_Rv": field_line_distance,
        "shock_normal_x": shock_normal[0],
        "shock_normal_y": shock_normal[1],
        "shock_normal_z": shock_normal[2],
        "deltaB_rms_nT": delta_rms_nt,
        "free_energy_J_m3": free_energy_j_m3,
        "icw_energy_J_m3": icw_energy_j_m3,
        "n_H_m3": density_m3,
        "n_H_cm3": density_m3 / 1e6,
        "theta_Bn_deg": theta_bn,
        "shock_class": shock_class,
    }


def _process_event_group(
    day_keys: tuple[str, ...],
    events: list[SampleEvent],
    *,
    data_dir: str | Path,
    cutoff_hz: float,
    phi: float,
    v_sw_km_s: float,
    n_sw_cm3: float,
    shock_model: BowShockModel,
) -> list[dict]:
    frames = [_load_day(day, data_dir, cutoff_hz) for day in day_keys]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return []
    samples = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    samples = samples.sort_values("time").reset_index(drop=True)
    sample_times_ns = (
        pd.to_datetime(samples["time"], utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    )
    rows = []
    for event in events:
        row = _event_physics_row(
            event,
            samples,
            sample_times_ns,
            phi=phi,
            v_sw_km_s=v_sw_km_s,
            n_sw_cm3=n_sw_cm3,
            shock_model=shock_model,
        )
        if row is not None:
            rows.append(row)
    return rows


def calculate_event_physics(
    events: list[SampleEvent],
    *,
    data_dir: str | Path,
    cutoff_hz: float = 0.01,
    phi: float = DEFAULT_ICW_ENERGY_TRANSFER_EFFICIENCY,
    v_sw_km_s: float = DEFAULT_SOLAR_WIND_SPEED_KM_S,
    n_sw_cm3: float = DEFAULT_SOLAR_WIND_DENSITY_CM3,
    shock_model: BowShockModel = BOW_SHOCK_SOLAR_MINIMUM,
    workers: int | None = None,
    progress_callback: Callable[[int, int, int, int, int], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Calculate event physics grouped by date with bounded parallelism.

    ``progress_callback`` receives completed groups, total groups, completed
    events, total events, and successfully calculated rows.
    """

    grouped: dict[tuple[str, ...], list[SampleEvent]] = defaultdict(list)
    for event in events:
        day_keys = tuple(
            day.strftime("%Y-%m-%d")
            for day in pd.date_range(event.start.normalize(), event.end.normalize(), freq="D")
        )
        grouped[day_keys].append(event)

    group_items = list(grouped.items())
    total_groups = len(group_items)
    total_events = len(events)
    if workers is None:
        workers = min(6, max(1, (os.cpu_count() or 2) // 2))
    workers = max(1, min(int(workers), total_groups or 1))
    rows: list[dict] = []
    completed_groups = 0
    completed_events = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sample-physics") as pool:
        futures = {
            pool.submit(
                _process_event_group,
                day_keys,
                group_events,
                data_dir=data_dir,
                cutoff_hz=cutoff_hz,
                phi=phi,
                v_sw_km_s=v_sw_km_s,
                n_sw_cm3=n_sw_cm3,
                shock_model=shock_model,
            ): len(group_events)
            for day_keys, group_events in group_items
        }
        for future in as_completed(futures):
            if cancel_callback is not None:
                cancel_callback()
            rows.extend(future.result())
            completed_groups += 1
            completed_events += futures[future]
            if progress_callback is not None:
                progress_callback(
                    completed_groups,
                    total_groups,
                    completed_events,
                    total_events,
                    len(rows),
                )
    return pd.DataFrame(rows)
