"""Rebuild the physical event catalogue from BAM and annotated event intervals."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from venus_express.constants import M_PROTON, R_VENUS
from venus_express.physics.bow_shock import BOW_SHOCK_SOLAR_MINIMUM
from venus_express.physics.field_connection import event_field_connection
from venus_express.repro.continuum import background
from venus_express.repro.raw import cache, raw_root, require_days
from venus_express.repro.settings import BUILD, INPUTS, display_path, redact_output, sha256, write
from venus_express.sample_analysis.calculations.band_energy import tangential_band_peak_energy
from venus_express.sample_analysis.calculations.events import SampleEvent, merge_events, event_table
from venus_express.sample_analysis.calculations.paper_integrated_density import (
    MU0,
    DEFAULT_PHOTOIONIZATION_RATE_S,
    _resample_uniform_cadence,
    integrate_paper_icw_band,
    neutral_h_from_pickup_density,
    paper_perpendicular_psd,
)
from venus_express.sample_analysis.calculations.physics import _event_physics_row, _load_day


def read_events(path):
    frame = pd.read_csv(path)
    for key in ("start", "end"):
        frame[key] = pd.to_datetime(frame[key], format="mixed", utc=True)
    if (
        frame.empty
        or frame[["start", "end"]].isna().any().any()
        or (frame.end <= frame.start).any()
    ):
        raise ValueError("Events need valid start and end UTC timestamps")
    if frame.duplicated(["start", "end"]).any():
        raise ValueError("Event intervals must be unique")
    if "plasma_region" not in frame:
        frame["plasma_region"] = "unknown"
    if not frame.plasma_region.isin(["solar_wind", "magnetosheath", "unknown"]).all():
        raise ValueError("plasma_region must be solar_wind, magnetosheath or unknown")
    if "crossing_region" not in frame:
        frame["crossing_region"] = "unknown"
    if "overlaps_recorded_sheath" not in frame:
        frame["overlaps_recorded_sheath"] = False
    if not frame.overlaps_recorded_sheath.isin([True, False]).all():
        raise ValueError("overlaps_recorded_sheath must be true or false")
    frame["overlaps_recorded_sheath"] = frame.overlaps_recorded_sheath.astype(bool)
    for key in ("start", "end"):
        if not frame[key].eq(frame[key].dt.floor("s")).all():
            raise ValueError(f"{key} must use whole UTC seconds")
    return frame.sort_values(["start", "end"]).reset_index(drop=True)


def utc_days(start, end):
    return tuple(
        day.date().isoformat() for day in pd.date_range(start.normalize(), end.normalize())
    )


def prepare_day(day):
    source = cache()
    existing = Path(source._date_cache_path(day)).is_file()
    features = source.build_or_load_daily_features(day)
    return {
        "day": day,
        "samples": len(features["times_s"]),
        "cache": display_path(source._date_cache_path(day)),
        "existing_cache": existing,
    }


def precompute(path, days, workers):
    if path:
        events = read_events(path)
        available = set().union(*(utc_days(r.start, r.end) for r in events.itertuples()))
        if days and not set(days) <= available:
            raise ValueError("Requested dates are outside the event list")
    else:
        available = set(days)
    require_days(available if not days else days)
    target = sorted(days or available)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(prepare_day, target), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(target):
                print(f"MAG daily cache {i}/{len(target)}", flush=True)
    write(BUILD / "cache_summary.json", rows)


def invert(features, context):
    components = tuple(np.asarray(features[k], float) for k in ("bfac_x", "bfac_y", "fcp_curve"))
    times = features["times_s"]
    band = integrate_paper_icw_band(*components, phi=0.30, sample_times_s=times)
    if not np.allclose(np.diff(times), 1.0, rtol=0, atol=0.02):
        _, components, _ = _resample_uniform_cadence(times, components, fs_hz=1.0)
    freq, psd, _ = paper_perpendicular_psd(components[0], components[1])
    lo, hi = band["paper_band_low_hz"], band["paper_band_high_hz"]
    model, diagnostics = background(freq, psd, lo, hi)
    if model is None:
        raise ValueError(f"continuum unavailable: {diagnostics['status']}")
    mask = (freq >= lo) & (freq <= hi) & (freq > 0)
    power = float(np.sum(psd[mask] - model[mask]) * (freq[1] - freq[0]))
    if power <= 0:
        raise ValueError("nonpositive net wave power")
    va = context["B_bg_mean_nT"] * 1e-9 / np.sqrt(MU0 * 10e6 * M_PROTON)
    angle = np.deg2rad(context["injection_angle_deg"])
    denom = M_PROTON * context["injection_speed_km_s"] * 1e3 * va * (2 + 2 * np.cos(angle) ** 2)
    pickup = 4 * (power * 1e-18 / (2 * MU0 * 0.30)) / denom
    neutral = neutral_h_from_pickup_density(
        pickup,
        band["paper_fgyro_hz"],
        n_gyro_periods=80,
        photoionization_rate_s=DEFAULT_PHOTOIONIZATION_RATE_S,
        n_sw_cm3=10.0,
        v_sw_km_s=350.0,
    )
    return dict(
        paper_fgyro_hz=band["paper_fgyro_hz"],
        n_Hplus_pickup_cm3=pickup / 1e6,
        n_H_neutral_cm3=neutral["n_H_neutral_cm3"],
        net_integrated_psd_nT2=power,
    )


def measure_group(task):
    days, rows = task
    samples = pd.concat([_load_day(day, raw_root(), 0.01) for day in days], ignore_index=True)
    if not samples.empty:
        samples = samples.sort_values("time").reset_index(drop=True)
        ns = (
            pd.to_datetime(samples.time, utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        )
    source = cache()
    results = []
    for annotations in rows:
        start, end = annotations["start"], annotations["end"]
        event = SampleEvent(start, end)
        record = dict(
            annotations,
            geometry_status="unavailable",
            band_energy_status="unavailable",
            density_status="outside_density_sample",
            B0ShockConnection="unavailable",
        )
        results.append(record)
        if samples.empty:
            continue
        context = _event_physics_row(
            event,
            samples,
            ns,
            phi=0.30,
            v_sw_km_s=350.0,
            n_sw_cm3=10.0,
            shock_model=BOW_SHOCK_SOLAR_MINIMUM,
        )
        if context is None:
            continue
        record.update(context, geometry_status="ok")
        record["day"] = event.midpoint.date().isoformat()
        record["year"] = event.midpoint.year
        record["alt_km"] = (record["VSO_R_Rv"] - 1) * R_VENUS
        section = samples.iloc[
            ns.searchsorted(start.value) : ns.searchsorted(end.value, side="right")
        ]
        connected = event_field_connection(
            section[["pos_x", "pos_y", "pos_z"]].to_numpy(float),
            section[["Bx_bg", "By_bg", "Bz_bg"]].to_numpy(float),
        )
        record["B0ShockConnection"] = connected["status"]
        trange = [stamp.strftime("%Y-%m-%d/%H:%M:%S") for stamp in (start, end)]
        try:
            features = source.load_features_for_trange(trange)
            record.update(
                tangential_band_peak_energy(
                    features["bfac_x"], features["bfac_y"], features["fcp_curve"]
                ),
                band_energy_status="ok",
            )
        except (ValueError, OSError) as exc:
            record["band_energy_status"] = f"unavailable: {redact_output(str(exc))}"
            continue
        if (
            record["plasma_region"] == "solar_wind"
            and record["VSO_X_Rv"] > 0
            and connected["status"] == "no_intersection"
        ):
            try:
                record.update(invert(features, context), density_status="ok")
            except ValueError as exc:
                record["density_status"] = f"unavailable: {redact_output(str(exc))}"
    return results


def compute(path, days, workers):
    path = Path(path) if path else INPUTS / "manual/event_list.csv"
    events = read_events(path)
    if days:
        events = events[events.start.dt.strftime("%Y-%m-%d").isin(days)]
        if events.empty:
            raise ValueError("No listed events start on the requested dates")
    groups = defaultdict(list)
    for row in events.to_dict("records"):
        groups[utc_days(row["start"], row["end"])].append(row)
    require_days(set().union(*groups))
    dest = BUILD / "raw_catalogue"
    if days:
        dest /= "diagnostic"
    dest.mkdir(parents=True, exist_ok=True)
    all_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(measure_group, groups.items()), 1):
            all_rows.extend(rows)
            if i % 25 == 0 or i == len(groups):
                print(f"Catalogue MAG days {i}/{len(groups)}", flush=True)
    frame = pd.DataFrame(all_rows).sort_values(["start", "end"]).reset_index(drop=True)
    frame.to_csv(dest / "events.csv", index=False)
    clusters = event_table(merge_events([SampleEvent(r.start, r.end) for r in events.itertuples()]))
    clusters.to_csv(dest / "manual_clusters_10min.csv", index=False)
    selected = frame[
        frame.band_energy_status.eq("ok")
        & frame.plasma_region.eq("solar_wind")
        & ~frame.overlaps_recorded_sheath
        & frame.crossing_region.isin(["solar_wind", "unknown"])
    ].copy()
    selected["plasma_region"] = selected.crossing_region
    density = frame[frame.density_status.eq("ok")].copy()
    event_list_sha = sha256(path)
    summary = dict(
        events=len(frame),
        energy_available=int(frame.band_energy_status.eq("ok").sum()),
        annual_energy_events=len(selected),
        density_events=len(density),
        manual_clusters_10min=len(clusters),
        source=display_path(path.resolve()),
        event_list_sha256=event_list_sha,
        mag="VENUS_EXPRESS_RAW_DATA_DIR",
    )
    if len(density) >= 3:
        from venus_express.repro.density import fit_with_bootstrap

        summary["density"] = fit_with_bootstrap(density, dest / "density")
    if len(selected) >= 15 and set(selected.year) == set(range(2006, 2015)):
        from .energy import compute as annual_energy

        from .solar import annual_sunspots

        solar = annual_sunspots()
        sun = None
        annual = solar.set_index("year").reindex(range(2006, 2015))
        if annual.sunspot_number.notna().all():
            sun = annual.sunspot_number.to_numpy()
        summary["wave_energy"] = annual_energy(selected, dest / "energy", sun=sun)
    write(dest / "summary.json", summary)
    print(summary)
