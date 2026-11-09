"""Re-evaluate the 946 PCW event spectra with the paper's <=5% gap rule."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from venus_express.dataset import VenusExpressDataset
from venus_express.analysis_windows import export_analysis_geometry
from venus_express.physics.fac import convert_to_fac_coordinates
from venus_express.signal.fft_metrics import compute_perp_fft_metrics
from venus_express.repro.raw import raw_root, require_days
from venus_express.repro.settings import BUILD, INPUTS, write


def process(pair):
    days, items = pair
    data = VenusExpressDataset(
        str(raw_root()), [days[0] + "/00:00:00", days[-1] + "/23:59:59"]
    ).dataframe
    res = []
    for row, g in items:
        v = dict(row)
        az = row["az"]
        start, end = g.event_start, g.event_end
        try:
            raw = data[
                (data.time_ut >= g.analysis_start - pd.Timedelta(seconds=2))
                & (data.time_ut <= g.analysis_end + pd.Timedelta(seconds=2))
            ].copy()
            if len(raw) < 32:
                raise ValueError("insufficient_context")
            raw["grid"] = pd.DatetimeIndex(raw.time_ut).round("s")
            raw["offset_seconds"] = (raw.time_ut - raw.grid).abs().dt.total_seconds()
            dup = raw.grid.duplicated(keep=False)
            v["duplicate_seconds_in_event"] = int(
                raw.loc[dup & (raw.grid >= start) & (raw.grid < end), "grid"].nunique()
            )
            raw = (
                raw.sort_values(["grid", "offset_seconds"])
                .drop_duplicates("grid", keep="first")
                .set_index("grid")
                .sort_index()
            )
            grid = pd.date_range(start, end - pd.Timedelta(seconds=1), freq="1s")
            expected = len(grid)
            if expected < 16:
                raise ValueError("event_too_short")
            a = raw[["Bx", "By", "Bz"]].reindex(grid)
            present = np.isfinite(a.to_numpy()).all(axis=1)
            missing = int((~present).sum())
            frac = missing / expected
            v.update(
                expected_seconds=expected,
                missing_seconds=missing,
                missing_fraction=frac,
                interpolated=missing > 0,
            )
            if missing / expected > 0.05 + 1e-12:
                v.update(status="over_5pct_missing", reason="event raw cadence missing >5%")
                res.append(v)
                continue
            # Event clock only: use observations outside the interval as interpolation anchors.
            support = raw[["Bx", "By", "Bz"]]
            idx = support.index.union(grid).sort_values()
            support = support.reindex(idx)
            support = support.interpolate(method="time", limit_area="inside")
            event = support.reindex(grid)
            if not np.isfinite(event.to_numpy()).all():
                raise ValueError("unbracketed_missing")
            # Keep observed context; insert the filled event samples into it. No context gaps are filled.
            allidx = raw.index.union(grid).sort_values()
            full = raw[["Bx", "By", "Bz"]].reindex(allidx)
            full.loc[grid, ["Bx", "By", "Bz"]] = event.to_numpy()
            full = full.loc[(full.index >= g.analysis_start) & (full.index <= g.analysis_end)]
            if full.isna().any().any():
                raise ValueError("nonfinite_context")
            fac = convert_to_fac_coordinates(
                {k: full[k].to_numpy(float) for k in ["Bx", "By", "Bz"]},
                cutoff_freq=0.01 / az,
                fs=1,
            )
            m = full.index.isin(grid)
            x, y, z = [np.asarray(fac[k])[m] for k in ("B_fac_x", "B_fac_y", "B_fac_z")]
            bt = np.linalg.norm(event.to_numpy(float), axis=1)
            if len(x) != expected:
                raise ValueError("event_grid_alignment")
            ft = compute_perp_fft_metrics(
                x, y, z, bt, fs=1, pad_factor=2, smooth_points=1, fcp_band=(0.8 / az, 1.2 / az)
            )
            fci = ft["f_cp_ref"] / az
            if not np.isfinite(fci) or fci <= 0:
                raise ValueError("invalid_fci")
            freq = ft["freq"]
            psd = ft["psd_perp"]
            peak = float(freq[np.argmax(psd)])
            ratio = peak / fci
            v.update(
                status="in_band" if 0.8 <= ratio <= 1.2 else "out_of_band",
                fci_hz=fci,
                global_peak_hz=peak,
                peak_over_fci=ratio,
                band_peak_hz=float(ft["peak_freq"]),
                context_missing_not_filled=True,
                reason="",
            )
        except Exception as e:
            v.update(status="unavailable", reason=f"{type(e).__name__}: {e}")
        res.append(v)
    return res


def compute(workers, days, events_path=None):
    from .catalogue import read_events

    catalogue = read_events(events_path or INPUTS / "manual/event_list.csv")
    if days:
        catalogue = catalogue[catalogue.start.dt.strftime("%Y-%m-%d").isin(days)]
    if catalogue.empty:
        raise ValueError("No frozen PCW catalogue events on the requested dates")
    groups = defaultdict(list)
    for i, row in catalogue.iterrows():
        interval = (row.start.isoformat(), row.end.isoformat())
        duration = (pd.Timestamp(row.end) - pd.Timestamp(row.start)).total_seconds()
        geometry = export_analysis_geometry(
            interval, min_display_sec=max(duration / 0.6, 300), analysis_pad_sec=320
        )
        context = tuple(VenusExpressDataset._get_date_list(geometry.analysis_trange))
        groups[context].append(
            (dict(rank=i + 1, start_utc=interval[0], end_utc=interval[1], az=1), geometry)
        )
    require_days(catalogue.start.dt.strftime("%Y-%m-%d"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [event for events in pool.map(process, groups.items()) for event in events]
    frame = pd.DataFrame(rows).sort_values("rank")
    from .statistics import peak_summary

    result = peak_summary(frame)
    if not days and events_path is None:
        assert result == dict(events=946, usable=917, outside=245, below=223, above=22)
    BUILD.mkdir(exist_ok=True)
    frame.to_csv(BUILD / "raw_psd_audit.csv", index=False)
    write(BUILD / "raw_psd_summary.json", result)
    print(result)
