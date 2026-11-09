"""Compute daily and spatial observation coverage directly from MAG records."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from venus_express.analysis import DEFAULT_POL_WINDOW_SIZE
from venus_express.rx_exposure_cache import _bin_day

from .raw import observation_days, raw_root, require_days
from .settings import BUILD, write


def day_exposure(day):
    paths = sorted(str(p) for p in raw_root().glob(f"BAM_{day}T*_VSO_SAP*.dat"))
    cutoff = f"{day}T23:59:59.000000".encode()
    _, xs, rs, ss, _ = _bin_day(day, paths, 0.25, 10, stop_at=cutoff)
    counts = Counter()
    if ss.sum() >= DEFAULT_POL_WINDOW_SIZE:
        for x, r, seconds in zip(xs, rs, ss):
            counts[int(x) // 2, int(r) // 2] += int(seconds)
    return day, counts


def compute(workers, days=None):
    days = sorted(set(days if days is not None else observation_days()))
    if not days:
        raise ValueError("Select at least one MAG date")
    require_days(days)
    totals, daily = Counter(), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (day, counts) in enumerate(pool.map(day_exposure, days, chunksize=10), 1):
            totals.update(counts)
            daily.append(dict(day=day, observed_seconds=sum(counts.values())))
            if i % 100 == 0 or i == len(days):
                print(f"exposure {i}/{len(days)} days", flush=True)
    output = BUILD / "raw" / "exposure"
    output.mkdir(parents=True, exist_ok=True)
    keys = ["x_bin_min_rv", "r_bin_min_rv", "observed_seconds"]
    pd.DataFrame(
        [(x / 2, r / 2, count) for (x, r), count in sorted(totals.items())], columns=keys
    ).to_csv(output / "grid.csv", index=False)
    pd.DataFrame(daily).to_csv(output / "days.csv", index=False)
    write(
        output / "complete.json",
        {
            "raw": str(raw_root()),
            "days": days,
            "observed_seconds": sum(totals.values()),
        },
    )
