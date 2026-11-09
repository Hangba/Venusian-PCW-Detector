"""Recover the paper's temporal alignment of the AVSO exposure grid."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np
import pandas as pd

from venus_express.constants import R_VENUS
from venus_express.physics.bow_shock import vso_to_aberrated_vso
from venus_express.rx_exposure_cache import _bin_day

from .raw import require_days, raw_root
from .settings import BUILD, INPUTS, write


def line_bin(line):
    parts = line.split()
    vals = np.array([float(parts[i]) for i in [6, 7, 8, 10, 11, 12]])
    assert np.isfinite(vals).all() and not np.any(np.isclose(vals, 99999.999, atol=0.001, rtol=0))
    pos = vso_to_aberrated_vso(vals[3:] / R_VENUS)
    return math.floor(pos[0] / 0.25), math.floor(np.hypot(pos[1], pos[2]) / 0.25)


def tail_record_bin(day):
    """The excluded fractional last second is assigned to its original 10-record cache block."""
    paths = sorted(raw_root().glob(f"BAM_{day}T*.dat"))
    assert len(paths) == 1
    p = paths[0]
    with p.open("rb") as f:
        n = min(p.stat().st_size, 65536)
        f.seek(-n, 2)
        raw = f.read()
        lines = raw.splitlines()
        if n < p.stat().st_size:
            lines = lines[1:]
    lines = [x for x in lines if x.strip() and not x.lstrip().startswith(b"#")]
    cutoff = pd.Timestamp(f"{day}T23:59:59", tz="UTC").timestamp()
    excluded = [
        x
        for x in lines[-30:]
        if b"99999.999" not in x
        and pd.to_datetime(
            x.split()[0].decode(), format="%Y-%m-%dT%H:%M:%S.%f", utc=True
        ).timestamp()
        > cutoff
    ]
    assert len(excluded) == 1, (day, len(excluded))
    last = excluded[0]
    bins = [line_bin(x) for x in lines[-10:] if b"99999.999" not in x]
    if last == lines[-1] and bins and len(set(bins)) == 1:
        return (*bins[0], "last10_same_bin", str(p))
    # Rare cell crossing or invalid tail: recover the precise first-valid row of the original block.
    lines = [
        x for x in p.read_bytes().splitlines() if x.strip() and not x.lstrip().startswith(b"#")
    ]
    idx = next(i for i, x in enumerate(lines) if x == last)
    block = lines[(idx // 10) * 10 : (idx // 10 + 1) * 10]
    eligible = [x for x in block if b"99999.999" not in x]
    assert eligible
    return (*line_bin(eligible[0]), "exact_original_block", str(p))


def day_exposure(task):
    day, expected = task
    if expected == 0:
        return {}
    paths = sorted(str(p) for p in raw_root().glob(f"BAM_{day}T*.dat"))
    _, xs, rs, ss, _ = _bin_day(day, paths, bin_size_rv=0.25, position_stride=10)
    bins = {(int(x), int(r)): int(s) for x, r, s in zip(xs, rs, ss)}
    difference = sum(bins.values()) - expected
    if difference == 1:
        # The day feature window ends at 23:59:59.000. Assign the excluded fractional
        # last record to the bin of the first valid sample in its original 10-record block.
        x, r, _, _ = tail_record_bin(day)
        bins[x, r] -= 1
    elif difference:
        raise ValueError(f"{day}: {difference} observation seconds away from the frozen day count")
    result = Counter()
    for (x, r), seconds in bins.items():
        result[x // 2, r // 2] += seconds
    return result


def compute(workers):
    days = pd.read_csv(INPUTS / "mission/exposure_days.csv")
    require_days(days.day)
    tasks = list(zip(days.day, days.observed_seconds))
    totals = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, bins in enumerate(pool.map(day_exposure, tasks, chunksize=10), 1):
            totals.update(bins)
            if i % 100 == 0 or i == len(tasks):
                print(f"exposure {i}/{len(tasks)} days", flush=True)
    frozen = pd.read_csv(INPUTS / "mission/cluster_occurrence.csv")
    expected = {
        (round(x * 2), round(r * 2)): int(s)
        for x, r, s in zip(frozen.x_bin_min_rv, frozen.r_bin_min_rv, frozen.observed_seconds)
    }
    if totals != Counter(expected):
        raise ValueError("Raw AVSO grid differs from the frozen 0.5 Rv exposure grid")
    output = BUILD / "raw" / "exposure"
    output.mkdir(parents=True, exist_ok=True)
    frozen = frozen[["x_bin_min_rv", "r_bin_min_rv"]].assign(
        observed_seconds=[
            totals[round(x * 2), round(r * 2)]
            for x, r in zip(frozen.x_bin_min_rv, frozen.r_bin_min_rv)
        ]
    )
    frozen.to_csv(output / "grid.csv", index=False)
    write(
        output / "complete.json",
        {"raw": str(raw_root()), "observed_seconds": int(sum(totals.values()))},
    )
