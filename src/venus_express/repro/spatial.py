"""Restore frozen cluster midpoint positions from BAM."""
import numpy as np
import pandas as pd
from venus_express.constants import R_VENUS
from venus_express.physics.bow_shock import vso_to_aberrated_vso
from .raw import raw_root

def position_day(job):
    day, records = job
    out = []
    records = sorted(records, key=lambda r: r["representative_time_s"])
    todo = 0
    prev = None
    targets = [
        pd.Timestamp(r["representative_time_s"], unit="s", tz="UTC")
        .strftime("%Y-%m-%dT%H:%M:%S.%f")
        .encode()
        for r in records
    ]
    for p in sorted(raw_root().glob(f"BAM_{day}T*.dat")):
        with p.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(b"#") or b"99999.999" in line:
                    continue
                stamp = line.split(maxsplit=1)[0]
                while todo < len(records) and stamp >= targets[todo]:
                    r = records[todo]
                    target = float(r["representative_time_s"])
                    options = [x for x in [prev, line] if x is not None]
                    options = [
                        (
                            pd.to_datetime(
                                x.split()[0].decode(), format="%Y-%m-%dT%H:%M:%S.%f", utc=True
                            ).timestamp(),
                            x,
                        )
                        for x in options
                    ]
                    options = [
                        (t, x)
                        for t, x in options
                        if r["representative_start_s"] <= t <= r["representative_end_s"]
                    ]
                    assert options, (day, r)
                    time, line_chosen = min(options, key=lambda q: (abs(q[0] - target), q[0]))
                    parts = line_chosen.split()
                    values = np.array([float(parts[i]) for i in [6, 7, 8, 10, 11, 12]])
                    assert np.isfinite(values).all()
                    pos = values[3:] / R_VENUS
                    avso = vso_to_aberrated_vso(pos)
                    out.append(
                        dict(
                            cluster_id=r["cluster_id"],
                            representative_interval_id=r["representative_interval_id"],
                            representative_time_s=target,
                            raw_time_s=time,
                            nearest_time_difference_s=abs(time - target),
                            VSO_X_Rv=pos[0],
                            VSO_Y_Rv=pos[1],
                            VSO_Z_Rv=pos[2],
                            AVSO_X_Rv=avso[0],
                            AVSO_R_Rv=np.hypot(avso[1], avso[2]),
                            raw_file=str(p),
                        )
                    )
                    todo += 1
                prev = line
                if todo == len(records):
                    return day, out
    assert todo == len(records), (day, todo, len(records))
    return day, out
