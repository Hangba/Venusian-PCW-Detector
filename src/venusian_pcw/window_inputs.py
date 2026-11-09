"""Materialize the paper's fixed window membership on the BAM data source."""

import numpy as np
import pandas as pd

from venus_express.icw_inference import _build_joint_profile_inference
from venus_express.repro.raw import cache, require_days, raw_root
from venus_express.repro.settings import BUILD, INPUTS, checked_file, read, sha256, write


def window_features(source, start):
    end = start + pd.Timedelta(seconds=80)
    features = source.load_features_for_trange(
        [v.strftime("%Y-%m-%d/%H:%M:%S") for v in (start, end)]
    )
    names = ("bfac_x", "bfac_y", "ellip", "wna_deg", "fcp_curve", "freq_pol")
    x, y, ellipticity, wna, fcp, freq = (features[name] for name in names)
    if len(fcp) < 2:
        raise ValueError(f"Insufficient valid window samples at {start.isoformat()}")
    profile = _build_joint_profile_inference(x, y, ellipticity, wna, fcp, freq).mean(axis=-1)
    finite = fcp[np.isfinite(fcp)]
    if not len(finite) or not np.isfinite(profile).all():
        raise ValueError(f"Nonfinite model input at {start.isoformat()}")
    kth = (len(finite) - 1) // 2
    return profile, np.partition(finite, kth)[kth], freq


def materialize(stage, role, day=None):
    stem = f"{stage}_{role}" if stage else "fixed_test"
    manifest = INPUTS / "window_lists" / f"{stem}.csv.gz"
    frame = pd.read_csv(manifest, float_precision="round_trip")
    starts = pd.to_datetime(frame.win_start, unit="ns", utc=True)
    if day:
        keep = starts.dt.strftime("%Y-%m-%d").eq(day)
        frame = frame.loc[keep].reset_index(drop=True)
        starts = starts[keep].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No listed windows for the selected date and split")
    ends = starts + pd.Timedelta(seconds=80)
    require_days(set(starts.dt.strftime("%Y-%m-%d")) | set(ends.dt.strftime("%Y-%m-%d")))
    source = cache()
    profiles = np.empty((len(frame), 4, 54), dtype=np.float32)
    fcp = np.empty(len(frame), dtype=np.float32)
    freq = None
    for i, row in enumerate(np.argsort(starts.to_numpy()), 1):
        profiles[row], fcp[row], axis = window_features(source, starts.iloc[row])
        if freq is None:
            freq = axis
        else:
            np.testing.assert_array_equal(freq, axis)
        if i % 500 == 0 or i == len(frame):
            print(f"{stem} {i}/{len(frame)} windows", flush=True)
    output = BUILD / "window_features"
    if day:
        output = output / "diagnostic" / day
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        X_mean=profiles,
        fcp_median=fcp,
        freq_pol=freq,
        y=frame.y.to_numpy(dtype=np.float32),
        w=frame.w.to_numpy(dtype=np.float32),
        group=frame.group.to_numpy(dtype=str),
        source=frame.source.to_numpy(dtype=str),
        win_start=frame.win_start.to_numpy(dtype=np.int64),
    )
    tmp.replace(path)
    write(
        path.with_suffix(".json"),
        dict(
            window_list_sha256=sha256(manifest),
            sha256=sha256(path),
            raw=str(raw_root()),
            rows=len(frame),
            day=day,
        ),
    )
    return path


def prepared_path(stage, role):
    stem = f"{stage}_{role}" if stage else "fixed_test"
    path = BUILD / "window_features" / f"{stem}.npz"
    metadata = read(path.with_suffix(".json"))
    manifest = INPUTS / "window_lists" / f"{stem}.csv.gz"
    if metadata["day"] is not None or metadata["window_list_sha256"] != sha256(manifest):
        raise ValueError(f"Rebuild the complete window list for {stem}")
    return checked_file(path, metadata["sha256"])
