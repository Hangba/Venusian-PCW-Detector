"""Read the event-scale 1 Hz MAG panels from the configured BAM provider."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from venus_express.repro.raw import raw_root


def load_day(day, t0=None, t1=None):
    f = next(iter(sorted(raw_root().glob(f"BAM_{day}T*.dat"))), None)
    if f is None:
        raise FileNotFoundError(f"No MAG BAM file for {day}")
    cols = [
        "utc",
        "obt",
        "bix",
        "biy",
        "biz",
        "bit",
        "bx",
        "by",
        "bz",
        "bt",
        "x",
        "y",
        "z",
        "r",
        "q",
    ]
    df = pd.read_csv(f, comment="#", sep=r"\s+", names=cols)
    df["utc"] = pd.to_datetime(df.utc)
    df = df.set_index("utc")
    if t0 is not None:
        df = df.loc[t0:t1]
    for c in ["bx", "by", "bz", "bt"]:
        df.loc[df[c].abs() > 9e4, c] = np.nan
    return df


def fac(df, cutoff):
    b = df[["bx", "by", "bz"]].interpolate(limit=30).bfill().ffill().to_numpy()
    sos = butter(4, cutoff, btype="low", fs=1.0, output="sos")
    bg = sosfiltfilt(sos, b, axis=0)
    e_par = bg / np.linalg.norm(bg, axis=1, keepdims=True)
    db = b - bg
    par = (db * e_par).sum(1)
    perp = db - par[:, None] * e_par
    # stable perpendicular basis from the mean field
    m = e_par.mean(0)
    m /= np.linalg.norm(m)
    a = np.cross(m, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    c = np.cross(m, a)
    return perp @ a, perp @ c, par, np.linalg.norm(bg, axis=1)
