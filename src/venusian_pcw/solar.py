"""Read optional SILSO V2 observations supplied by the reader."""

import os
from pathlib import Path

import pandas as pd

from venus_express.repro.settings import INPUTS


def annual_sunspots():
    location = os.environ.get("VENUS_EXPRESS_SILSO_YEARLY", INPUTS / "silso.csv")
    sun = pd.read_csv(
        Path(location).expanduser(),
        sep=";",
        header=None,
        usecols=[0, 1],
        names=["year", "sunspot_number"],
    )
    sun = sun[sun.sunspot_number.ge(0)].copy()
    sun["year"] = sun.year.astype(int)
    return sun
