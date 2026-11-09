"""Local, read-only VEX BAM provider for paper replay."""

import hashlib
import os
from pathlib import Path

import numpy as np

from venus_express.cache import ICWFeatureCache
from .settings import BUILD, INPUTS, read


def raw_root():
    raw = os.environ.get("VENUS_EXPRESS_RAW_DATA_DIR")
    if not raw or not Path(raw).expanduser().is_dir():
        raise ValueError("Set VENUS_EXPRESS_RAW_DATA_DIR to the local 1 Hz VSO BAM directory")
    return Path(raw).expanduser().resolve()


class PaperFeatureCache(ICWFeatureCache):
    def _build_daily_features(self, date_str):
        features = super()._build_daily_features(date_str)
        # Published inference read the serialized cache. Use the same precision
        # on the first read of a freshly built day and on subsequent cache hits.
        features["ellip"] = features["ellip"].astype(np.float16).astype(np.float32)
        wna = np.clip(np.rint(features["wna_deg"] * 2), 0, 180).astype(np.uint8)
        features["wna_deg"] = wna.astype(np.float32) * 0.5
        return features


def cache():
    raw = raw_root()
    namespace = hashlib.sha256(os.fsencode(raw)).hexdigest()[:16]
    return PaperFeatureCache(
        str(raw),
        str(BUILD / "feature_cache" / namespace),
        64,
        memory_cache_days=2,
        preload_days_ahead=0,
    )


def require_days(days):
    root = raw_root()
    found = {p.name[4:14] for p in root.glob("BAM_*T*_VSO_SAP*.dat")}
    missing = sorted(set(days) - found)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} required UTC dates are missing from {root}; first: {missing[:5]}"
        )


def check_inventory(days):
    wanted = set(days)
    root = raw_root()
    require_days(wanted)
    expected = {}
    for record in read(INPUTS / "mag_inventory.json"):
        if record["day"] in wanted:
            expected[record["name"]] = record["bytes"]
    actual = {
        p.name: p.stat().st_size
        for p in root.glob("BAM_*T*_VSO_SAP*.dat")
        if p.name[4:14] in wanted
    }
    if expected != actual:
        changed = sorted(
            name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name)
        )
        raise ValueError(f"BAM file names or sizes differ from the paper inventory: {changed[:5]}")
