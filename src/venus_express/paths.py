"""Canonical repository paths.

All paths are anchored to the repository, never to the process working
directory.  Environment overrides make large local stores relocatable without
embedding machine-specific absolute paths in source code.
"""

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = _path_from_env(
    "VENUS_EXPRESS_PROJECT_ROOT", Path(__file__).resolve().parents[2]
)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
TMP_DIR = PROJECT_ROOT / "tmp"

HISTORICAL_RUNS_DIR = RUNS_DIR / "legacy_output"
VEX_DATA_DIR = _path_from_env(
    "VENUS_EXPRESS_RAW_DATA_DIR", RAW_DATA_DIR / "VEX_1Hz"
)
CHECKPOINTS_DIR = _path_from_env(
    "VENUS_EXPRESS_CHECKPOINTS_DIR", OUTPUTS_DIR / "models" / "checkpoints"
)
CATALOGS_DIR = PROCESSED_DATA_DIR / "catalogs"
