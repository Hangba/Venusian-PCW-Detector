"""Paths for the standalone paper repository and its external inputs."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "inputs"
BUILD = ROOT / "build"
FIGURES = BUILD / "figures"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked_file(path, digest):
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != digest:
        raise ValueError(f"File differs from the frozen paper input: {path}")
    return path


def training_file(spec):
    source = os.environ.get("VENUS_EXPRESS_TRAINING_INPUT_DIR")
    if not source:
        raise ValueError(
            "Set VENUS_EXPRESS_TRAINING_INPUT_DIR to the root of the frozen training inputs"
        )
    return checked_file(Path(source).expanduser() / spec["path"], spec["sha256"])
