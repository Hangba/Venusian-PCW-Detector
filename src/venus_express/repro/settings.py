"""Paths for the standalone paper repository and its external inputs."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "inputs"
BUILD = ROOT / "build"
FIGURES = BUILD / "figures"



def display_path(path):
    """Show repository-relative paths and omit external parent directories."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def redact_output(text, *external_paths):
    """Hide the working directory and local input paths in user-facing logs."""
    replacements = {str(Path.home()): "~", str(ROOT): "."}
    for name in ("VENUS_EXPRESS_RAW_DATA_DIR", "VENUS_EXPRESS_TRAINING_INPUT_DIR"):
        if value := os.environ.get(name):
            replacements[str(Path(value).expanduser().resolve())] = f"<{name}>"
    for path in external_paths:
        if isinstance(path, Path) and path.is_absolute():
            replacements[str(path)] = display_path(path)
    for path, label in sorted(replacements.items(), key=lambda item: -len(item[0])):
        text = text.replace(path, label)
    return text


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
