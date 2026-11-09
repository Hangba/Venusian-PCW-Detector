"""Build and query sparse VEX R-X observation-time caches."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from venus_express.constants import R_VENUS
from venus_express.paths import CACHE_DIR
from venus_express.physics.bow_shock import BOW_SHOCK_ABERRATION_DEG

RX_EXPOSURE_SCHEMA_VERSION = 1
DEFAULT_RX_BIN_SIZE_RV = 0.25
DEFAULT_POSITION_STRIDE = 10
DEFAULT_RX_EXPOSURE_PATH = CACHE_DIR / "rx_exposure_0p25rv.npz"


def _source_signature(paths: list[Path]) -> str:
    return json.dumps(
        [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths],
        separators=(",", ":"),
    )


def _is_valid(values: tuple[float, ...]) -> bool:
    invalid_flag = 99999.999
    return all(
        math.isfinite(value) and not math.isclose(value, invalid_flag, abs_tol=1e-3)
        for value in values
    )


def _bin_day(
    day: str,
    path_strings: list[str],
    bin_size_rv: float,
    position_stride: int,
    stop_at: bytes | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, str]:
    """Process one day in a child process and return sparse exposure bins."""

    paths = [Path(value) for value in path_strings]
    counts: dict[tuple[int, int], int] = defaultdict(int)
    angle = math.radians(BOW_SHOCK_ABERRATION_DEG)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    bin_size_km = bin_size_rv * R_VENUS

    for path in paths:
        block_records = 0
        block_valid_records = 0
        block_bin: tuple[int, int] | None = None
        with path.open("rb") as handle:
            for raw_line in handle:
                stripped = raw_line.lstrip()
                if not stripped or stripped.startswith(b"#"):
                    continue
                if stop_at is not None:
                    stamp = stripped.split(maxsplit=1)[0]
                    if stamp > stop_at:
                        break
                valid_record = b"99999.999" not in stripped
                if valid_record:
                    block_valid_records += 1
                if valid_record and block_bin is None:
                    parts = stripped.split()
                    if len(parts) >= 13:
                        magnetic = tuple(float(value) for value in parts[6:9])
                        position = tuple(float(value) for value in parts[10:13])
                        if _is_valid((*magnetic, *position)):
                            x_km, y_km, z_km = position
                            x_avso_km = cosine * x_km - sine * y_km
                            y_avso_km = sine * x_km + cosine * y_km
                            block_bin = (
                                math.floor(x_avso_km / bin_size_km),
                                math.floor(math.hypot(y_avso_km, z_km) / bin_size_km),
                            )
                block_records += 1
                if block_records == position_stride:
                    if block_bin is not None:
                        counts[block_bin] += block_valid_records
                    block_records = 0
                    block_valid_records = 0
                    block_bin = None
        if block_records and block_bin is not None:
            counts[block_bin] += block_valid_records

    bins = sorted(counts)
    x_bin = np.asarray([item[0] for item in bins], dtype=np.int16)
    r_bin = np.asarray([item[1] for item in bins], dtype=np.uint16)
    observed_seconds = np.asarray([counts[item] for item in bins], dtype=np.int32)
    return day, x_bin, r_bin, observed_seconds, _source_signature(paths)


def _write_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _valid_daily_shard(
    path: Path,
    *,
    signature: str,
    bin_size_rv: float,
    position_stride: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                int(payload["schema_version"].item()) == RX_EXPOSURE_SCHEMA_VERSION
                and str(payload["source_signature"].item()) == signature
                and float(payload["bin_size_rv"].item()) == bin_size_rv
                and int(payload["position_stride"].item()) == position_stride
            )
    except (KeyError, OSError, ValueError):
        return False


def build_rx_exposure_cache(
    data_dir: str | Path,
    output_path: str | Path = DEFAULT_RX_EXPOSURE_PATH,
    *,
    bin_size_rv: float = DEFAULT_RX_BIN_SIZE_RV,
    position_stride: int = DEFAULT_POSITION_STRIDE,
    workers: int | None = None,
    rebuild: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
) -> Path:
    """Build resumable daily shards and one sparse mission-level cache."""

    if bin_size_rv <= 0 or position_stride <= 0:
        raise ValueError("bin_size_rv and position_stride must be positive")
    source = Path(data_dir)
    output = Path(output_path)
    day_dir = output.with_suffix("").with_name(output.stem + "_days")
    file_index: dict[str, list[Path]] = defaultdict(list)
    for path in source.glob("BAM_*T*.dat"):
        if len(path.name) >= 14:
            file_index[path.name[4:14]].append(path)
    if not file_index:
        raise ValueError(f"No BAM files found in {source}")
    for paths in file_index.values():
        paths.sort()

    pending: list[tuple[str, list[Path]]] = []
    for day, paths in sorted(file_index.items()):
        shard = day_dir / f"{day}.npz"
        signature = _source_signature(paths)
        if rebuild or not _valid_daily_shard(
            shard,
            signature=signature,
            bin_size_rv=bin_size_rv,
            position_stride=position_stride,
        ):
            pending.append((day, paths))

    worker_count = max(1, min(workers or min(6, os.cpu_count() or 1), len(pending) or 1))
    completed = len(file_index) - len(pending)
    if progress_callback is not None:
        progress_callback(completed, len(file_index), "cached")
    if pending:
        pool = ProcessPoolExecutor(max_workers=worker_count)
        futures: dict[object, str] = {}
        pending_iter = iter(pending)
        try:
            for _ in range(min(len(pending), worker_count * 2)):
                day, paths = next(pending_iter)
                futures[
                    pool.submit(
                        _bin_day,
                        day,
                        [str(path) for path in paths],
                        bin_size_rv,
                        position_stride,
                    )
                ] = day
            while futures:
                if cancel_callback is not None:
                    cancel_callback()
                future = next(as_completed(futures))
                futures.pop(future)
                day, x_bin, r_bin, observed_seconds, signature = future.result()
                _write_npz_atomic(
                    day_dir / f"{day}.npz",
                    schema_version=np.array(RX_EXPOSURE_SCHEMA_VERSION),
                    source_signature=np.array(signature),
                    bin_size_rv=np.array(bin_size_rv),
                    position_stride=np.array(position_stride),
                    x_bin=x_bin,
                    r_bin=r_bin,
                    observed_seconds=observed_seconds,
                )
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(file_index), day)
                if cancel_callback is not None:
                    cancel_callback()
                try:
                    day, paths = next(pending_iter)
                except StopIteration:
                    continue
                futures[
                    pool.submit(
                        _bin_day,
                        day,
                        [str(path) for path in paths],
                        bin_size_rv,
                        position_stride,
                    )
                ] = day
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    dates: list[str] = []
    offsets = [0]
    all_x: list[np.ndarray] = []
    all_r: list[np.ndarray] = []
    all_seconds: list[np.ndarray] = []
    totals: dict[tuple[int, int], int] = defaultdict(int)
    for day in sorted(file_index):
        with np.load(day_dir / f"{day}.npz", allow_pickle=False) as shard:
            x_bin = shard["x_bin"].astype(np.int16, copy=False)
            r_bin = shard["r_bin"].astype(np.uint16, copy=False)
            seconds = shard["observed_seconds"].astype(np.int32, copy=False)
        dates.append(day)
        all_x.append(x_bin)
        all_r.append(r_bin)
        all_seconds.append(seconds)
        offsets.append(offsets[-1] + len(x_bin))
        for x_value, r_value, duration in zip(x_bin, r_bin, seconds, strict=True):
            totals[(int(x_value), int(r_value))] += int(duration)

    total_bins = sorted(totals)
    _write_npz_atomic(
        output,
        schema_version=np.array(RX_EXPOSURE_SCHEMA_VERSION),
        created_utc=np.array(datetime.now(UTC).isoformat()),
        bin_size_rv=np.array(bin_size_rv),
        position_stride=np.array(position_stride),
        aberration_deg=np.array(BOW_SHOCK_ABERRATION_DEG),
        dates=np.asarray(dates, dtype="U10"),
        day_offsets=np.asarray(offsets, dtype=np.int64),
        x_bin=np.concatenate(all_x),
        r_bin=np.concatenate(all_r),
        observed_seconds=np.concatenate(all_seconds),
        total_x_bin=np.asarray([item[0] for item in total_bins], dtype=np.int16),
        total_r_bin=np.asarray([item[1] for item in total_bins], dtype=np.uint16),
        total_observed_seconds=np.asarray([totals[item] for item in total_bins], dtype=np.int64),
    )
    load_rx_exposure_cache.cache_clear()
    return output


@dataclass(frozen=True)
class RxExposureCache:
    bin_size_rv: float
    dates: np.ndarray
    day_offsets: np.ndarray
    x_bin: np.ndarray
    r_bin: np.ndarray
    observed_seconds: np.ndarray

    def exposure_by_bin(
        self,
        start_date: str,
        end_date: str,
        *,
        target_bin_size_rv: float | None = None,
    ) -> dict[tuple[int, int], int]:
        """Sum daily exposure and optionally coarsen integer-aligned bins."""

        target = self.bin_size_rv if target_bin_size_rv is None else target_bin_size_rv
        factor = target / self.bin_size_rv
        rounded_factor = round(factor)
        if rounded_factor < 1 or not math.isclose(factor, rounded_factor, abs_tol=1e-9):
            raise ValueError("target bin size must be an integer multiple of the cache bin size")
        selected = np.flatnonzero((self.dates >= start_date) & (self.dates <= end_date))
        totals: dict[tuple[int, int], int] = defaultdict(int)
        for day_index in selected:
            left = int(self.day_offsets[day_index])
            right = int(self.day_offsets[day_index + 1])
            for x_value, r_value, duration in zip(
                self.x_bin[left:right],
                self.r_bin[left:right],
                self.observed_seconds[left:right],
                strict=True,
            ):
                key = (int(x_value) // rounded_factor, int(r_value) // rounded_factor)
                totals[key] += int(duration)
        return dict(totals)


@lru_cache(maxsize=4)
def load_rx_exposure_cache(path: str | Path = DEFAULT_RX_EXPOSURE_PATH) -> RxExposureCache:
    """Load the runtime cache once per process."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if int(payload["schema_version"].item()) != RX_EXPOSURE_SCHEMA_VERSION:
            raise ValueError("Unsupported R-X exposure-cache schema")
        return RxExposureCache(
            bin_size_rv=float(payload["bin_size_rv"].item()),
            dates=payload["dates"].copy(),
            day_offsets=payload["day_offsets"].copy(),
            x_bin=payload["x_bin"].copy(),
            r_bin=payload["r_bin"].copy(),
            observed_seconds=payload["observed_seconds"].copy(),
        )
