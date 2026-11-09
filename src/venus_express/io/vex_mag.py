"""Readers for Venus Express MAG ASCII data files."""

import glob
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_INVALID_FLAG = 99999.999
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

_file_index: Optional[Dict[str, List[str]]] = None
_file_index_dir: Optional[str] = None


def _build_file_index(data_dir: str) -> Dict[str, List[str]]:
    """Build a date -> file list mapping by scanning the directory once."""
    global _file_index, _file_index_dir
    if _file_index is not None and _file_index_dir == data_dir:
        return _file_index

    index: Dict[str, List[str]] = {}
    pattern = os.path.join(data_dir, "BAM_*.dat")
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        # BAM_YYYY-MM-DDTHH-MM-SS_...
        if len(basename) >= 14:
            date_str = basename[4:14]  # "YYYY-MM-DD"
            index.setdefault(date_str, []).append(path)

    for key in index:
        index[key].sort()

    _file_index = index
    _file_index_dir = data_dir
    return index


def _date_range_for_trange(trange):
    start_time = datetime.strptime(trange[0], "%Y-%m-%d/%H:%M:%S")
    end_time = datetime.strptime(trange[1], "%Y-%m-%d/%H:%M:%S")

    date_list = []
    current_date = start_time.date()
    end_date = end_time.date()
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return date_list


def _read_single_dat(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[datetime], Optional[datetime]]:
    """Read a single .dat file in one pass, extracting header times and data."""
    try:
        file_start: Optional[datetime] = None
        file_end: Optional[datetime] = None
        utc_col: List[str] = []
        numeric_rows: List[Tuple[float, ...]] = []

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    if line.startswith("#StartUTC"):
                        text = line.split("=", 1)[1].strip()
                        try:
                            file_start = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f")
                        except ValueError:
                            file_start = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
                    elif line.startswith("#StopUTC"):
                        text = line.split("=", 1)[1].strip()
                        try:
                            file_end = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f")
                        except ValueError:
                            file_end = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
                    continue
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 13:
                    utc_col.append(parts[0])
                    numeric_rows.append((
                        float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9]),
                        float(parts[10]), float(parts[11]), float(parts[12]),
                    ))

        if not numeric_rows:
            return None, file_start, file_end

        arr = np.array(numeric_rows, dtype=np.float64)
        valid = (
            (arr[:, 0] != _INVALID_FLAG)
            & (arr[:, 1] != _INVALID_FLAG)
            & (arr[:, 2] != _INVALID_FLAG)
        )

        utc_valid = [utc_col[i] for i in range(len(utc_col)) if valid[i]]
        times = pd.to_datetime(utc_valid, format=_UTC_FORMAT, utc=True).tz_localize(None)
        arr = arr[valid]

        df = pd.DataFrame({
            "time": times,
            "Bx": arr[:, 0], "By": arr[:, 1], "Bz": arr[:, 2], "Bt": arr[:, 3],
            "pos_x": arr[:, 4], "pos_y": arr[:, 5], "pos_z": arr[:, 6],
        })
        return df, file_start, file_end
    except Exception as exc:
        print(f"读取文件 {file_path} 时出错: {exc}")
        return None, None, None


def read_dat_files(trange, data_dir):
    """
    Read VEX MAG .dat files within a requested time range.

    Parameters
    ----------
    trange : list[str]
        ['YYYY-MM-DD/HH:MM:SS', 'YYYY-MM-DD/HH:MM:SS']
    data_dir : str
        Directory containing BAM_*.dat files.

    Returns
    -------
    tuple[dict | None, list]
        (data_dict with keys time/Bx/By/Bz/Bt/pos_x/pos_y/pos_z, trange_list)
    """
    df, trange_list = read_dat_files_df(trange, data_dir)
    if df is None:
        return None, None

    data_dict = {
        "time": df["time"].values,
        "Bx": df["Bx"].values,
        "By": df["By"].values,
        "Bz": df["Bz"].values,
        "Bt": df["Bt"].values,
        "pos_x": df["pos_x"].values,
        "pos_y": df["pos_y"].values,
        "pos_z": df["pos_z"].values,
    }
    return data_dict, trange_list


def read_dat_files_df(trange, data_dir):
    """
    Read VEX MAG .dat files, returning a DataFrame directly.

    Parameters
    ----------
    trange : list[str]
        ['YYYY-MM-DD/HH:MM:SS', 'YYYY-MM-DD/HH:MM:SS']
    data_dir : str
        Directory containing BAM_*.dat files.

    Returns
    -------
    tuple[pd.DataFrame | None, list]
    """
    file_index = _build_file_index(data_dir)
    all_dataframes: List[pd.DataFrame] = []
    trange_list = []

    for date_str in _date_range_for_trange(trange):
        matching_files = file_index.get(date_str)
        if not matching_files:
            continue

        for file_path in matching_files:
            df, file_start, file_end = _read_single_dat(file_path)
            if file_start and file_end:
                trange_list.append([file_start, file_end])
            if df is not None and not df.empty:
                all_dataframes.append(df)

    if not all_dataframes:
        return None, None

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df, trange_list
