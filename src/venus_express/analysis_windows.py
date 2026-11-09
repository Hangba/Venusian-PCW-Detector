"""Shared event/display window geometry for analysis and review figures."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from venus_express.analysis import DEFAULT_POL_WINDOW_SIZE

EXPORT_DISPLAY_SECONDS = 600
EXPORT_POL_STEP_SECONDS = 10


@dataclass(frozen=True)
class AnalysisWindowGeometry:
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    display_start: pd.Timestamp
    display_end: pd.Timestamp
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp

    @staticmethod
    def _format(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d/%H:%M:%S")

    @property
    def event_trange(self) -> tuple[str, str]:
        return self._format(self.event_start), self._format(self.event_end)

    @property
    def display_trange(self) -> tuple[str, str]:
        return self._format(self.display_start), self._format(self.display_end)

    @property
    def analysis_trange(self) -> tuple[str, str]:
        return self._format(self.analysis_start), self._format(self.analysis_end)


def export_analysis_geometry(
    trange: tuple[str, str] | list[str],
    *,
    min_display_sec: float = EXPORT_DISPLAY_SECONDS,
    analysis_pad_sec: int = DEFAULT_POL_WINDOW_SIZE,
) -> AnalysisWindowGeometry:
    """Return the exact event, widened display, and padded analysis windows."""

    event_start = pd.to_datetime(trange[0], utc=True)
    event_end = pd.to_datetime(trange[1], utc=True)
    if event_end <= event_start:
        raise ValueError("event end must be later than event start")

    duration_sec = float((event_end - event_start).total_seconds())
    if duration_sec < min_display_sec:
        center = event_start + (event_end - event_start) / 2
        half = pd.Timedelta(seconds=float(min_display_sec) / 2.0)
        display_start = center - half
        display_end = center + half
    else:
        display_start = event_start
        display_end = event_end

    pad = pd.Timedelta(seconds=int(analysis_pad_sec))
    return AnalysisWindowGeometry(
        event_start=event_start,
        event_end=event_end,
        display_start=display_start,
        display_end=display_end,
        analysis_start=display_start - pad,
        analysis_end=display_end + pad,
    )
