"""Load, filter, and merge ICW sample time ranges."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SampleEvent:
    start: pd.Timestamp
    end: pd.Timestamp
    score: float = 1.0

    @property
    def midpoint(self) -> pd.Timestamp:
        return self.start + (self.end - self.start) / 2

    @property
    def duration_s(self) -> float:
        return float((self.end - self.start).total_seconds())


def _timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return pd.Timestamp(ts)


def load_events(
    path: str | Path,
    *,
    min_score: float = 0.0,
    enabled_only: bool = True,
) -> list[SampleEvent]:
    """Read candidates.json or a CSV containing start/end time columns."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else []
    else:
        rows = pd.read_csv(path).to_dict("records")

    events: list[SampleEvent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (
            enabled_only
            and "enable" in row
            and str(row["enable"]).strip().lower() in {"0", "false", "no", "off"}
        ):
            continue
        trange = row.get("trange")
        start_value = (
            trange[0]
            if isinstance(trange, list) and len(trange) == 2
            else row.get("start", row.get("start_ut", row.get("StartTimeUT")))
        )
        end_value = (
            trange[1]
            if isinstance(trange, list) and len(trange) == 2
            else row.get("end", row.get("end_ut", row.get("EndTimeUT")))
        )
        try:
            score = float(row.get("score", 1.0))
            start, end = _timestamp(start_value), _timestamp(end_value)
        except (TypeError, ValueError):
            continue
        if score >= min_score and end > start:
            events.append(SampleEvent(start=start, end=end, score=score))
    return sorted(events, key=lambda event: event.start)


def merge_events(
    events: list[SampleEvent], gap: pd.Timedelta = pd.Timedelta(minutes=10)
) -> list[SampleEvent]:
    if not events:
        return []
    merged: list[SampleEvent] = []
    start, end, score = events[0].start, events[0].end, events[0].score
    for event in sorted(events[1:], key=lambda item: item.start):
        if event.start - end <= gap:
            end = max(end, event.end)
            score = max(score, event.score)
        else:
            merged.append(SampleEvent(start, end, score))
            start, end, score = event.start, event.end, event.score
    merged.append(SampleEvent(start, end, score))
    return merged


def event_table(events: list[SampleEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": [event.start for event in events],
            "end": [event.end for event in events],
            "midpoint": [event.midpoint for event in events],
            "duration_s": [event.duration_s for event in events],
            "score": [event.score for event in events],
        }
    )
