"""Shared 1 s threshold crossing and context rules for mission and 2013."""

import math
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from venus_express.analysis_windows import export_analysis_geometry
from venus_express.production_event_refinement import CachedWindowProvider
from .raw import cache


def allowed_span(a, b, days):
    return all(day in days for day in range(math.floor(a / 86400), math.floor(b / 86400) + 1))


def utc(second):
    return pd.Timestamp(second, unit="s", tz="UTC").isoformat()


class _BoundaryScanner:
    def __init__(self, model, days):
        self.model = model
        self.provider = CachedWindowProvider(cache())
        self.days = days
        self.values = {}
        self.reasons = {}
        self.forwards = 0

    def score(self, centers):
        todo = sorted(set(int(t) for t in centers) - self.values.keys())
        pending = []

        def flush():
            if not pending:
                return
            x = torch.as_tensor(
                np.stack([r[1][0] for r in pending]), device=next(self.model.parameters()).device
            )
            fc = torch.as_tensor(
                np.stack([r[1][1] for r in pending]), device=next(self.model.parameters()).device
            )
            freq = torch.as_tensor(
                self.provider.freq_pol, device=next(self.model.parameters()).device
            )
            with torch.inference_mode():
                z = self.model(x, fcp_curve=fc, freq_pol=freq).detach().cpu().numpy().reshape(-1)
            for (t, _), v in zip(pending, z):
                self.values[t] = float(expit(float(v))) if np.isfinite(v) else None
                if not np.isfinite(v):
                    self.reasons[t] = "nonfinite_score"
            self.forwards += 1
            pending.clear()

        for t in todo:
            if not allowed_span(t - 40 - 240, t + 40 + 240, self.days):
                self.values[t] = None
                self.reasons[t] = "eligible_date_boundary"
                continue
            data = self.provider(t - 40, t + 40)
            if data is None:
                self.values[t] = None
                self.reasons[t] = getattr(self.provider, "last_interpolation", {}).get(
                    "unavailable_reason", "unavailable_features"
                )
                continue
            pending.append((t, data))
            if len(pending) == 256:
                flush()
        flush()

    threshold = 0.7

    def walk(self, t, direction):
        high = t
        while True:
            candidates = [high + direction * i for i in range(1, 257)]
            self.score(candidates)
            for c in candidates:
                z = self.values[c]
                if z is None:
                    return dict(
                        status="unresolved", reason=self.reasons[c], last_high=high, stop_center=c
                    )
                if z < self.threshold:
                    hz = self.values[high]
                    cross = high + (self.threshold - hz) / (z - hz) * (c - high)
                    return dict(
                        status="crossing", crossing_s=float(cross), high_center=high, low_center=c
                    )
                high = c

    def parent(self, a, b):
        self.score(range(a, b + 1))
        high = [
            t
            for t in range(a, b + 1)
            if self.values[t] is not None and self.values[t] >= self.threshold
        ]
        if not high:
            return dict(status="no_high_range", start_s=None, end_s=None, left=None, right=None)
        left = self.walk(min(high), -1)
        right = self.walk(max(high), 1)
        if left["status"] != "crossing" or right["status"] != "crossing":
            return dict(
                status="unresolved_crossing", start_s=None, end_s=None, left=left, right=right
            )
        start = math.floor(left["crossing_s"] - 20)
        end = math.ceil(right["crossing_s"] + 20)
        geo = export_analysis_geometry([utc(start), utc(end)])
        if not allowed_span(
            geo.analysis_start.timestamp(), geo.analysis_end.timestamp(), self.days
        ):
            return dict(
                status="context_outside_eligible_dates",
                start_s=None,
                end_s=None,
                left=left,
                right=right,
            )
        return dict(status="resolved", start_s=start, end_s=end, left=left, right=right)


class ThresholdScanner(_BoundaryScanner):
    def parent(self, a, b):
        result = super().parent(a, b)
        if result["status"] != "resolved":
            return result
        start = math.floor(result["left"]["crossing_s"] - 40)
        end = math.ceil(result["right"]["crossing_s"] + 40)
        assert end - start >= 80
        geo = export_analysis_geometry([utc(start), utc(end)])
        if not allowed_span(
            geo.analysis_start.timestamp(), geo.analysis_end.timestamp(), self.days
        ):
            return dict(result, status="context_outside_eligible_dates", start_s=None, end_s=None)
        return dict(result, start_s=start, end_s=end, boundary_padding_s=40)
