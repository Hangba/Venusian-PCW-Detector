"""Whole-interval six gates and bounded STFT rescue."""

import time
from .raw import raw_root


def physical(span):
    """Whole check of one extended span, STFT rescue if it fails (2013 evaluate(), unchanged)."""
    from venus_express.event_rescue import _full_pass, _seconds
    from venus_express.production_event_refinement import WholeEventEvaluator
    from venus_express.stft_event_rescue import rescue_stft_scope

    a, b = span
    tick = time.monotonic()
    ev = WholeEventEvaluator([], str(raw_root()))
    whole = ev(a, b)
    qualified, rescue, mode = [], None, "short_or_unavailable"
    if b - a >= 80 and whole is not None and _full_pass(whole):
        qualified = [dict(start_s=a, end_s=b, measurement=whole, path="whole_pass")]
        mode = "whole_pass"
    elif b - a >= 80 and ev.source is not None:
        outputs, meta = rescue_stft_scope(ev.source, ev.measure_loaded, a, b)
        mode = "rescue"
        for e in outputs:
            left, right = _seconds(e["start"]), _seconds(e["end"])
            assert (
                a <= left < right <= b
                and right - left >= 80
                and _full_pass(e["whole_interval_measurement"])
            )
            qualified.append(
                dict(
                    start_s=left,
                    end_s=right,
                    measurement=e["whole_interval_measurement"],
                    path="rescue",
                )
            )
        rescue = (
            {k: meta[k] for k in ("budget", "checked", "passing_runs", "support_runs") if k in meta}
            if isinstance(meta, dict)
            else None
        )
    result = dict(
        start_s=a,
        end_s=b,
        mode=mode,
        whole_measurement=whole,
        qualified=qualified,
        rescue=rescue,
        elapsed_s=time.monotonic() - tick,
    )
    return result
