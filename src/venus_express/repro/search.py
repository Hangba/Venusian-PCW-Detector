"""Recompute coarse parents, localize and rescue on the original observation days."""

import contextlib
import io
from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from venus_express.event_rescue import _seconds
from venus_express.icw_inference import sliding_window_inference

from .coarse import group_arrays, profiles
from .localization import ThresholdScanner
from .model import PROTOCOL, load_model
from .physical import physical
from .raw import cache, observation_days, raw_root
from .settings import BUILD, read, write


def mission_day(day, source, model, device):
    features = source.build_or_load_daily_features(day)
    start = pd.Timestamp(day, tz="UTC").timestamp()
    times = features["times_s"]
    mask = (times >= start) & (times <= start + 86399)
    features = {
        key: value if key == "freq_pol" else value[..., mask] for key, value in features.items()
    }
    times = features["times_s"]
    indices = np.arange(0, len(times) - 79, 20, dtype=np.int64)
    logits = []
    freq = torch.as_tensor(features["freq_pol"], device=device)
    with torch.inference_mode():
        for j in range(0, len(indices), 512):
            x, fc = profiles(features, indices[j : j + 512])
            z = model(
                torch.as_tensor(x, device=device),
                fcp_curve=torch.as_tensor(fc, device=device),
                freq_pol=freq,
            )
            logits.append(z.cpu().numpy())
    starts = np.floor(times[indices]).astype(np.int64)
    ends = np.floor(np.maximum(times[indices + 79], times[indices] + 80)).astype(np.int64)
    return starts, ends, np.concatenate(logits) if logits else np.empty(0, np.float32)


class CaptureLogits(torch.nn.Module):
    """Capture uncalibrated float32 logits from the 2013 window iterator."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.batches = []
        self.probability_calibration = None

    def forward(self, x, **kwargs):
        z = self.model(x.mean(-1) if x.ndim == 4 else x, **kwargs)
        self.batches.append(z.detach().cpu().numpy())
        return z


def holdout_day(day, source, model, device):
    model.batches.clear()
    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        windows, _, _ = sliding_window_inference(
            model,
            [day + "/00:00:00", day + "/23:59:59"],
            data_dir=str(raw_root()),
            feature_cache=source,
            device=torch.device(device),
            window_size=80,
            step_size=20,
            num_freqs=64,
            threshold=1.1,
            inference_batch_size=4096,
            freq_scale_factor=1.0,
            vectorized_feature_build=False,
            fcp_band_gate=False,
            ellip_gate=False,
            apply_pf_search_gate=False,
            compute_window_fft_diagnostics=False,
        )
    z = (
        np.concatenate(model.batches).astype(np.float32)
        if model.batches
        else np.empty(0, np.float32)
    )
    assert z.shape == (len(windows),)
    return (
        np.array([_seconds(w["start"]) for w in windows], dtype=np.int64),
        np.array([_seconds(w["end"]) for w in windows], dtype=np.int64),
        z,
    )


def deduplicate(rows):
    ranked = sorted(
        rows, key=lambda r: (-r["final_score"], r["start_s"], r["end_s"], r["parent_id"])
    )
    kept = []
    for row in ranked:
        a, b = row["start_s"], row["end_s"]
        if any(
            max(0, min(b, q["end_s"]) - max(a, q["start_s"]))
            >= 0.5 * (max(b, q["end_s"]) - min(a, q["start_s"]))
            for q in kept
        ):
            continue
        kept.append(row)
    return kept


def coarse_parents(scope, days, model, device, spec, folder):
    source = cache()
    capture = CaptureLogits(model) if scope == "holdout" else model
    predict = holdout_day if scope == "holdout" else mission_day
    cuts = [spec["threshold"]] if scope == "holdout" else PROTOCOL["mission_cuts"]
    parts = {cut: [] for cut in cuts}
    folder.mkdir(parents=True, exist_ok=True)
    for n, day in enumerate(days, 1):
        file = folder / f"{day}.npz"
        if file.exists():
            with np.load(file, allow_pickle=False) as data:
                starts, ends, logits = (data[k] for k in ("starts", "ends", "logits"))
        else:
            try:
                starts, ends, logits = predict(day, source, capture, device)
            except (ValueError, IndexError, KeyError):
                from venus_express.io.vex_mag import read_dat_files_df

                frame, _ = read_dat_files_df([f"{day}/00:00:00", f"{day}/23:59:59"], str(raw_root()))
                if frame is not None and len(frame) >= source.pol_window_size:
                    raise
                print(f"{day}: insufficient records for polarization analysis", flush=True)
                starts = ends = np.empty(0, dtype=np.int64)
                logits = np.empty(0, dtype=np.float32)
            if not np.isfinite(logits).all():
                raise ValueError(f"Nonfinite scores on {day}")
            tmp = file.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, starts=starts, ends=ends, logits=logits)
            tmp.replace(file)
        for cut in cuts:
            score = logits if scope == "holdout" else expit(logits.astype(float))
            cutoff = spec["threshold_logit"] if scope == "holdout" else cut
            take = score >= cutoff
            parts[cut].append((starts[take], ends[take], score[take]))
        if n % 25 == 0 or n == len(days):
            print(f"{scope} coarse {n}/{len(days)} dates", flush=True)
    parents = []
    for cut, chunks in parts.items():
        if not chunks:
            continue
        starts, ends, score = (np.concatenate(col) for col in zip(*chunks))
        order = np.argsort(starts, kind="stable")
        for group in group_arrays(starts[order], ends[order], score[order], -np.inf):
            parents.append(dict(group, threshold=cut, parent_id=len(parents) + 1))
    return parents


def rescore(scanner, item):
    centers = range(item["start_s"] + 40, item["end_s"] - 40 + 1)
    scanner.score(centers)
    valid = [(scanner.values[c], c) for c in centers if scanner.values[c] is not None]
    return max(valid, key=lambda x: (x[0], -x[1]), default=None)


def search(scope, device, workers, days=None, resume=False):
    official_days = observation_days(scope)
    days = official_days if days is None else sorted(set(days))
    if not days or not set(days) <= set(official_days):
        raise ValueError("Requested dates are not available in this MAG scope")
    spec = (
        PROTOCOL["holdout2013"]
        if scope == "holdout"
        else next(m for m in PROTOCOL["models"] if m["stage"] == "MStandard" and m["seed"] == 2026)
    )
    torch.set_num_threads(4)
    model = load_model(spec, device)
    if scope == "holdout":
        from venusian_pcw.window_inputs import prepared_path
        from .training import arrays, predict

        validation = arrays(prepared_path("holdout2013", "validation"))
        positive = np.sort(predict(model, validation, device)[validation["y"] > 0])[::-1]
        rank = int(np.ceil(spec["validation_recall"] * len(positive)))
        if not rank:
            raise ValueError("Holdout validation needs positive windows")
        cutoff = float(positive[rank - 1])
        spec = dict(spec, threshold_logit=cutoff, threshold=float(expit(cutoff)))
    dest = BUILD / "raw" / scope
    context = dict(
        days=days, checkpoint_sha256=spec["checkpoint_sha256"], raw=str(raw_root()), device=device,
        threshold=spec.get("threshold"), run_schema=2
    )
    if resume:
        previous = read(dest / "context.json")
        if previous["inputs"] != context:
            raise ValueError("Resume requires the same dates, checkpoint, device and data source")
        run_id = previous["run_id"]
    else:
        run_id = uuid4().hex
        write(dest / "context.json", {"inputs": context, "run_id": run_id})
    (dest / "complete.json").unlink(missing_ok=True)
    parents = coarse_parents(scope, days, model, device, spec, dest / "coarse" / run_id)
    write(dest / "parents.json", parents)
    scanner = ThresholdScanner(
        model, {int(pd.Timestamp(d, tz="UTC").timestamp()) // 86400 for d in days}
    )
    records, spans = [], set()
    for n, parent in enumerate(parents, 1):
        scanner.threshold = parent["threshold"]
        loc = scanner.parent(parent["start_s"], parent["end_s"])
        record = dict(
            threshold=scanner.threshold,
            parent_id=parent["parent_id"],
            parent_start_s=parent["start_s"],
            parent_end_s=parent["end_s"],
            status=loc["status"],
            span_start_s=loc.get("start_s"),
            span_end_s=loc.get("end_s"),
            mode=None,
            qualified=[],
        )
        records.append(record)
        if loc["status"] == "resolved":
            spans.add((loc["start_s"], loc["end_s"]))
        if n % 100 == 0 or n == len(parents):
            print(f"{scope} localization {n}/{len(parents)} parents", flush=True)
    ordered_spans = sorted(spans)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            measurements = dict(zip(ordered_spans, pool.map(physical, ordered_spans)))
    else:
        measurements = {span: physical(span) for span in ordered_spans}

    representatives = []
    for record in records:
        if record["status"] != "resolved":
            continue
        result = measurements[record["span_start_s"], record["span_end_s"]]
        record["mode"] = result["mode"]
        for q in result["qualified"]:
            best = rescore(scanner, q)
            if best is None:
                continue
            value, center = best
            m = q["measurement"]
            record["qualified"].append(
                dict(
                    start_s=q["start_s"],
                    end_s=q["end_s"],
                    path=q["path"],
                    final_score=value,
                    score_center=center,
                    **m["metrics"],
                    ellip=m["ellip_at_fmax"],
                    wna=m["wna_at_fmax"],
                )
            )
        if scope == "holdout" and record["qualified"]:
            q = max(
                record["qualified"], key=lambda r: (r["final_score"], -r["start_s"], -r["end_s"])
            )
            representatives.append(dict(q, parent_id=record["parent_id"]))
    write(dest / "parent_outcomes.json", records)
    if scope == "holdout":
        write(dest / "ranked.json", deduplicate(representatives))
    write(dest / "complete.json", {"inputs": context, "run_id": run_id, "parents": len(parents)})
    print(f"{scope}: {len(days)} dates, {len(parents)} parents, {len(spans)} localized spans")
