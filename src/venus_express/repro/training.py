"""Replay the four paper training stages and the separate 2013 holdout fit."""

import math
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

from venus_express.icw_training import weighted_bce_loss

from .model import build_model
from .settings import BUILD, write

MASS = {
    "MSeed": {"P": 0.4, "bg": 0.6},
    "MExtend1": {"P": 0.4, "bg": 0.45, "other_neg": 0.15},
    "MExtend2": {"P": 0.4, "bg": 0.45, "other_neg": 0.15},
}


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        result = dict(z)
    if "class_source" not in result:
        result["class_source"] = result["source"]
    np.testing.assert_array_equal(result["class_source"], result["source"])
    return result


def subset(data, mask):
    return {key: value if key == "freq_pol" else value[mask] for key, value in data.items()}


def schedule(n, slots, seed):
    if not 0 < n <= slots:
        raise ValueError(f"{n} rows do not fit in {slots} slots")
    rng = np.random.default_rng(seed)
    indices = np.r_[np.arange(n), rng.integers(0, n, size=slots - n)]
    counts = np.bincount(indices, minlength=n)
    return rng.permutation(indices), counts


def source_weights(data, mass):
    weights = data["w"].astype(np.float64)
    classes = data["class_source"]
    assert set(classes) == set(mass)
    for label, fraction in mass.items():
        mask = classes == label
        weights[mask] *= fraction / weights[mask].sum()
    return (weights * (len(weights) / weights.sum())).astype(np.float32)


def logits(model, data, indices, freq, device):
    x = torch.as_tensor(data["X_mean"][indices].copy(), device=device)
    fc = torch.as_tensor(data["fcp_median"][indices].copy(), device=device).reshape(-1, 1)
    return model(x, fcp_curve=fc, freq_pol=freq)


def predict(model, data, device):
    model.eval()
    freq = torch.as_tensor(data["freq_pol"].copy(), device=device)
    with torch.no_grad():
        return np.concatenate(
            [
                logits(model, data, slice(lo, lo + 512), freq, device).cpu().numpy()
                for lo in range(0, len(data["y"]), 512)
            ]
        )


def weighted_ap(y, score, weights):
    order = np.argsort(-score, kind="stable")
    y, score, weights = y[order], score[order], weights[order]
    tp, fp = np.cumsum(weights * y), np.cumsum(weights * (1 - y))
    last = np.r_[np.flatnonzero(np.diff(score) != 0), len(score) - 1]
    recall = tp[last] / tp[-1]
    precision = tp[last] / (tp[last] + fp[last])
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def validation_metrics(data, z):
    z = z.astype(np.float64)
    y = data["y"].astype(float)
    classes = data["class_source"]
    _, inverse, counts = np.unique(data["group"], return_inverse=True, return_counts=True)
    parent = (1.0 / counts[inverse]).astype(np.float32).astype(float)
    weights = parent.copy()
    manual_prior = 0.1 if "other_neg" in classes else 0.0
    for label, prior in {
        "P": 0.1,
        "other_neg": manual_prior,
        "bg": 1 - (0.1 + manual_prior),
    }.items():
        if prior:
            mask = classes == label
            weights[mask] *= prior / weights[mask].sum()
    pb = np.isin(classes, ["P", "bg"])
    pb_weight = parent[pb]
    for label in (0, 1):
        mask = y[pb] == label
        pb_weight[mask] /= pb_weight[mask].sum()
    return dict(
        AP=weighted_ap(y[pb], 1 / (1 + np.exp(-z[pb])), pb_weight),
        BCE=float(np.sum(weights * (np.logaddexp(0, z) - y * z))),
    )


def fit(stage, seed, device, prepared):
    if stage == "holdout2013" and seed != 2026:
        raise ValueError("The 2013 holdout fit uses seed 2026")
    train, validation = [arrays(prepared[role]) for role in ("train", "validation")]
    assert "F" not in validation["class_source"]
    is_standard = stage in {"MStandard", "holdout2013"}
    is_f = train["class_source"] == "F"
    pb = subset(train, ~is_f)
    f = subset(train, is_f) if is_standard else None
    weight = source_weights(pb, MASS["MExtend2" if is_standard else stage])
    f_weight = f["w"].astype(np.float64) if is_standard else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = build_model().to(device)
    output = BUILD / "training_from_bam" / stage / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    freq = torch.as_tensor(pb["freq_pol"], device=device)
    history = []
    for epoch in range(1, 101):
        rng_seed = seed * 1000 + epoch
        order, counts = schedule(len(pb["y"]), 52 * 256, rng_seed)
        if is_standard:
            f_order, f_counts = schedule(len(f["y"]), 52 * 16, rng_seed + 40_000_000)
        model.train()
        for step in range(52):
            idx = order[step * 256 : (step + 1) * 256]
            z = logits(model, pb, idx, freq, device)
            target = torch.as_tensor(pb["y"][idx], dtype=torch.float32, device=device)
            adjusted = torch.as_tensor(
                weight[idx] / counts[idx], dtype=torch.float32, device=device
            )
            loss = weighted_bce_loss(z, target, adjusted)
            if is_standard:
                fi = f_order[step * 16 : (step + 1) * 16]
                fz = logits(model, f, fi, freq, device)
                adjusted = torch.as_tensor(
                    f_weight[fi] / f_counts[fi], dtype=torch.float32, device=device
                )
                per_window = F.binary_cross_entropy_with_logits(
                    fz, torch.zeros_like(fz), reduction="none"
                )
                f_loss = (per_window * adjusted).sum() * (52 / float(f_weight.sum()))
                loss = loss + (1 / 3) * f_loss
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at epoch {epoch}, step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        metrics = validation_metrics(validation, predict(model, validation, device))
        history.append(dict(epoch=epoch, **metrics))
        torch.save(model.state_dict(), output / f"epoch{epoch:03d}.pt")
        write(output / "history.json", history)
        if epoch == 1 or epoch % 10 == 0:
            print(stage, seed, epoch, metrics, flush=True)
    maximum = max(r["AP"] for r in history)
    chosen = min(
        (r for r in history if r["AP"] >= maximum - 0.02),
        key=lambda r: (r["BCE"], -r["AP"], r["epoch"]),
    )
    selected = output / "selected.pt"
    shutil.copyfile(output / f"epoch{chosen['epoch']:03d}.pt", selected)
    model = build_model().cpu()
    model.load_state_dict(torch.load(selected, weights_only=True, map_location="cpu"))
    scores = predict(model, validation, "cpu")
    positive = np.sort(scores[validation["y"] > 0])[::-1]
    cuts = []
    for recall in [0.7, 0.75, 0.8, 0.85, 0.9]:
        cut = float(positive[math.ceil(recall * len(positive)) - 1])
        cuts.append(dict(recall=recall, logit=cut, score=float(expit(cut))))
    write(output / "selection.json", dict(chosen=chosen, validation_cuts=cuts))
    return chosen
