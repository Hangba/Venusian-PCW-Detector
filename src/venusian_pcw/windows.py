"""Recompute validation operating points and the paper's fixed-window table."""

import numpy as np
import pandas as pd
from scipy.special import expit

from venus_express.repro.model import PROTOCOL, load_model
from venus_express.repro.settings import BUILD
from venus_express.repro.training import arrays, predict, weighted_ap


def average_precision(test, negative, negative_mass):
    sample = test[test.source.isin(["P", negative])]
    label = sample.source.eq("P").to_numpy()
    weights = np.where(label, 1 / label.sum(), negative_mass / (~label).sum())
    return weighted_ap(label.astype(float), sample.logit.to_numpy(), weights)


def measures(test, threshold):
    accepted = test.logit >= threshold
    return dict(
        P_recall=accepted[test.source == "P"].mean(),
        background_FPR=accepted[test.source == "bg"].mean(),
        F_rejection=1 - accepted[test.source == "F"].mean(),
    )


def compute(device="cpu"):
    """Score BAM-rebuilt validation and test windows using the released weights."""
    import torch
    from .window_inputs import prepared_path

    torch.set_num_threads(4)
    output = BUILD / "windows_from_bam"
    output.mkdir(parents=True, exist_ok=True)
    points, matched = [], []
    test_features = arrays(prepared_path(None, None))
    validation = {}
    for spec in PROTOCOL["models"]:
        stage, seed = spec["stage"], spec["seed"]
        stem = f"{stage}.{seed}"
        model = load_model(spec, device)
        if stage not in validation:
            validation[stage] = arrays(prepared_path(stage, "validation"))
        features = validation[stage]
        val = pd.DataFrame(dict(source=features["source"], logit=predict(model, features, device).astype(float)))
        test = pd.DataFrame(dict(source=test_features["source"], logit=predict(model, test_features, device).astype(float)))
        ap = dict(
            AP_PB_1_8=average_precision(test, "bg", 8.0),
            AP_PF_1_1=average_precision(test, "F", 1.0),
        )
        positive = np.sort(val.loc[val.source.eq("P"), "logit"])[::-1]
        for recall in [0.7, 0.75, 0.8, 0.85, 0.9]:
            cutoff = float(positive[int(np.ceil(recall * len(positive))) - 1])
            points.append(
                dict(
                    stage=stage,
                    seed=seed,
                    target_recall=recall,
                    threshold=cutoff,
                    **measures(test, cutoff),
                    **ap,
                )
            )
        test_positive = np.sort(test.loc[test.source.eq("P"), "logit"])[::-1]
        test_cut = float(test_positive[48])
        matched.append(
            dict(
                stage=stage,
                seed=seed,
                threshold=test_cut,
                score_threshold=float(expit(test_cut)),
                bg_accepted=int(((test.logit >= test_cut) & test.source.eq("bg")).sum()),
                **measures(test, test_cut),
                **ap,
            )
        )
        test.to_csv(output / f"{stem}.test.csv", index=False)
    result = pd.DataFrame(points)
    result["protocol"] = "recomputed_validation"
    result["score_threshold"] = expit(result.threshold)
    result.to_csv(output / "working_points.csv", index=False)
    focus = result[np.isclose(result.target_recall, 0.8)]
    summary = focus.groupby("stage")[
        ["P_recall", "background_FPR", "F_rejection", "AP_PB_1_8", "AP_PF_1_1"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.to_csv(output / "model_performance.csv")
    matched_frame = pd.DataFrame(matched)
    matched_frame.to_csv(output / "matched_49_of_61.csv", index=False)
    return dict(
        validation_operating_point=summary.reset_index().to_dict("records"),
        equal_test_recall=matched_frame.groupby("stage")[["bg_accepted", "F_rejection"]]
        .mean()
        .reset_index()
        .to_dict("records"),
    )
