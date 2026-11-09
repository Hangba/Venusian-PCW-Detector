"""Recompute validation operating points and the paper's fixed-window table."""

import numpy as np
import pandas as pd
from scipy.special import expit

from venus_express.repro.model import PROTOCOL, load_model
from venus_express.repro.settings import BUILD, ROOT, INPUTS, read
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


def compute(rescore=False, device="cpu"):
    output = BUILD / "windows"
    output.mkdir(parents=True, exist_ok=True)
    points, matched = [], []
    if rescore:
        import torch

        torch.set_num_threads(4)
        test_features = arrays(read(INPUTS / "windows/external_test.json"))
    for spec in PROTOCOL["models"]:
        stage, seed = spec["stage"], spec["seed"]
        stem = f"{stage}.{seed}"
        val, test = [
            pd.read_csv(INPUTS / f"windows/{stem}.{part}.csv", float_precision="round_trip")
            for part in ["validation", "test"]
        ]
        assert test.source.value_counts().to_dict() == {"P": 61, "bg": 720, "F": 90}
        if rescore:
            model = load_model(spec, device)
            validation_features = arrays(PROTOCOL["training"][stage]["validation"])
            for frame, features, target in [
                (val, validation_features, "cpu"),
                (test, test_features, device),
            ]:
                np.testing.assert_array_equal(frame.source, features["source"])
                if target != device:
                    model.to(target)
                actual = predict(model, features, target)
                np.testing.assert_allclose(actual, frame.logit, rtol=3e-4, atol=3e-4)
                frame["logit"] = actual.astype(float)
                model.to(device)
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
    keys = ["stage", "seed", "target_recall"]
    expected = pd.read_csv(ROOT / "reference/working_points.csv", float_precision="round_trip")

    def ordered(frame):
        return frame[expected.columns].sort_values(keys).reset_index(drop=True)

    if not rescore:
        pd.testing.assert_frame_equal(
            ordered(result), ordered(expected), check_dtype=False, atol=1e-12, rtol=1e-12
        )
    result["protocol"] = "own_development_frozen"
    result["score_threshold"] = expit(result.threshold)
    result.to_csv(output / "working_points.csv", index=False)
    focus = result[np.isclose(result.target_recall, 0.8)]
    summary = focus.groupby("stage")[
        ["P_recall", "background_FPR", "F_rejection", "AP_PB_1_8", "AP_PF_1_1"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.to_csv(output / "model_performance.csv")
    matched_frame = pd.DataFrame(matched)
    expected_matched = pd.read_csv(
        ROOT / "reference/matched_points.csv", float_precision="round_trip"
    )
    if not rescore:
        cols = expected_matched.columns
        pd.testing.assert_frame_equal(
            matched_frame[cols].sort_values(["stage", "seed"]).reset_index(drop=True),
            expected_matched.sort_values(["stage", "seed"]).reset_index(drop=True),
            check_dtype=False,
            atol=1e-12,
            rtol=1e-12,
        )
    matched_frame.to_csv(output / "matched_49_of_61.csv", index=False)
    return dict(
        validation_operating_point=summary.reset_index().to_dict("records"),
        equal_test_recall=matched_frame.groupby("stage")[["bg_accepted", "F_rejection"]]
        .mean()
        .reset_index()
        .to_dict("records"),
    )
