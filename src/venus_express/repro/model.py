"""Load the exact architecture and selected state dictionaries used in the paper."""

import torch

from venus_express.residual_model import DualBranch
from .settings import ROOT, INPUTS, read, checked_file

PROTOCOL = read(INPUTS / "protocol.json")


def build_model():
    model = DualBranch(**PROTOCOL["model_kwargs"])
    assert sum(p.numel() for p in model.parameters()) == 24084
    return model


def load_model(spec, device):
    path = checked_file(ROOT / spec["checkpoint"], spec["checkpoint_sha256"])
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.probability_calibration = None
    return model.to(device).eval()
