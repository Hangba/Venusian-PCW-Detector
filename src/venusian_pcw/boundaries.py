"""Reference bow-shock geometry and digitized crossing positions."""

from contextlib import redirect_stdout
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from venus_express.constants import R_VENUS
from venus_express.io.vex_mag import read_dat_files_df
from venus_express.physics.bow_shock import BowShockModel, vso_to_aberrated_vso
from venus_express.repro.raw import raw_root
from venus_express.repro.settings import INPUTS, redact_output

# Published model parameters (x0, L, eccentricity); no measured crossings.
BOW_SHOCK_MODELS = {
    "低活动期": (0.626, 1.559, 1.024),
    "高活动期": (0.638, 1.687, 1.041),
}


def bow_shock_curve(x0, L, eps, theta=np.linspace(-2.4, 2.4, 600)):
    denominator = 1 + eps * np.cos(theta)
    theta, denominator = theta[denominator > 0], denominator[denominator > 0]
    rr = L / denominator
    x, r = x0 + rr * np.cos(theta), rr * np.sin(theta)
    keep = r >= 0
    return x[keep], r[keep]


def positions_by_period(table):
    columns = ["X_AVSO_Rv", "R_AVSO_Rv"]
    if set(table.period) != {"low", "high"}:
        raise ValueError("Shock positions must contain both low and high activity periods")
    if not np.isfinite(table[columns].to_numpy(dtype=float)).all() or table.R_AVSO_Rv.lt(0).any():
        raise ValueError("Shock positions must be finite and have nonnegative cylindrical radius")
    return {period: rows[columns].to_numpy(dtype=float) for period, rows in table.groupby("period")}


def read_shock_positions(path=None):
    """Read low/high activity AVSO crossing positions in Venus radii."""
    if path is None:
        path = INPUTS / "boundaries/shock_positions.csv"
    table = pd.read_csv(
        Path(path).expanduser(),
        usecols=["X_AVSO_Rv", "R_AVSO_Rv", "period"],
        float_precision="round_trip",
    )
    return positions_by_period(table)


def measure_shock_positions(crossings_path):
    """Match each annotated UTC crossing to its nearest valid MAG position."""
    crossings = pd.read_csv(Path(crossings_path).expanduser(), usecols=["utc", "period"])
    crossings["utc"] = pd.to_datetime(crossings.utc, format="mixed", utc=True)
    if crossings.utc.isna().any() or set(crossings.period) != {"low", "high"}:
        raise ValueError("Crossings need UTC timestamps and low/high activity labels")
    if crossings.utc.duplicated().any():
        raise ValueError("Crossing timestamps must be unique")

    tolerance = pd.Timedelta(seconds=2)
    coordinates = ["pos_x", "pos_y", "pos_z"]
    measured = []
    data_dir = str(raw_root())
    for day, group in crossings.groupby(crossings.utc.dt.normalize()):
        trange = [
            value.strftime("%Y-%m-%d/%H:%M:%S")
            for value in (group.utc.min() - tolerance, group.utc.max() + tolerance)
        ]
        messages = io.StringIO()
        with redirect_stdout(messages):
            samples, _ = read_dat_files_df(trange, data_dir)
        if messages.getvalue():
            print(redact_output(messages.getvalue()), end="")
        if samples is None:
            raise ValueError(f"No valid MAG positions for crossings on {day.date()}")
        xyz = samples[coordinates].to_numpy(dtype=float)
        valid = np.isfinite(xyz).all(axis=1) & ~np.isclose(xyz, 99999.999, atol=0.001, rtol=0).any(
            axis=1
        )
        samples = samples.loc[valid, ["time", *coordinates]].copy()
        samples["sample_utc"] = pd.to_datetime(samples.pop("time"), utc=True).astype(
            "datetime64[ns, UTC]"
        )
        group = group.assign(utc=group.utc.astype("datetime64[ns, UTC]"))
        match = pd.merge_asof(
            group.sort_values("utc"),
            samples.sort_values("sample_utc"),
            left_on="utc",
            right_on="sample_utc",
            direction="nearest",
            tolerance=tolerance,
        )
        if match.sample_utc.isna().any():
            missing = match.loc[match.sample_utc.isna(), "utc"].iloc[0]
            raise ValueError(f"No valid MAG position within 2 s of crossing {missing.isoformat()}")
        avso = vso_to_aberrated_vso(match[coordinates].to_numpy() / R_VENUS)
        measured.append(
            match[["utc", "sample_utc", "period"]].assign(
                X_AVSO_Rv=avso[:, 0],
                R_AVSO_Rv=np.hypot(avso[:, 1], avso[:, 2]),
            )
        )
    return pd.concat(measured, ignore_index=True).sort_values("utc")


def fit_shock_models(points):
    """Fit the implicit conic residual separately in each activity period."""
    models = {}
    for period, label in (("low", "低活动期"), ("high", "高活动期")):
        x, r = points[period].T
        if len(x) < 3:
            raise ValueError(f"At least three crossing positions are required for {period}")

        def residual(params):
            x0, L, eps = params
            model = BowShockModel(period, L, eps, focus_x_rv=x0)
            return model.signed_implicit_distance_rv(x, r, 0)

        fit = least_squares(
            residual,
            BOW_SHOCK_MODELS[label],
            bounds=([-np.inf, 1e-6, 0], [np.inf, np.inf, np.inf]),
        )
        if not fit.success or np.linalg.matrix_rank(fit.jac) < 3:
            raise ValueError(f"Crossing positions do not constrain a conic for {period}")
        models[label] = tuple(fit.x)
    return models
