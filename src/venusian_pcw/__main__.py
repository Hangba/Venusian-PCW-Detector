"""Detect and analyze proton cyclotron waves in Venus Express MAG observations."""

import argparse
import os

from venus_express.repro.settings import BUILD, display_path

os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(prog="venusian-pcw", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify checksums of the frozen inputs")
    replay = sub.add_parser("replay", help="Recompute numerical benchmarks for the plotting notebook")
    replay.add_argument(
        "--raw-search",
        action="store_true",
        help="Use the complete new mission and holdout searches",
    )
    replay.add_argument(
        "--raw-density",
        action="store_true",
        help="Recompute the 78 density PSDs from BAM",
    )
    replay.add_argument(
        "--rescore-windows",
        action="store_true",
        help="Score external frozen window features with the selected weights",
    )
    replay.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    cache = sub.add_parser("cache", help="Build reusable daily MAG features from BAM")
    cache_source = cache.add_mutually_exclusive_group()
    cache_source.add_argument("--events")
    cache_source.add_argument("--scope", choices=["mission", "holdout"])
    cache.add_argument("--day", action="append")
    cache.add_argument("--workers", type=int, default=2)
    catalogue = sub.add_parser(
        "catalogue", help="Rebuild event physics and fits from BAM plus event intervals"
    )
    catalogue.add_argument("--events")
    catalogue.add_argument("--day", action="append")
    catalogue.add_argument("--workers", type=int, default=2)
    search = sub.add_parser("search", help="Run the current detector on 1 Hz VSO MAG")
    search.add_argument("scope", choices=["mission", "holdout"])
    search.add_argument("--day", action="append")
    search.add_argument("--resume", action="store_true")
    search.add_argument("--workers", type=int, default=1)
    search.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    exposure = sub.add_parser(
        "exposure", help="Rebuild the full AVSO exposure grid from BAM"
    )
    exposure.add_argument("--workers", type=int, default=4)
    evaluate = sub.add_parser(
        "evaluate-windows", help="Rescore rebuilt windows using the paper checkpoints"
    )
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    prepare = sub.add_parser(
        "prepare-windows", help="Build frozen-membership window features from BAM"
    )
    prepare.add_argument(
        "stage",
        choices=[
            "MSeed",
            "MExtend1",
            "MExtend2",
            "MStandard",
            "holdout2013",
            "fixed_test",
        ],
    )
    prepare.add_argument(
        "--role", choices=["train", "validation", "both"], default="both"
    )
    prepare.add_argument("--day", help="Write a separate one-date diagnostic")
    train = sub.add_parser(
        "train", help="Fit a stage/seed on the frozen training arrays"
    )
    train.add_argument(
        "stage", choices=["MSeed", "MExtend1", "MExtend2", "MStandard", "holdout2013"]
    )
    train.add_argument(
        "--from-raw", action="store_true", help="Use the locally rebuilt feature arrays"
    )
    train.add_argument("--seed", type=int, choices=[2026, 2027, 2028], default=2026)
    train.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    audit = sub.add_parser(
        "psd-audit", help="Recompute the full-frequency PSD audit from BAM"
    )
    audit.add_argument("--workers", type=int, default=4)
    audit.add_argument("--day", action="append")
    audit.add_argument("--events")
    ions = sub.add_parser(
        "ima-moments",
        help="Integrate an externally normalized IMA count and calibration bundle",
    )
    ions.add_argument(
        "bundle", help="Path to the normalized NPZ described in docs/data.md"
    )
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")

    if args.command == "verify":
        from .statistics import verify_inputs

        verify_inputs()
    elif args.command == "replay":
        from .statistics import compute

        compute(args.raw_search, args.raw_density, args.rescore_windows, args.device)
    elif args.command == "cache":
        from .catalogue import precompute
        from venus_express.repro.settings import INPUTS

        if args.scope:
            import pandas as pd
            from venus_express.repro.model import PROTOCOL

            valid = PROTOCOL[f"{args.scope}_days"]
            if args.scope == "mission":
                exposure = pd.read_csv(INPUTS / "mission/exposure_days.csv")
                valid = exposure.loc[exposure.scorable, "day"].tolist()
            if args.day and not set(args.day) <= set(valid):
                parser.error("--day is outside the selected scorable scope")
            precompute(None, args.day or valid, args.workers)
        else:
            precompute(
                args.events or (None if args.day else INPUTS / "manual/event_list.csv"),
                args.day,
                args.workers,
            )
    elif args.command == "catalogue":
        from .catalogue import compute

        compute(args.events, args.day, args.workers)
    elif args.command == "search":
        from venus_express.repro.search import search as run

        run(args.scope, args.device, args.workers, args.day, args.resume)
    elif args.command == "exposure":
        from venus_express.repro.exposure import compute

        compute(args.workers)
    elif args.command == "evaluate-windows":
        from .windows import compute
        from venus_express.repro.settings import write

        write(
            BUILD / "windows_from_bam" / "summary.json",
            compute(device=args.device, raw_windows=True),
        )
    elif args.command == "prepare-windows":
        from .window_inputs import materialize

        if args.stage == "fixed_test":
            print(display_path(materialize(None, None, args.day)))
        else:
            for role in ["train", "validation"] if args.role == "both" else [args.role]:
                print(display_path(materialize(args.stage, role, args.day)))
    elif args.command == "train":
        from venus_express.repro.training import fit
        from .window_inputs import prepared_path

        prepared = (
            {role: prepared_path(args.stage, role) for role in ("train", "validation")}
            if args.from_raw
            else None
        )
        fit(args.stage, args.seed, args.device, prepared)
    elif args.command == "psd-audit":
        from .spectral import compute

        compute(args.workers, args.day, args.events)
    elif args.command == "ima-moments":
        from .ions import compute

        compute(args.bundle)


if __name__ == "__main__":
    main()
