"""Reproduce the current 揭榜挂帅 paper on pickup ions at Venus."""

import argparse
import os
import subprocess

from venus_express.repro.settings import ROOT, BUILD, FIGURES, read

os.environ.setdefault("MPLCONFIGDIR", str(BUILD / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def compile_paper(generated):
    names = read(ROOT / "reference/paper_snapshot.json")["figures"]
    if generated:
        missing = sorted(name for name in names if not (FIGURES / name).is_file())
        if missing:
            raise FileNotFoundError(f"Run replay before compiling generated figures: {missing}")
    out = BUILD / "paper" / "揭榜挂帅论文-复现.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "typst",
            "compile",
            "--root",
            str(ROOT),
            "--font-path",
            str(ROOT / "paper/fonts"),
            "--input",
            f"figures={'generated' if generated else 'reference'}",
            str(ROOT / "paper/main.typ"),
            str(out),
        ],
        check=True,
    )
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify checksums of the frozen inputs")
    paper = sub.add_parser("paper", help="Compile the current 揭榜挂帅 Typst manuscript")
    paper.add_argument("--generated", action="store_true")
    replay = sub.add_parser("replay", help="Recompute the paper's tables and figures")
    replay.add_argument(
        "--raw-figures", action="store_true", help="Redraw the three BAM-based figures"
    )
    replay.add_argument(
        "--raw-search",
        action="store_true",
        help="Use the complete new mission and holdout searches",
    )
    replay.add_argument(
        "--raw-density", action="store_true", help="Recompute the 78 density PSDs from BAM"
    )
    replay.add_argument(
        "--rescore-windows",
        action="store_true",
        help="Score external frozen window features with the selected weights",
    )
    replay.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    search = sub.add_parser("search", help="Run the current detector on 1 Hz VSO MAG")
    search.add_argument("scope", choices=["mission", "holdout"])
    search.add_argument("--day", action="append")
    search.add_argument("--resume", action="store_true")
    search.add_argument("--workers", type=int, default=1)
    search.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    exposure = sub.add_parser("exposure", help="Rebuild the full AVSO exposure grid from BAM")
    exposure.add_argument("--workers", type=int, default=4)
    train = sub.add_parser("train", help="Fit a stage/seed on the frozen training arrays")
    train.add_argument(
        "stage", choices=["MSeed", "MExtend1", "MExtend2", "MStandard", "holdout2013"]
    )
    train.add_argument("--seed", type=int, choices=[2026, 2027, 2028], default=2026)
    train.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    audit = sub.add_parser("psd-audit", help="Recompute the full-frequency PSD audit from BAM")
    audit.add_argument("--workers", type=int, default=4)
    audit.add_argument("--day", action="append")
    ions = sub.add_parser(
        "ima-moments", help="Integrate an externally normalized IMA count and calibration bundle"
    )
    ions.add_argument("bundle", help="Path to the normalized NPZ described in docs/data.md")
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")

    if args.command == "verify":
        from .statistics import verify_inputs

        verify_inputs()
    elif args.command == "paper":
        compile_paper(args.generated)
    elif args.command == "replay":
        from .statistics import compute
        from .figures import render

        compute(args.raw_search, args.raw_density, args.rescore_windows, args.device)
        render(args.raw_figures or args.raw_search)
    elif args.command == "search":
        from venus_express.repro.search import search as run

        run(args.scope, args.device, args.workers, args.day, args.resume)
    elif args.command == "exposure":
        from venus_express.repro.exposure import compute

        compute(args.workers)
    elif args.command == "train":
        from venus_express.repro.training import fit

        fit(args.stage, args.seed, args.device)
    elif args.command == "psd-audit":
        from .spectral import compute

        compute(args.workers, args.day)
    elif args.command == "ima-moments":
        from .ions import compute

        compute(args.bundle)


if __name__ == "__main__":
    main()
