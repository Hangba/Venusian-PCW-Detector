r"""Portable New Computer Modern matplotlib theme.

Self-contained: works in any project, with the font bundled next door in
``assets/``. Font resolution order (first hit wins):

1. ``$PAPER_THEME_FONT``
2. a NewCM font in the current project (``assets/fonts/``, ``paper/fonts/``, ``fonts/``)
3. the copy bundled in the manuscript font directory
4. graceful degrade to a Computer-Modern-ish serif stack + matplotlib's own
   ``cm`` mathtext, so a plot still looks close instead of falling all the way
   back to DejaVu Sans

Usage::

    from paper_theme import paper_theme

    with paper_theme(font_size=11):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel(r"$f\,/\,f_{cp}$")
        fig.savefig("out.png", dpi=300, bbox_inches="tight")
        fig.savefig("out.pdf", bbox_inches="tight")

Both the plotting calls and ``savefig`` must sit inside the ``with`` block:
tick labels are regenerated at save time and would otherwise fall back to the
default font, leaving one figure with two typefaces.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

FONT_FILENAME = "NewCMMath-Book.otf"

_BUNDLED = Path(__file__).resolve().parent.parent / "assets/fonts" / FONT_FILENAME

_PROJECT_PATHS = (
    "assets/fonts",  # VenusExpress layout
    "paper/fonts",
    "fonts",
    "assets",
)

# Reasonable stand-ins if no NewCM file is reachable. All are Computer Modern
# descendants except the last, which is just the least-bad default.
_SERIF_FALLBACKS = [
    "New Computer Modern",
    "CMU Serif",
    "Latin Modern Roman",
    "Libertinus Serif",
    "DejaVu Serif",
]


def find_font() -> Path | None:
    """Locate a New Computer Modern OTF, or None if nothing is reachable."""
    env = os.environ.get("PAPER_THEME_FONT")
    if env and Path(env).expanduser().is_file():
        return Path(env).expanduser()

    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        for rel in _PROJECT_PATHS:
            candidate = parent / rel / FONT_FILENAME
            if candidate.is_file():
                return candidate
        if (parent / ".git").exists():
            break  # stop at the repo root; don't wander up into $HOME

    return _BUNDLED if _BUNDLED.is_file() else None


def paper_rc(font_size: float | None = None, dpi: float | None = None) -> dict:
    """rcParams for the paper theme.

    ``font_size`` sets the base size and derives the rest of the ladder
    (titles +2, labels +1, legend -1) so text scales with the canvas instead of
    being pinned per-element. Leave it None to keep whatever is already set.
    """
    import matplotlib.font_manager as fm

    font_path = find_font()
    if font_path is not None:
        fm.fontManager.addfont(str(font_path))
        family = fm.FontProperties(fname=str(font_path)).get_name()
        rc = {"font.family": family}
    else:
        rc = {"font.family": "serif", "font.serif": _SERIF_FALLBACKS}

    rc.update(
        {
            # Computer Modern math so $..$ subscripts match the surrounding text.
            "mathtext.fontset": "cm",
            # Keep text as text in vector output: the SVG/PDF stays editable in
            # Illustrator or Inkscape instead of every glyph becoming a path.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.formatter.use_mathtext": True,
        }
    )

    if font_size is not None:
        rc.update(
            {
                "font.size": font_size,
                "axes.titlesize": font_size + 2,
                "axes.labelsize": font_size + 1,
                "legend.fontsize": max(font_size - 1, 8),
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "axes.linewidth": 0.9,
            }
        )
    if dpi is not None:
        rc["savefig.dpi"] = dpi
    return rc


@contextmanager
def paper_theme(font_size: float | None = None, dpi: float | None = None):
    """Render inside the paper theme. Wrap the plotting AND the savefig."""
    import matplotlib

    with matplotlib.rc_context(paper_rc(font_size=font_size, dpi=dpi)):
        yield


def font_report() -> str:
    """One-line diagnostic: which font actually got picked up."""
    path = find_font()
    if path is None:
        return f"no {FONT_FILENAME} found; degrading to {_SERIF_FALLBACKS[0]}-ish serif stack"
    origin = "bundled with skill" if path == _BUNDLED else "project"
    return f"using {path} ({origin})"


if __name__ == "__main__":
    print(font_report())
