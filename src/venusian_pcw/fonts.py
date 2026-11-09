"""Font selection for the figure notebook."""

from functools import lru_cache
import os
from pathlib import Path

from venus_express.repro.settings import ROOT

BUNDLED_FONTS = ROOT / "src" / "venus_express" / "assets" / "fonts"
CJK_FALLBACKS = (
    "Source Han Serif SC",
    "Noto Serif CJK SC",
    "Noto Sans CJK SC",
    "Songti SC",
    "PingFang SC",
    "Microsoft YaHei",
    "WenQuanYi Zen Hei",
)


def font_paths(paths):
    configured = os.environ.get("VENUSIAN_PCW_FONT_PATH") or os.environ.get(
        "JIEBANG_CJK_FONT_DIR"
    )
    supplied = [*paths, *(configured.split(os.pathsep) if configured else ())]
    directories = [Path(path).expanduser().resolve() for path in supplied]
    for path in directories:
        if not path.is_dir():
            raise ValueError(f"Font directory does not exist: {path}")
    return list(dict.fromkeys([*directories, BUNDLED_FONTS]))


def plot_families(families, paths):
    from matplotlib import font_manager

    for path in sorted(font_manager.findSystemFonts(fontpaths=font_paths(paths))):
        font_manager.fontManager.addfont(path)

    def installed(family):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=[family]), fallback_to_default=False
            )
        except ValueError:
            return False
        return True

    missing = [family for family in families if not installed(family)]
    if missing:
        raise ValueError(
            f"Font families not found: {', '.join(missing)}. Add a --font-path or use installed families."
        )

    latin = font_manager.FontProperties(
        fname=BUNDLED_FONTS / "NewCMMath-Book.otf"
    ).get_name()
    fallback = [name for name in (latin, *CJK_FALLBACKS) if installed(name)]
    return list(dict.fromkeys([*families, *fallback, "DejaVu Sans"]))


@lru_cache(maxsize=32)
def mixed_text_family(families):
    """Find a family with Chinese glyphs for labels that also use MathText."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    glyphs = set(map(ord, "金星磁场时间频率"))
    for family in families:
        path = font_manager.findfont(
            font_manager.FontProperties(family=[family]), fallback_to_default=False
        )
        if glyphs <= FT2Font(path).get_charmap().keys():
            return family
    return None
