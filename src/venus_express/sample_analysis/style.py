"""Figure style matching the secondary-calibration export figures."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib

from venus_express.export import export_figure_rc


@contextmanager
def analysis_figure_style(font_size: float = 12.0):
    """Use the bundled New Computer Modern font and consistent paper sizing."""

    rc = {
        **export_figure_rc(),
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size + 1,
        "legend.fontsize": max(font_size - 1, 8),
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.linewidth": 0.9,
        "savefig.dpi": 200,
    }
    with matplotlib.rc_context(rc):
        yield
