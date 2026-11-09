"""Render the reference figures for the Venus Express PCW study."""

from contextlib import contextmanager
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MarkerPath
from matplotlib import patheffects
from matplotlib.text import Text
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import NullFormatter, PercentFormatter
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import distance_transform_edt
from scipy.special import expit
from scipy.stats import t

from venus_express.sample_analysis.style import analysis_figure_style
from venus_express.constants import R_VENUS
from venus_express.physics.bow_shock import BOW_SHOCK_SOLAR_MINIMUM, vso_to_aberrated_vso
from venus_express.sample_analysis.plots.rx_common import (
    add_rx_references,
    format_rx_axis,
    prepare_rx_coordinates,
)

from venus_express.repro.settings import INPUTS, BUILD, FIGURES as OUT
from .boundaries import bow_shock_data, bow_shock_curve
from .mag import load_day, fac
from .fonts import mixed_text_family

INK, GRAY, BLUE = "#20252b", "#67717c", "#1f5f8b"
GREEN, RED, GOLD = "#347758", "#9e4e48", "#9a6c2c"
LIGHT = "#c3cad1"
STAGES = ["MSeed", "MExtend1", "MExtend2", "MStandard"]
STYLES = [(GRAY, "--", "o"), (GREEN, ":", "s"), (RED, "-.", "^"), (BLUE, "-", "D")]


def save(fig, name):
    # MathText renders surrounding prose with its primary font.
    for label in fig.findobj(Text):
        value = label.get_text()
        if "$" in value and any("\u3400" <= char <= "\u9fff" for char in value):
            family = mixed_text_family(tuple(matplotlib.rcParams["font.family"]))
            if family:
                label.set_fontfamily(family)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def panel(ax, letter, x=0.02, y=0.975):
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        va="top",
        weight="bold",
        zorder=30,
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )


def axes_style(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="in", length=3.4, width=0.7, color=INK)
    if grid:
        ax.grid(axis="y", lw=0.45, color=LIGHT, alpha=0.5)
        ax.set_axisbelow(True)


def axis_break(ax, axis, position):
    """Mark the omitted lower portion of a truncated linear axis."""
    if axis == "x":
        anchor = (position, 0)
        mask = [(-1, 0), (1, 0)]
        strokes = [(-0.67, -0.70), (-0.16, 0.70), (0.16, -0.70), (0.67, 0.70)]
    else:
        anchor = (0, position)
        mask = [(0, -1), (0, 1)]
        strokes = [(-0.68, -0.65), (0.68, -0.20), (-0.68, 0.20), (0.68, 0.65)]
    for vertices, codes, color, weight in [
        (mask, [1, 2], "white", 1.45),
        (strokes, [1, 2, 1, 2], INK, 0.62),
    ]:
        ax.plot(
            *anchor,
            transform=ax.transAxes,
            marker=MarkerPath(vertices, codes),
            markersize=4.8,
            markeredgecolor=color,
            markeredgewidth=weight,
            linestyle="none",
            clip_on=False,
            zorder=40,
        )


def node(ax, x, y, w, h, title, detail=None, active=False, size=9):
    ax.add_patch(
        Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            linewidth=0.7,
            facecolor="#edf3f7" if active else "white",
            edgecolor=BLUE if active else GRAY,
            zorder=3,
        )
    )
    ax.text(
        x,
        y + (h * 0.21 if detail else 0),
        title,
        ha="center",
        va="center",
        fontsize=size,
        weight="bold" if detail else "normal",
        color=INK,
        zorder=4,
    )
    if detail:
        ax.text(
            x,
            y - h * 0.18,
            detail,
            ha="center",
            va="center",
            fontsize=size - 0.35,
            color=INK,
            linespacing=1.25,
            zorder=4,
        )


def arrow(ax, start, stop, color=GRAY, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            stop,
            arrowstyle="-|>",
            mutation_scale=8,
            shrinkA=0,
            shrinkB=0,
            linewidth=0.8,
            linestyle="--" if dashed else "-",
            color=color,
            zorder=2,
        )
    )


def schematic(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def architecture():
    fig, ax = schematic(6.45, 6.4)
    for y, title in [
        (0.974, "(a)  窗口编码与双支路"),
        (0.624, "(b)  单支路的频率残差编码器"),
        (0.302, "(c)  频率注意力与支路对数几率"),
    ]:
        ax.text(0.017, y, title, fontsize=10, weight="bold", va="top")
    for y in [0.649, 0.326]:
        ax.hlines(y, 0.016, 0.985, color=LIGHT, lw=0.65)
    node(ax, 0.135, 0.812, 0.236, 0.107, "80 秒窗口", r"$X:\ 4\times54\times80$" + "\n四个物理通道")
    node(ax, 0.385, 0.812, 0.205, 0.107, "时间平均", r"$4\times54$" + "\n共享通道掩码")
    arrow(ax, (0.253, 0.812), (0.2825, 0.812))
    for y, title, detail, lab, active in [
        (
            0.878,
            "局部支路",
            r"重采样至 $0.6$–$1.4\,f_{cp}$" + "\n编码与分类见（b、c）",
            r"$z_L$",
            True,
        ),
        (0.745, "全频支路", "原网格（54 点）\n编码与分类见（b、c）", r"$z_C$", False),
    ]:
        node(ax, 0.64, y, 0.252, 0.101, title, detail, active)
        ax.plot([0.4875, 0.508, 0.508], [0.812, 0.812, y], lw=0.8, color=GRAY)
        arrow(ax, (0.508, y), (0.514, y))
        ax.plot([0.766, 0.812, 0.812], [y, y, 0.812], color=GRAY, lw=0.8)
        ax.text(0.793, y + 0.012, lab, va="bottom", ha="center", fontsize=9.1)
    node(ax, 0.904, 0.812, 0.153, 0.107, "融合", r"$z=z_L+z_C$" + "\n逻辑函数")
    arrow(ax, (0.812, 0.812), (0.8275, 0.812))
    ax.text(0.365, 0.691, "仅训练时使用掩码", ha="center", fontsize=8.4)
    ax.text(0.903, 0.712, r"$s_{\rm PCW}$", ha="center", va="top", fontsize=9.5)
    arrow(ax, (0.904, 0.7585), (0.904, 0.722))

    xs = [0.171, 0.50, 0.83]
    for x, title, detail in [
        (xs[0], "卷积块 1", r"$\mathrm{Conv}_{5}: 4\to20$" + "\n四组归一化、整流"),
        (
            xs[1],
            "卷积块 2",
            r"$\mathrm{Conv}_{3,5,7}: 20\to3\times12$" + "\n拼接、六组归一化、残差、整流",
        ),
        (xs[2], "卷积块 3", r"$\mathrm{Conv}_{3}: 36\to48$" + "\n六组归一化、残差、整流"),
    ]:
        node(
            ax,
            x,
            0.48,
            0.271,
            0.123,
            title,
            detail,
            active=x == xs[1],
            size=8.2 if x == xs[1] else 9.1,
        )
    for i, name in enumerate([r"$20\times54$", r"$36\times54$"]):
        x = xs[i] + 0.1355
        arrow(ax, (x, 0.48), (xs[i + 1] - 0.1355, 0.48))
        ax.text((xs[i] + xs[i + 1]) / 2, 0.556, name, ha="center", fontsize=8.8)
    for x, label in [
        (xs[1], r"残差：20 个通道补 16 个零通道"),
        (xs[2], r"残差：36 个通道补 12 个零通道"),
    ]:
        left = x - 0.163
        ax.plot([left, left, x], [0.48, 0.389, 0.389], color=GRAY, lw=0.8, ls="--")
        arrow(ax, (x, 0.389), (x, 0.4185), dashed=True)
        ax.text((left + x) / 2, 0.363, label, ha="center", fontsize=8.6)
    ax.text(0.171, 0.399, "频率维保持 54 点", ha="center", fontsize=8.6)

    node(ax, 0.149, 0.159, 0.248, 0.112, "频率特征", r"$H: 48\times54$")
    node(
        ax,
        0.436,
        0.159,
        0.250,
        0.112,
        "注意力池化",
        r"$\alpha=\mathrm{softmax}_f(e+\log g)$" + "\n" + "沿频率轴加权求和",
        True,
        size=8.6,
    )
    node(
        ax,
        0.767,
        0.159,
        0.337,
        0.112,
        "分类头",
        r"$48\to48\to1$" + "\n层归一化、整流、训练时随机失活",
    )
    for a, b in [(0.273, 0.311), (0.561, 0.5985), (0.9355, 0.967)]:
        arrow(ax, (a, 0.159), (b, 0.159))
    ax.text(0.974, 0.159, r"$z_b$", ha="left", va="center", fontsize=9)
    ax.text(
        0.437,
        0.043,
        r"$e=\mathrm{Conv}_{1}(H)$ 为学习得到的证据，$g$ 为固定回旋频率先验",
        ha="center",
        fontsize=8.6,
    )
    save(fig, "02_model_architecture")


def model_results():
    selected = pd.read_csv(BUILD / "windows/matched_49_of_61.csv")
    blind = pd.read_csv(BUILD / "review_ranked.csv")
    logits = pd.read_csv(BUILD / "windows/MStandard.2026.test.csv", float_precision="round_trip")
    fig, axes = plt.subplots(2, 2, figsize=(6.45, 4.9), layout="constrained")
    a, b, c, d = axes.flat
    matched = selected.groupby("stage").mean(numeric_only=True).loc[STAGES]
    stages = np.arange(len(STAGES))
    for ax, values, caption in [
        (a, matched.bg_accepted, "普通背景误通过数（窗／种子）"),
        (b, matched.F_rejection, "超低频波拒绝率"),
    ]:
        ax.plot(stages, values, color=BLUE, marker="o", lw=1.2, ms=4)
        ax.set(xticks=stages, xticklabels=STAGES, xlim=(-0.25, 3.25), ylabel=caption)
        ax.tick_params(axis="x", labelsize=7.9)
        for n, value in zip(stages, values):
            label = f"{value:.2f}" if ax == a else f"{value:.2%}"
            alignment = {0: "left", len(STAGES) - 1: "right"}.get(n, "center")
            ax.annotate(
                label,
                (n, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha=alignment,
                fontsize=7.5,
            )
    a.set(ylim=(0, 1.65), yticks=[0, 0.5, 1, 1.5])
    b.set(ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1])
    b.yaxis.set_major_formatter(PercentFormatter(1, 0))
    score = expit(logits["logit"].astype(float))
    for key, label, color, ls in [
        ("bg", "普通背景", GRAY, "--"),
        ("F", "超低频波", RED, ":"),
        ("P", "质子回旋波", BLUE, "-"),
    ]:
        v = np.sort(score[logits["source"] == key])
        c.step(
            np.r_[0, v, 1],
            np.r_[0, np.arange(1, len(v) + 1) / len(v), 1],
            where="post",
            color=color,
            lw=1.15,
            ls=ls,
            label=label,
        )
    cutoff = selected[(selected.stage == "MStandard") & (selected.seed == 2026)].threshold.item()
    c.axvline(expit(cutoff), color=INK, ls=":", lw=0.85)
    c.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="MStandard 评分", ylabel="经验累积分布")
    c.legend(loc="center left", bbox_to_anchor=(0.27, 0.38), fontsize=7.6)
    d.step(blind["rank"], blind.precision_at_k, color=BLUE, where="post", lw=1.25)
    for rank in (10, 25, 50):
        y = blind.loc[blind["rank"] == rank, "precision_at_k"].item()
        d.plot(rank, y, "o", color=BLUE, ms=3.5)
        d.annotate(
            f"{y:.0%}",
            (rank, y),
            xytext=(0, 10 if rank == 10 else 12),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
        )
    d.set(xlim=(0, 52), ylim=(0.58, 1.04), xlabel="累计复审候选数", ylabel="累计精确率")
    d.yaxis.set_major_formatter(PercentFormatter(1, 0))
    for ax, letter in zip(axes.flat, "abcd"):
        axes_style(ax)
        panel(ax, letter)
    axis_break(d, "y", 0.09)
    save(fig, "03_model_evaluation")


def field_axes(ax, data):
    for name, color, ls, label in [
        ("bx", BLUE, "-", r"$B_x$"),
        ("by", GREEN, "-", r"$B_y$"),
        ("bz", RED, "-", r"$B_z$"),
        ("bt", INK, "-", r"$B_T$"),
    ]:
        ax.plot(
            data.index,
            data[name],
            color=color,
            ls=ls,
            lw=0.66 if name != "bt" else 0.95,
            label=label,
        )
    ax.set_ylabel("磁场（nT）")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncols=4,
        frameon=False,
        borderaxespad=0,
        handlelength=1.6,
    )
    axes_style(ax)


def time_format(ax, start, end, interval):
    ax.set(xlim=(pd.Timestamp(start), pd.Timestamp(end)), xlabel="世界时（时:分）")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, interval)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def wave_event():
    day = "2014-02-22"
    start, end = pd.Timestamp(day + " 00:00"), pd.Timestamp(day + " 04:15")
    data = (
        load_day(day, start, end)
        .resample("1s")
        .mean(numeric_only=True)
        .reindex(pd.date_range(start, end, freq="s"))
    )
    assert data.bt.notna().mean() > 0.8
    first, second, _, bmag = fac(data, 0.01)
    frequencies = np.geomspace(0.015, 0.32, 92)
    wav = "cmor1.5-1.0"
    scale = pywt.central_frequency(wav) / frequencies
    wave1, _ = pywt.cwt(first, scale, wav, method="fft")
    wave2, _ = pywt.cwt(second, scale, wav, method="fft")
    log_power = np.log10(np.abs(wave1) ** 2 + np.abs(wave2) ** 2 + 1e-8)
    # Raw missing seconds and the wavelet edge region are visually suppressed.
    missing = data[["bx", "by", "bz"]].isna().any(axis=1).to_numpy()
    edge = np.arange(len(data))
    outside = np.minimum(edge, edge[::-1])[None, :] < np.ceil(np.sqrt(2) * scale[:, None])
    gap_lengths = (
        pd.Series(missing)
        .groupby(np.cumsum(np.r_[True, missing[1:] != missing[:-1]]))
        .transform("sum")
        .to_numpy()
    )
    around_gap = missing & (gap_lengths > 30)
    if around_gap.any():
        outside |= distance_transform_edt(~around_gap)[None, :] < np.ceil(
            np.sqrt(2) * scale[:, None]
        )
    log_power = np.ma.masked_where(missing[None, :] | outside, log_power)
    fcp = bmag * 0.01524518646
    marks = pd.read_csv(INPUTS / "manual/hero_events.csv", parse_dates=["start", "end"])
    assert len(marks) == 8
    fig = plt.figure(figsize=(6.45, 4.05), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 0.038), height_ratios=(1, 1.2))
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[1, 0], sharex=a)
    field_axes(a, data)
    a.tick_params(labelbottom=False)
    for i, r in enumerate(marks.itertuples(), 1):
        a.axvspan(r.start, r.end, color=BLUE, alpha=0.16, lw=0)
        a.text(
            r.start + (r.end - r.start) / 2,
            0.87,
            str(i),
            transform=a.get_xaxis_transform(),
            fontsize=8,
            ha="center",
            color=BLUE,
        )
        for instant in (r.start, r.end):
            b.axvline(instant, color="white", lw=0.45, alpha=0.78)
    bs = pd.Timestamp(day + " 02:30:23")
    for ax in (a, b):
        ax.axvline(bs, color=GOLD if ax == a else "white", lw=0.9, ls=":")
    a.text(
        bs + pd.Timedelta(minutes=3), 0.84, "弓激波", transform=a.get_xaxis_transform(), color=GOLD
    )
    im = b.pcolormesh(
        data.index,
        frequencies,
        log_power,
        cmap="turbo",
        vmin=-2.5,
        vmax=1.6,
        shading="auto",
        rasterized=True,
    )
    b.plot(data.index, np.where(~missing, fcp, np.nan), color="white", ls="--", lw=0.8)
    b.set(yscale="log", ylim=(0.015, 0.32), ylabel="频率（Hz）")
    b.set_yticks([0.02, 0.05, 0.10, 0.20, 0.30], labels=["0.02", "0.05", "0.10", "0.20", "0.30"])
    b.yaxis.set_minor_formatter(NullFormatter())
    time_format(b, start, end, 30)
    panel(a, "a")
    panel(b, "b")
    fig.colorbar(
        im,
        cax=fig.add_subplot(grid[1, 1]),
        ticks=[-2, -1, 0, 1],
        label=r"$\log_{10}(P_w / \mathrm{nT}^2)$",
        extend="both",
    )
    save(fig, "04_pcw_event")


def annual_response():
    occ = pd.read_csv(BUILD / "occurrence/annual_rates_with_silso.csv")
    energy = pd.read_csv(BUILD / "energy/annual_energy.csv")
    assert occ[occ.threshold == 0.7].clusters.sum() == 1150 and energy.n.sum() == 796
    fig, (a, b, c) = plt.subplots(
        3,
        1,
        figsize=(6.45, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 0.68]},
        layout="constrained",
    )
    for value, (color, ls, mark) in zip([0.5, 0.6, 0.8, 0.7], STYLES):
        q = occ[occ.threshold == value]
        a.plot(
            q.year,
            q.clusters_per_observed_day,
            color=color,
            ls=ls,
            lw=1.0 if value != 0.7 else 1.6,
            marker=mark,
            ms=3,
            label=f"{value:.2f}",
            zorder=4 if value == 0.7 else 2,
        )
    a.legend(
        title="评分阈值",
        loc="upper left",
        bbox_to_anchor=(0.10, 1),
        ncols=4,
        fontsize=8,
        handlelength=1.6,
    )
    a.set(ylim=(0, 0.85), ylabel="事件簇数\n（每观测日）")
    y = energy.adjusted_geometric_mean_energy
    b.errorbar(
        energy.year,
        y,
        yerr=[y - energy.adjusted_ci95_low, energy.adjusted_ci95_high - y],
        color=BLUE,
        fmt="o-",
        lw=1.25,
        ms=3,
        capsize=2.2,
    )
    b.set(ylabel=r"波能（pJ m$^{-3}$）", ylim=(0, 0.088))
    c.bar(energy.year, energy.sunspot_number, width=0.6, color=LIGHT, edgecolor=GRAY, lw=0.35)
    c.set(xlabel="年份", ylabel="太阳黑子数", xticks=range(2006, 2015), ylim=(0, 130))
    for ax, letter in zip((a, b, c), "abc"):
        axes_style(ax)
        panel(ax, letter)
    save(fig, "05_annual_response")


def density():
    root = BUILD / "density"
    events = pd.read_csv(root / "exospheric_events.csv")
    p = pd.read_csv(root / "square_12Rv_bootstrap_profile.csv")
    assert len(events) == 78
    fig, ax = plt.subplots(figsize=(5.45, 3.8), layout="constrained")
    x = p.alt_km / 1000
    ax.fill_between(x, p.n_lo, p.n_hi, color=BLUE, alpha=0.14, lw=0, label="逐高度 95% 区间")
    ax.scatter(
        events.alt_km / 1000,
        events.n_H_neutral_cm3,
        s=15,
        facecolors="none",
        edgecolors=GRAY,
        linewidths=0.6,
        alpha=0.8,
        label="由质子回旋波反演",
    )
    inside = p.within_event_altitude_range
    ax.plot(x, p.n_fit, color=BLUE, ls="--", lw=1.25)
    ax.plot(x[inside], p.n_fit[inside], color=BLUE, lw=1.55, label="Chamberlain 模型拟合")
    ax.plot(x, p.SPICAV_hot_cm3, color=GREEN, ls=":", lw=1.4)
    obs = x <= 8
    ax.plot(x[obs], p.SPICAV_hot_cm3[obs], color=GREEN, lw=1.35, label="Chaufray 等日侧热氢参考")
    q = p.loc[np.isclose(p.alt_km, 25000)].iloc[0]
    ax.errorbar(
        25,
        q.n_fit,
        yerr=[[q.n_fit - q.n_lo], [q.n_hi - q.n_fit]],
        color=BLUE,
        fmt="D",
        ms=4,
        capsize=2.5,
        zorder=5,
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(4.2, 73),
        ylim=(0.1, 200),
        xlabel=r"距金星表面高度（$10^3$ km）",
        ylabel=r"中性氢密度（cm$^{-3}$）",
    )
    ax.set_xticks([5, 8, 15, 25, 40, 70], labels=["5", "8", "15", "25", "40", "70"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    axes_style(ax)
    ax.legend(loc="upper right", fontsize=8.4)
    save(fig, "06_h_density")


def solar_cycle():
    raw = pd.read_csv(
        INPUTS / "silso_monthly.csv",
        sep=";",
        header=None,
        usecols=[0, 1, 3],
        names=["year", "month", "sunspots"],
    )
    q = raw[raw.year.between(2007, 2014)].copy()
    q["date"] = pd.to_datetime(q[["year", "month"]].assign(day=15))
    fig, ax = plt.subplots(figsize=(6.0, 2.65), layout="constrained")
    ax.plot(q.date, q.sunspots, color=INK, marker="o", ms=2.4, lw=0.8, label="月平均黑子数")
    for start, end, color, label in [
        ("2008-06-20", "2009-05-31", GRAY, "低活动期"),
        ("2013-10-01", "2014-09-30", BLUE, "高活动期"),
    ]:
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.13, lw=0, label=label
        )
    ax.set(
        xlim=(pd.Timestamp("2007-01-01"), pd.Timestamp("2015-01-01")),
        ylim=(0, 161),
        xlabel="年份",
        ylabel="太阳黑子数",
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncols=3, fontsize=8.3)
    axes_style(ax)
    save(fig, "07_solar_cycle")


def shock_figures():
    old, points = bow_shock_data()
    groups = ["0—45°", "80—100°"]
    n = {"low": [99, 40], "high": [68, 24]}
    fig, ax = plt.subplots(figsize=(5.6, 2.85), layout="constrained")
    for period, color, marker, shift, label in [
        ("low", GRAY, "o", -0.12, "低活动期"),
        ("high", BLUE, "s", 0.12, "高活动期"),
    ]:
        for i, g in enumerate(groups):
            row = old[period, g]
            mean, sd = row["mean"], row["sd"]
            ci = t.ppf(0.975, n[period][i] - 1) * sd / np.sqrt(n[period][i])
            ax.errorbar(i + shift, mean, yerr=sd, color=color, alpha=0.55, lw=0.8, fmt="none")
            ax.errorbar(
                i + shift,
                mean,
                yerr=ci,
                color=color,
                lw=1.6,
                capsize=3,
                fmt=marker,
                ms=5,
                label=label if i == 0 else None,
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["0–45", "80–100"],
        xlim=(-0.52, 1.52),
        ylim=(0, 3.85),
        ylabel=r"磁场压缩比 $B_d/B_u$",
        xlabel="太阳天顶角（°）",
    )
    ax.legend(loc="upper right", ncols=2, fontsize=8.3)
    axes_style(ax)
    save(fig, "08_shock_compression")

    pars = {"low": (0.626, 1.559, 1.024), "high": (0.638, 1.687, 1.041)}
    fig, axes = plt.subplots(1, 3, figsize=(6.45, 2.55), sharey=True, layout="constrained")
    colors = {"low": GRAY, "high": BLUE}
    for ax, period, letter, title in zip(
        axes, ["low", "high", "both"], "abc", ["低活动期", "高活动期", "拟合曲线对比"]
    ):
        ax.add_patch(Circle((0, 0), 1, color=LIGHT, alpha=0.36, lw=0, zorder=0))
        if period in points:
            p = points[period]
            ax.scatter(p[:, 0], p[:, 1], s=3.6, c=colors[period], alpha=0.36, lw=0)
        for category in pars if period == "both" else [period]:
            ax.plot(
                *bow_shock_curve(*pars[category]),
                color=colors[category],
                ls="--" if category == "low" else "-",
                lw=1.15,
            )
        ax.set(
            xlim=(2.05, -3.1),
            ylim=(0, 4.0),
            xticks=[2, 0, -2],
            xlabel=r"$X_{\rm AVSO}/R_V$",
            title=title,
            aspect="equal",
        )
        ax.title.set_fontsize(9)
        axes_style(ax, False)
        panel(ax, letter)
    axes[0].set_ylabel(r"$\sqrt{Y^2+Z^2}/R_V$")
    axes[-1].legend(
        handles=[
            Line2D([0], [0], color=colors["low"], ls="--", label="低活动期"),
            Line2D([0], [0], color=colors["high"], label="高活动期"),
        ],
        loc="lower right",
        fontsize=8,
    )
    save(fig, "09_shock_geometry")


def radial_event():
    day = "2008-12-03"
    start, end = pd.Timestamp(day + " 08:00"), pd.Timestamp(day + " 11:30")
    data = (
        load_day(day, start, end)
        .resample("1s")
        .mean(numeric_only=True)
        .reindex(pd.date_range(start, end, freq="s"))
    )
    assert data.bt.notna().mean() > 0.8
    ratio = (
        (data.bx.abs() / data.bt.where(data.bt > 0)).rolling(60, center=True, min_periods=30).mean()
    )
    bounds = [pd.Timestamp(f"{day} {x}") for x in ("09:03:57", "09:16:48", "09:39:57", "10:09:35")]
    p, q = [pd.Timestamp(f"{day} {x}") for x in ("10:42:11", "10:43:34")]
    fig, (a, b) = plt.subplots(
        2,
        1,
        figsize=(6.45, 3.45),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.35, 1]},
    )
    field_axes(a, data)
    for ax in (a, b):
        ax.axvspan(bounds[0], bounds[-1], color=GRAY, alpha=0.09, lw=0)
        ax.axvspan(bounds[1], bounds[2], color=BLUE, alpha=0.11, lw=0)
        ax.axvspan(p, q, color=GREEN, alpha=0.3, lw=0)
        for i, instant in enumerate(bounds):
            ax.axvline(
                instant,
                color=BLUE if i in (1, 2) else GRAY,
                ls="--" if i in (1, 2) else ":",
                lw=0.9,
            )
    for instant, label in zip(bounds, ["入弓激波", "入边界", "出边界", "出弓激波"]):
        a.annotate(
            label,
            (instant, 0.96 if label == "入边界" else 0.87),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=7.7,
            color=INK,
        )
    a.annotate(
        "质子回旋波",
        (p + (q - p) / 2, 0.92),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="bottom",
        fontsize=7.7,
        color=GREEN,
    )
    b.plot(ratio.index, ratio, color=INK, lw=0.85)
    b.axhline(0.9, color=GOLD, ls=":", lw=0.8)
    b.set(ylim=(0, 1.045), ylabel=r"$\langle |B_x|/B_T \rangle_{60\,\mathrm{s}}$")
    time_format(b, start, end, 30)
    axes_style(b)
    panel(a, "a")
    panel(b, "b")
    save(fig, "10_radial_imf_event")


def rx_axis(ax, bounds, *, light=False, model=BOW_SHOCK_SOLAR_MINIMUM):
    xmin, xmax, rmax = bounds
    add_rx_references(ax, xmin, xmax, rmax, model=model, light=light)
    format_rx_axis(ax, xmin, xmax, rmax)
    ax.set_xlabel(r"$X_{\rm AVSO}/R_V$")
    ax.set_ylabel(r"$R_{\rm AVSO}/R_V$")
    axes_style(ax, grid=False)
    ax.grid(False)


def spatial_maps():
    root = BUILD / "occurrence"
    p = pd.read_csv(root / "cluster_positions.csv")
    grid = pd.read_csv(root / "cluster_occurrence.csv")
    kept = grid.loc[grid.displayed]
    assert len(p) == grid.cluster_count.sum() == 1150
    assert grid.observed_seconds.sum() == 233963418
    assert len(kept) == 376 and kept.cluster_count.sum() == 1092
    assert (kept.clusters_per_observed_day > 3).sum() == 8
    xmin = int(np.floor(grid.x_bin_min_rv.min()))
    xmax = int(np.ceil(grid.x_bin_max_rv.max()))
    rmax = int(np.ceil(grid.r_bin_max_rv.max()))
    bounds = (xmin, xmax, rmax)
    xe = np.arange(xmin, xmax + 0.5, 0.5)
    re = np.arange(0, rmax + 0.5, 0.5)
    values = np.full((len(re) - 1, len(xe) - 1), np.nan)
    for row in kept.itertuples():
        values[round(row.r_bin_min_rv / 0.5), round((row.x_bin_min_rv - xmin) / 0.5)] = (
            row.clusters_per_observed_day
        )
    fig = plt.figure(figsize=(6.45, 3.55), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.045))
    a, b = [fig.add_subplot(gs[0, k]) for k in (0, 1)]
    a.scatter(p.AVSO_X_Rv, p.AVSO_R_Rv, s=6, color=BLUE, lw=0, alpha=0.58)
    mesh = b.pcolormesh(
        xe, re, values, cmap="cividis", vmin=0, vmax=4, shading="flat", rasterized=True
    )
    fig.colorbar(
        mesh,
        cax=fig.add_subplot(gs[0, 2]),
        ticks=[0, 1, 2, 3, 4],
        label="发生率（簇／观测日）",
        extend="max",
    )
    rx_axis(a, bounds)
    rx_axis(b, bounds)
    for line in b.lines[:2]:
        line.set_path_effects(
            [patheffects.Stroke(linewidth=2.4, foreground="white", alpha=0.9), patheffects.Normal()]
        )
    b.set_ylabel("")
    b.tick_params(labelleft=False)
    for ax, label in [(a, "（a）事件簇位置"), (b, "（b）空间发生率")]:
        ax.set_title(label, loc="left", fontsize=9.4)
    save(fig, "11_rx_cluster_occurrence")

    p = prepare_rx_coordinates(pd.read_csv(INPUTS / "manual/spatial.csv"))
    assert len(p) == 946 and p.wave_energy_pJ_m3.gt(0).all()
    sun = p.plasma_region.eq("solar_wind")
    parallel = sun & p.shock_class.eq("quasi_parallel")
    other = sun & ~parallel
    sheath = p.plasma_region.eq("magnetosheath")
    assert (parallel.sum(), other.sum(), sheath.sum()) == (98, 699, 149)
    fig = plt.figure(figsize=(6.45, 3.85), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.045))
    a, b = [fig.add_subplot(gs[0, k]) for k in (0, 1)]
    for flag, kw, label in [
        (other, dict(s=8, facecolors="none", edgecolors=GRAY, lw=0.48), "其余太阳风事件"),
        (parallel, dict(s=10, color=BLUE, lw=0), "准平行激波上游"),
        (sheath, dict(s=12, color=RED, marker="x", lw=0.65), "磁鞘"),
    ]:
        a.scatter(p.RX_X_Rv[flag], p.RX_R_Rv[flag], label=label, alpha=0.75, **kw)
    a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=7.5,
        columnspacing=0.8,
        handletextpad=0.35,
        handlelength=1.0,
    )
    p = p.sort_values("wave_energy_pJ_m3")
    m = p.wave_energy_pJ_m3
    mesh = b.scatter(
        p.RX_X_Rv,
        p.RX_R_Rv,
        c=m,
        s=9,
        lw=0,
        cmap="viridis",
        norm=LogNorm(m.min(), m.quantile(0.99)),
        rasterized=True,
    )
    fig.colorbar(
        mesh,
        cax=fig.add_subplot(gs[0, 2]),
        ticks=[0.001, 0.01, 0.1, 1],
        label=r"波能（pJ m$^{-3}$）",
        extend="max",
    )
    for ax, label in [(a, "（a）人工目录的区域分类"), (b, "（b）事件波能")]:
        rx_axis(ax, bounds)
        ax.set_title(label, loc="left", fontsize=9.4)
    b.set_ylabel("")
    b.tick_params(labelleft=False)
    save(fig, "12_rx_manual_energy")


def rx_examples():
    windows = [
        ("2011-02-20", "19:57:36", "20:02:26", "（a）准垂直激波上游"),
        ("2014-02-22", "01:53:09", "01:56:27", "（b）准平行激波上游"),
        ("2009-10-12", "02:06:52", "02:14:37", "（c）磁鞘"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(6.45, 3.6), layout="constrained")
    tracks = []
    for day, t0, t1, label in windows:
        start, end = [pd.Timestamp(f"{day} {t}") for t in (t0, t1)]
        delta = pd.Timedelta(minutes=30)
        raw = load_day(day, start - delta, end + delta)
        coords = raw[["x", "y", "z"]].where(lambda p: p.abs() < 9e4).to_numpy(float) / R_VENUS
        points = vso_to_aberrated_vso(coords)
        x, r = points[:, 0], np.hypot(points[:, 1], points[:, 2])
        inside = raw.index.to_series().between(start, end).to_numpy()
        valid = np.flatnonzero(inside & np.isfinite(x) & np.isfinite(r))
        assert valid.size > 0
        tracks.append((x, r, inside, valid[len(valid) // 2], day, label))
    rmax = float(np.ceil(max(np.nanmax(r) for _, r, *_ in tracks))) + 0.5
    for i, (ax, (x, r, inside, mid, day, label)) in enumerate(zip(axs, tracks)):
        rx_axis(ax, (-4.5, 3.0, rmax))
        ax.plot(x, r, color=GRAY, lw=0.9, zorder=5)
        ax.plot(x[inside], r[inside], color=BLUE, lw=2.4, solid_capstyle="butt", zorder=6)
        ax.scatter(x[mid], r[mid], s=16, color=BLUE, edgecolors="white", lw=0.5, zorder=7)
        ax.set_xticks([2, 0, -2, -4])
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_title(label + "\n" + day, loc="left", fontsize=8.8, pad=4)
        if i:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    save(fig, "13_rx_case_orbits")


@contextmanager
def quiet_font_messages():
    """Suppress routine font fallback messages only while exporting figures."""
    fallback = logging.getLogger("matplotlib.font_manager")
    font_header = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")

    def keep_font_message(record):
        return not record.getMessage().startswith("findfont: Failed to find font weight bold, now using ")

    def keep_header_message(record):
        return record.getMessage() not in {
            "'created' timestamp seems very low; regarding as unix timestamp",
            "'modified' timestamp seems very low; regarding as unix timestamp",
        }

    fallback.addFilter(keep_font_message)
    font_header.addFilter(keep_header_message)
    try:
        yield
    finally:
        fallback.removeFilter(keep_font_message)
        font_header.removeFilter(keep_header_message)


@contextmanager
def figure_style(font_families):
    """Apply the selected fonts and shared plot styling to a notebook cell."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (
        quiet_font_messages(),
        analysis_figure_style(font_size=9.4),
        matplotlib.rc_context(
            {
                "font.family": font_families,
                "axes.labelsize": 9.4,
                "xtick.labelsize": 8.8,
                "ytick.labelsize": 8.8,
                "legend.frameon": False,
                "lines.solid_capstyle": "round",
            }
        ),
    ):
        yield
