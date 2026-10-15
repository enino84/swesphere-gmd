# -*- coding: utf-8 -*-
"""Figures and tables for the GMD manuscript, from whatever results exist.

Every panel is skipped with a message if its CSV is not there yet, so this
can be run while the long experiments are still going.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))
FIG = Path("figures")
NAVY, SKY, DEEP, AMBER, MUTED, RULE = (
    "#0F2B4C", "#4BA2DE", "#1F5FA8", "#F0A92B", "#5B6F84", "#D6E3EE")

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": NAVY,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "text.color": NAVY, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": True,
    "grid.color": RULE, "grid.linewidth": 0.6,
})


def load(name):
    p = RESULTS / f"{name}.csv"
    if not p.exists():
        print(f"  skip {name}: not run yet")
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def save(fig, stem):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {stem}")


def fig_convergence():
    df = load("exp01_convergence")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, sponge in zip(axes, (0, 1)):
        d = df[df.sponge == sponge]
        for alpha, c, mk in zip(sorted(d.alpha_deg.unique()),
                                (DEEP, SKY, AMBER), ("o", "s", "^")):
            g = d[d.alpha_deg == alpha].sort_values("LMAX")
            if not len(g):
                continue
            ax.loglog(g.Nlat, g.l2_h, marker=mk, color=c, lw=1.6,
                      label=f"$\\alpha$ = {alpha:.0f}$^\\circ$")
        # second-order reference slope
        if len(d):
            n = np.array(sorted(d.Nlat.unique()), dtype=float)
            ref = d.l2_h.max() * (n[0] / n) ** 2
            ax.loglog(n, ref, ls=":", color=MUTED, lw=1.2, label="2nd order")
        ax.set_xlabel("latitude rows")
        ax.set_title("sponge on" if sponge else "sponge off", loc="left")
    axes[0].set_ylabel("normalized $l_2$ error of $h$")
    axes[1].legend(fontsize=8)
    save(fig, "fig_convergence")


def fig_stationarity():
    df = load("exp05_stationarity")
    win = load("exp05_windows")
    if df is None:
        return
    q = "eddy_std_h"
    d = df[df.quantity == q]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels, y = [], 0
    for (preset, seed), g in d.groupby(["preset", "seed"]):
        r = g.iloc[0]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=DEEP, lw=3, solid_capstyle="butt")
        ax.plot([r["mean"]], [y], "o", color=NAVY, ms=5)
        if win is not None:
            w = win[(win.preset == preset) & (win.seed == seed)
                    & (win.quantity == q) & (win.n_windows == 2)]
            for _, rw in w.iterrows():
                ax.plot([rw.ci_lo, rw.ci_hi], [y + 0.22, y + 0.22],
                        color=AMBER, lw=1.6, solid_capstyle="butt")
        labels.append(f"{preset} / seed {seed}")
        y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("eddy standard deviation of $h$ (m)")
    ax.set_title("record mean with a block-bootstrap interval,\n"
                 "and the two half-record intervals in amber", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_stationarity")


def fig_assimilation():
    df = load("exp08_assimilation")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, col, title in ((axes[0], "rmse_h_a", "observed: depth"),
                           (axes[1], "incr_wind", "unobserved: wind increment")):
        for (method, cross), g in df.groupby(["method", "cross"]):
            g = g.groupby("cycle")[col].mean()
            ax.plot(g.index, g.values, lw=1.6,
                    color=DEEP if cross else MUTED,
                    ls="-" if method == "letkf" else "--",
                    label=f"{method}, cross={bool(cross)}")
        ax.set_xlabel("cycle")
        ax.set_title(title, loc="left")
        ax.set_yscale("log")
    axes[0].set_ylabel("analysis RMSE (m)")
    axes[1].set_ylabel("mean |$\\Delta$(u, v)| (m s$^{-1}$)")
    axes[1].legend(fontsize=7)
    save(fig, "fig_assimilation")


def tables():
    for name in ("exp02_rossby_haurwitz", "exp03_galewsky", "exp06_stability",
                 "exp07_filter_protocol", "exp09_timing"):
        df = load(name)
        if df is None:
            continue
        out = FIG / f"table_{name}.tex"
        try:
            out.write_text(df.to_latex(index=False, float_format="%.4g"))
        except ImportError:
            # pandas routes to_latex through its Styler, which wants jinja2.
            # The tables are small; writing them by hand costs nothing and
            # removes a dependency from the image.
            cols = list(df.columns)
            fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
            body = " \\\\\n".join(
                " & ".join(fmt(v) for v in row) for row in df.itertuples(index=False))
            out.write_text(
                "\\begin{tabular}{" + "l" * len(cols) + "}\n\\hline\n"
                + " & ".join(c.replace("_", " ") for c in cols)
                + " \\\\\n\\hline\n" + body
                + " \\\\\n\\hline\n\\end{tabular}\n")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    print("figures from", RESULTS.resolve())
    fig_convergence()
    fig_stationarity()
    fig_assimilation()
    tables()
    print("done")
