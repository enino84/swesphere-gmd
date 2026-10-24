# -*- coding: utf-8 -*-
"""The four figures that carry a measurement rather than a field.

``make_gallery.py`` draws what the flows look like. This file draws what was
measured about them, for the three sections whose strongest results lived
only in tables: the autocorrelation of Sect. 6, the error growth of Sect. 6
and the assimilation of Sect. 7. The fourth is a field figure, and it is here
rather than in the gallery because its job is an argument and not a portrait:
it shows the three prognostic variables of one state together, which is the
whole of why this testbed is not Lorenz 96.

Nothing is integrated here. Every panel reads a CSV or a ``.npy`` that an
experiment wrote, so a restyle costs seconds. A figure whose input is missing
is skipped with a message rather than failing the run.

The drawing helpers, the colour maps and the autocorrelation estimator are
imported rather than copied. The estimator in particular: the taus annotated
on Fig. 2 have to be the same numbers as ``exp05_stationarity.csv``, and the
only way to guarantee that is to call the same function.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from common import integral_autocorrelation_time          # noqa: E402
from make_gallery import (AMBER, BLUE, DIVERGING, GREEN,   # noqa: E402
                          MUTED, NAVY, RULE, SEQUENTIAL, colorbar, globe,
                          pos_norm,
                          load, save, si_scale, sym_norm, vorticity)

RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))
FIG = Path(os.environ.get("FIGURES_DIR", "figures"))

PRESET_COLOUR = {"waves": GREEN, "one_jet": BLUE, "two_jets": AMBER}
PRESET_LABEL = {"waves": "waves", "one_jet": "one jet", "two_jets": "two jets"}


def read(name):
    p = RESULTS / name
    if not p.exists():
        print(f"  skip: {p} is not there")
        return None
    return pd.read_csv(p)


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, lw=0.5, color=RULE)
    ax.set_axisbelow(True)


def plain_log(ax, axis="y", minor=True):
    """Plain numbers on a log axis.

    Matplotlib's default writes 6 as 6x10^0 on a log scale, which on a range
    like 3 to 12 is noise around the only thing the reader wants. ``minor``
    is for ranges under one decade, where the minor ticks carry the reading;
    over several decades their labels collide and are turned off instead.
    """
    a = ax.yaxis if axis == "y" else ax.xaxis
    f = ScalarFormatter(); f.set_scientific(False)
    a.set_major_formatter(f)
    if minor:
        g = ScalarFormatter(); g.set_scientific(False); a.set_minor_formatter(g)
    else:
        a.set_minor_formatter(NullFormatter())


def lead_axis(ax, days):
    """Tick a log lead-time axis at the lead times actually run."""
    ax.set_xscale("log")
    ax.set_xticks(days)
    ax.set_xticklabels([("%g" % d) for d in days])
    ax.xaxis.set_minor_formatter(NullFormatter())


# ----------------------------------------------------------------------
# Figure 1: the cycled assimilation
# ----------------------------------------------------------------------
#: The observation spacing the figure is drawn at. The other two are in
#: Table 8; drawing all three would triple the lines and change no conclusion.
SPACING = 4

STYLE = {
    ("letkf", 0): (BLUE, "--", "LETKF, univariate"),
    ("letkf", 1): (BLUE, "-", "LETKF, cross-variable"),
    ("enkf-modified-cholesky", 0): (AMBER, "--", "EnKF-MC, univariate"),
    ("enkf-modified-cholesky", 1): (AMBER, "-", "EnKF-MC, cross-variable"),
}


def fig_assimilation():
    print("fig_assimilation")
    d = read("exp08_assimilation.csv")
    if d is None:
        return
    d = d[(d.completed == 1) & (d.spacing == SPACING)]
    if d.empty:
        print(f"  skip: nothing at spacing {SPACING}")
        return
    d = d.copy()
    d["calib"] = d.spread_h_b / d.rmse_h_b
    g = d.groupby(["method", "cross", "cycle"])[
        ["rmse_h_b", "rmse_wind_b", "calib"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))
    panels = [("rmse_h_b", "background RMSE of $h$ (m)", True),
              ("rmse_wind_b", "background RMSE of wind (m s$^{-1}$)", True),
              ("calib", "ensemble spread / actual error", False)]
    for ax, (col, ylab, logy) in zip(axes, panels):
        for (m, c), (colour, ls, lab) in STYLE.items():
            s = g[(g.method == m) & (g.cross == c)].sort_values("cycle")
            if s.empty:
                continue
            ax.plot(s.cycle, s[col], ls, color=colour, lw=1.5, label=lab)
        if logy:
            ax.set_yscale("log")
            plain_log(ax)
        ax.set_xlabel("analysis cycle")
        ax.set_ylabel(ylab)
        tidy(ax)
    # A calibrated ensemble sits on this line. The reader should be able to
    # see which of the four does and which does not without reading a number.
    axes[2].axhline(1.0, color=MUTED, lw=1.0, ls=":")
    axes[2].text(0.98, 1.0, " calibrated", transform=axes[2].get_yaxis_transform(),
                 ha="right", va="bottom", color=MUTED, fontsize=8)
    # One legend for the three panels, below them: the four series are the
    # same everywhere, and inside a panel it sits on the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"observations every {SPACING} grid points, "
                 f"mean of three realizations", fontsize=9, color=MUTED, y=1.02)
    fig.tight_layout()
    save(fig, "fig_assimilation")


# ----------------------------------------------------------------------
# Figure 2: how long the flow remembers
# ----------------------------------------------------------------------
def series_from_record(S):
    """Eddy amplitude and peak wind per snapshot.

    The eddy amplitude is defined exactly as in ``exp04_regimes.py``: the
    standard deviation of the depth about its zonal mean. Version 0.6.0 runs
    with no sponge, so the interior is the whole field and no rows are cut.
    """
    n, dim = S.shape
    fs = dim // 3
    nlat = int(round((fs / 2) ** 0.5))
    nlon = 2 * nlat
    u = S[:, :fs].reshape(n, nlat, nlon)
    v = S[:, fs:2 * fs].reshape(n, nlat, nlon)
    h = S[:, 2 * fs:].reshape(n, nlat, nlon)
    eddy = (h - h.mean(axis=2, keepdims=True)).std(axis=(1, 2))
    umax = np.hypot(u, v).max(axis=(1, 2))
    return eddy, umax


def acf(x, max_lag):
    x = np.asarray(x, float) - np.mean(x)
    var = np.dot(x, x) / x.size
    if var <= 0:
        return np.zeros(max_lag + 1)
    return np.array([1.0] + [float(np.dot(x[:-k], x[k:]) / (x.size * var))
                             for k in range(1, max_lag + 1)])


def fig_autocorrelation():
    print("fig_autocorrelation")
    meta = read("exp04_regimes.csv")
    if meta is None:
        return
    every = float(meta.every_days.iloc[0])
    max_lag = 100
    presets = [p for p in ("waves", "one_jet", "two_jets")
               if p in set(meta.preset)]
    fig, axes = plt.subplots(1, len(presets), figsize=(3.6 * len(presets), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    drew = False
    for ax, preset in zip(axes, presets):
        rows = meta[meta.preset == preset]
        curves, taus = {"eddy": [], "umax": []}, {"eddy": [], "umax": []}
        for _, r in rows.iterrows():
            p = RESULTS / str(r.record_file)
            if not p.exists():
                continue
            eddy, umax = series_from_record(np.load(p).astype(float))
            for key, s in (("eddy", eddy), ("umax", umax)):
                curves[key].append(acf(s, max_lag))
                taus[key].append(integral_autocorrelation_time(s))
        if not curves["eddy"]:
            print(f"  skip {preset}: no records on disk")
            continue
        drew = True
        lags = np.arange(max_lag + 1) * every
        for i, (key, colour, lab) in enumerate(
                (("eddy", PRESET_COLOUR[preset], "eddy amplitude of $h$"),
                 ("umax", MUTED, "peak wind"))):
            m = np.mean(curves[key], axis=0)
            ax.plot(lags, m, color=colour, lw=1.6, label=lab)
            tv = float(np.mean(taus[key])) * every
            ax.axvline(tv, color=colour, lw=0.9, ls=":")
            # The tau values go in a fixed corner rather than beside their
            # lines: the peak wind decorrelates in a day or two, so its label
            # would sit on the axis and clip.
            ax.text(0.97, 0.93 - 0.09 * i, f"$\\tau$ = {tv:.0f} d",
                    transform=ax.transAxes, ha="right", va="top",
                    color=colour, fontsize=8.5)
        ax.axhline(0.0, color=RULE, lw=1.0)
        ax.set_title(PRESET_LABEL[preset])
        ax.set_xlabel("lag (days)")
        ax.set_xlim(0, lags[-1])
        tidy(ax)
    if not drew:
        plt.close(fig)
        return
    axes[0].set_ylabel("autocorrelation")
    # Below the panels, so it cannot sit on the tau values in the corner.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig_autocorrelation")


# ----------------------------------------------------------------------
# Figure 3: how fast a small error grows
# ----------------------------------------------------------------------
#: The amplitude the assimilation experiment uses. Panel (a) is drawn at this
#: one; panel (b) shows all three, which is how saturation becomes visible.
SCALE = 0.05


def fig_predictability():
    print("fig_predictability")
    d = read("exp12_predictability.csv")
    if d is None:
        return
    d = d[d.completed == 1]
    if d.empty:
        print("  skip: nothing completed")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.3))

    s = d[np.isclose(d.scale, SCALE)]
    for preset, grp in s.groupby("preset"):
        m = grp.groupby("lead_h")[["growth_h", "growth_wind"]].mean()
        c = PRESET_COLOUR.get(preset, NAVY)
        a.plot(m.index / 24.0, m.growth_h, "-", color=c, lw=1.6,
               label=PRESET_LABEL.get(preset, preset))
        a.plot(m.index / 24.0, m.growth_wind, "--", color=c, lw=1.3)
    a.axhline(2.0, color=MUTED, lw=1.0, ls=":")
    a.text(0.98, 2.0, "doubled ", transform=a.get_yaxis_transform(),
           ha="right", va="bottom", color=MUTED, fontsize=8)
    a.set_yscale("log"); plain_log(a)
    lead_axis(a, sorted(s.lead_h.unique() / 24.0))
    a.set_xlabel("lead time (days)")
    a.set_ylabel("separation / initial separation")
    a.set_title(f"solid: depth,  dashed: wind   (amplitude {SCALE})")
    a.legend(fontsize=8, loc="upper left")
    tidy(a)

    # Saturation. The two small amplitudes lie on top of each other, which is
    # what "linear regime" looks like; the large one peels away.
    tj = d[d.preset == "two_jets"]
    if not tj.empty:
        # Amplitude is an ordered variable, so it gets one hue at three
        # lightnesses rather than three hues, which would read as three
        # unrelated things.
        shades = ["#E0B37A", "#B8731A", "#6B420D"]
        for (sc, grp), col in zip(tj.groupby("scale"), shades):
            m = grp.groupby("lead_h").growth_h.mean()
            lw = 2.0 if np.isclose(sc, SCALE) else 1.3
            b.plot(m.index / 24.0, m.values, "-o", ms=3, lw=lw, color=col,
                   label=f"amplitude {sc:g}")
        b.set_yscale("log"); plain_log(b)
        lead_axis(b, sorted(tj.lead_h.unique() / 24.0))
        b.set_xlabel("lead time (days)")
        b.set_ylabel("separation / initial separation")
        b.set_title("two jets, three initial amplitudes")
        b.legend(fontsize=8, loc="upper left")
        tidy(b)
    fig.tight_layout()
    save(fig, "fig_predictability")


# ----------------------------------------------------------------------
# Figure 4: the three variables of one state
# ----------------------------------------------------------------------
def fig_multivariate(preset="two_jets"):
    print("fig_multivariate")
    got = load(f"regime_{preset}_f0")
    if got is None:
        print(f"  skip: no regime_{preset}_f0 snapshot")
        return
    (u, v, h), meta = got
    k = meta.get("n_pole_rows", 0)
    sl = slice(k, h.shape[0] - k) if k else slice(None)

    anom = h - h.mean(axis=1, keepdims=True)
    speed = np.hypot(u, v)
    zeta, zunit = si_scale(vorticity(u, v, meta))

    panels = [
        (anom, DIVERGING, sym_norm(anom[sl], 99.0),
         "depth anomaly", "$h - \\overline{h}^{\\,\\lambda}$ (m)"),
        (speed, SEQUENTIAL,
         pos_norm(speed[sl], 99.0),
         "wind speed", "$|U|$ (m s$^{-1}$)"),
        (zeta, DIVERGING, sym_norm(zeta[sl], 99.0),
         "relative vorticity", f"$\\zeta$ ({zunit})"),
    ]
    fig = plt.figure(figsize=(4.3 * len(panels), 4.6))
    for j, (F, cmap, norm, title, label) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), j + 1, projection="3d")
        globe(ax, F, cmap, norm, mask_rows=k)
        ax.set_title(title, fontsize=11, pad=-4)
        colorbar(fig, ax, cmap, norm, label)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.04, wspace=0.0)
    save(fig, "fig_multivariate")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_assimilation, fig_autocorrelation, fig_predictability,
              fig_multivariate):
        try:
            f()
        except Exception as exc:                      # one bad panel is not
            print(f"  {f.__name__} failed: {exc!r}")  # a reason to lose the rest
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
