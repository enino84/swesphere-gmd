# -*- coding: utf-8 -*-
"""The field figures, drawn from the snapshots EXP-11 saved.

Nothing here integrates anything. Every panel reads a ``.npy`` written by
``exp11_gallery.py``, so restyling a figure costs seconds and the twentieth
revision is as cheap as the first. A figure whose snapshot is missing is
skipped with a message, so this can be run while EXP-11 is still going.

On the colour maps, because a reviewer of a geophysical model will look at
them before they read anything. Signed fields (vorticity, the departure of
the depth from its mean) get a two-hue diverging map through a neutral grey,
so that zero is unmistakable and the sign is readable by someone with any of
the common colour vision deficiencies; blue against amber is the safest such
pair. Unsigned fields (the magnitude of an error, a wind speed) get a single
hue from light to dark, which encodes magnitude by lightness and therefore
survives being printed in grey. No panel uses a rainbow: a rainbow invents
boundaries where the field is smooth and hides them where it is not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))
SNAPS = RESULTS / "snapshots"
FIG = Path(os.environ.get("FIGURES_DIR", "figures"))

# Ink and rules, matching make_figures.py.
NAVY, MUTED, RULE = "#0F2B4C", "#5B6F84", "#D6E3EE"

# Categorical, for the line panels: validated for colour vision deficiency
# and for contrast against a white surface.
BLUE, AMBER, GREEN = "#1F5FA8", "#B8731A", "#0F9070"

#: Signed fields. Two hues through a neutral grey; grey, not white, so the
#: zero contour does not disappear into the page.
DIVERGING = colors.LinearSegmentedColormap.from_list(
    "blue_amber", ["#0B3C78", "#2E7FC0", "#9FC4E4", "#DCDCDC",
                   "#EBC58B", "#D08B22", "#7A4A10"])

#: Unsigned magnitudes. One hue, light to dark, monotone in lightness.
SEQUENTIAL = colors.LinearSegmentedColormap.from_list(
    "navy_ramp", ["#F4F8FC", "#BBD6EC", "#6FA8D6", "#2E6FB0", "#0F2B4C"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": NAVY,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "text.color": NAVY, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load(name):
    """``(fields, meta)`` for one snapshot, or ``None`` if it is not there."""
    p, j = SNAPS / f"{name}.npy", SNAPS / f"{name}.json"
    if not p.exists():
        return None
    meta = json.loads(j.read_text()) if j.exists() else {}
    x = np.load(p).astype(float)
    nlat, nlon = meta.get("Nlat"), meta.get("Nlon")
    if nlat is None:
        nlat = int(round((x.size / 3 / 2) ** 0.5))
        nlon = 2 * nlat
    fs = nlat * nlon
    return (x[:fs].reshape(nlat, nlon),
            x[fs:2 * fs].reshape(nlat, nlon),
            x[2 * fs:].reshape(nlat, nlon)), meta


def need(*names):
    out = [load(n) for n in names]
    missing = [n for n, o in zip(names, out) if o is None]
    if missing:
        print(f"  skip: missing {', '.join(missing)}")
        return None
    return out


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        # dpi applies to the rasterized elements inside the PDF too, so 300
        # keeps the spheres sharp in print without the vector blow-up.
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}")


# ----------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------
def vorticity(u, v, meta):
    """Relative vorticity from the saved winds, on the grid they were saved on."""
    nlat, nlon = u.shape
    a = meta.get("a", 6.371e6)
    lat = np.radians(90 - np.arange(nlat) * 180 / nlat)[:, None]
    cos = np.cos(lat)
    cos_safe = np.where(np.abs(cos) < 0.08, np.sign(cos + 1e-10) * 0.08, cos)
    k = np.fft.rfftfreq(nlon, d=1.0 / nlon)
    dv_dlon = np.fft.irfft(1j * k * np.fft.rfft(v, axis=1), n=nlon, axis=1)
    dlat = np.radians(180 / nlat)
    du = np.zeros_like(u)
    du[1:-1] = (u[2:] - u[:-2]) / (2 * dlat)
    du[0] = (u[1] - u[0]) / dlat
    du[-1] = (u[-1] - u[-2]) / dlat
    du = -du
    return dv_dlon / (a * cos_safe) - du / a + np.sin(lat) / cos_safe * u / a


def upsample(F, factor=3):
    """Smooth a field for drawing, by zero-padding its spherical harmonics.

    Only for the picture: the grid a model runs on is coarse enough that a
    sphere drawn from it shows facets, and a reader should be looking at the
    flow rather than at the mesh. Nothing is computed from the result.
    """
    try:
        import pyshtools as pysh
    except ImportError:
        return F
    nlat = F.shape[0]
    L = nlat // 2 - 1
    c = pysh.expand.SHExpandDH(F, norm=1, sampling=2, lmax_calc=L)
    LL = factor * (L + 1) - 1
    cc = np.zeros((2, LL + 1, LL + 1))
    cc[:, :L + 1, :L + 1] = c
    return pysh.expand.MakeGridDH(cc, norm=1, sampling=2, lmax=LL)


# ----------------------------------------------------------------------
# The globe
# ----------------------------------------------------------------------
def globe(ax, F, cmap, norm, mask_rows=0, relief=0.0,
          elev=24.0, azim=-58.0, up=3):
    """Draw a scalar field on a shaded sphere.

    The shading is a plain Lambert term with a small specular highlight and a
    rim light. It is there to make the sphere read as a sphere: without it a
    filled contour on a circle is ambiguous about which way the surface
    turns, and the eye cannot tell a pole from a limb.
    """
    Ff = upsample(np.asarray(F, dtype=float), up)
    nlat, nlon = Ff.shape
    kk = int(round(mask_rows * up))
    lat = np.radians(90 - np.arange(nlat) * 180 / nlat)
    lon = np.radians(np.arange(nlon + 1) * 360 / nlon)
    LON, LAT = np.meshgrid(lon, lat)
    Fw = np.concatenate([Ff, Ff[:, :1]], axis=1)

    scaled = np.clip((Fw - np.nanmean(Fw)) / max(norm.vmax - norm.vmin, 1e-9),
                     -1.5, 1.5)
    if kk:
        scaled[:kk] = scaled[nlat - kk:] = 0.0
    r = 1.0 + relief * scaled
    X = r * np.cos(LAT) * np.cos(LON)
    Y = r * np.cos(LAT) * np.sin(LON)
    Z = r * np.sin(LAT)

    C = matplotlib.colormaps[cmap](norm(Fw)) if isinstance(cmap, str) \
        else cmap(norm(Fw))
    C = np.array(C)
    if kk:
        C[:kk] = C[nlat - kk:] = (0.80, 0.80, 0.82, 1.0)

    light = np.array([-0.55, -0.62, 0.56]); light /= np.linalg.norm(light)
    el, az = np.radians(elev), np.radians(azim)
    eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                    np.sin(el)])
    nrm = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON),
                    np.sin(LAT)], axis=-1)
    lam = np.clip((nrm * light).sum(-1), 0.0, 1.0)
    half = light + eye; half /= np.linalg.norm(half)
    spec = np.clip((nrm * half).sum(-1), 0.0, 1.0) ** 28
    rim = np.clip(1.0 - (nrm * eye).sum(-1), 0.0, 1.0) ** 3
    C[..., :3] *= (0.52 + 0.50 * lam)[..., None]
    C[..., :3] = np.clip(C[..., :3] + 0.14 * spec[..., None]
                         + 0.07 * rim[..., None], 0.0, 1.0)

    # antialiased=False is not a quality setting here, it is a correctness
    # one: with it on, matplotlib feathers the edge of every quad and the
    # seams add up to a visible mesh laid over the field, which a reader
    # reasonably mistakes for structure in the flow. The upsampling above is
    # what makes the surface smooth; the quads must be drawn hard-edged.
    # rasterized, and this is not cosmetic. A sphere at this upsampling is
    # a few hundred thousand quads, and a vector backend writes every one of
    # them as a polygon: the five figures came to 190 MB of PDF, which no
    # journal will take. Rasterizing the surface alone embeds it as an image
    # while the titles, axes and colour bars stay vector and sharp.
    ax.plot_surface(X, Y, Z, facecolors=C, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False, rasterized=True)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    lim = 0.72
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)


def colorbar(fig, ax, cmap, norm, label):
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.036,
                      pad=-0.03, aspect=20, shrink=0.74)
    cb.set_label(label, fontsize=9); cb.ax.tick_params(labelsize=8)
    cb.outline.set_edgecolor(RULE)
    return cb


def si_scale(F, unit="s$^{-1}$"):
    """``(scaled field, axis label)`` with the power of ten folded into the label.

    Vorticity is around 1e-5 s-1, and left alone matplotlib parks a lone
    "1e-5" in the corner of the colour bar, where it collides with the label
    and is easy to read as part of the neighbouring number. Putting the
    exponent in the label instead costs nothing and removes the ambiguity.
    """
    m = float(np.nanmax(np.abs(F)))
    if not np.isfinite(m) or m == 0:
        return F, unit
    p = int(np.floor(np.log10(m)))
    if -2 <= p <= 2:
        return F, unit
    return F / 10.0 ** p, f"10$^{{{p}}}$ {unit}"


def flatten_contours(cs):
    """Remove the hairlines a filled contour shows between its own levels.

    A ``contourf`` draws each band as a separate polygon, and in a vector
    backend the seams between them render as faint lines across the field.
    A reader sees contours that are not in the data.
    """
    for c in getattr(cs, "collections", [cs]):
        try:
            c.set_edgecolor("face")
            c.set_linewidth(0)
        except AttributeError:
            pass


def sym_norm(F, q=99.0):
    v = float(np.nanpercentile(np.abs(F), q))
    return colors.Normalize(-v, v)


def pos_norm(F, q=99.5):
    return colors.Normalize(0.0, float(np.nanpercentile(F, q)))


# ----------------------------------------------------------------------
# Figure 1: the rotated test case. The one that carries the argument.
# ----------------------------------------------------------------------
def fig_rotated_tc2():
    print("fig_rotated_tc2")
    angles = []
    for a in (0, 45, 90):
        if load(f"tc2_a{a:02d}_rows10_exact") is not None:
            angles.append(a)
    if not angles:
        print("  skip: no tc2 snapshots")
        return

    data = []
    for a in angles:
        (u0, v0, h0), meta = load(f"tc2_a{a:02d}_rows10_exact")
        (u1, v1, h1), meta_final = load(f"tc2_a{a:02d}_rows10_final")
        k = meta.get("n_pole_rows", 10)
        sl = slice(k, h0.shape[0] - k) if k else slice(None)
        rel = float(np.sqrt(np.mean((h1 - h0)[sl] ** 2))
                    / np.sqrt(np.mean(h0[sl] ** 2)))
        # Days from the final snapshot, not the initial one: the initial one
        # is at day zero by definition, and reading the interval off it
        # labelled every panel in this figure "after 0 days".
        data.append(dict(a=a, anom=h0 - h0.mean(), err=np.abs(h1 - h0), k=k,
                         rel=rel, days=meta_final.get("days", 5.0)))

    # ONE scale per row, across all the angles. These panels exist to be
    # compared with each other, and a per-panel scale would defeat that
    # twice over: the exact solutions are the same field rotated, so three
    # different scales would be meaningless, and in the error row the whole
    # finding is that the rotated cases are an order of magnitude worse. On
    # a shared scale the grid-aligned panel comes out nearly blank, which is
    # not a rendering failure but the result; its own l2 is in its title so
    # the magnitude stays readable.
    n_top = sym_norm(np.concatenate([d["anom"].ravel() for d in data]))
    n_bot = pos_norm(np.concatenate([d["err"].ravel() for d in data]))

    fig = plt.figure(figsize=(4.3 * len(angles), 8.6))
    for j, d in enumerate(data):
        ax = fig.add_subplot(2, len(angles), j + 1, projection="3d")
        globe(ax, d["anom"], DIVERGING, n_top, mask_rows=0, relief=0.035)
        ax.set_title(f"exact solution, $\\alpha$ = {d['a']}$^\\circ$",
                     fontsize=11, pad=-10)

        ax = fig.add_subplot(2, len(angles), len(angles) + j + 1,
                             projection="3d")
        globe(ax, d["err"], SEQUENTIAL, n_bot, mask_rows=d["k"])
        dd = f"{d['days']:g}"
        ax.set_title(f"error after {dd} day{'' if d['days'] == 1 else 's'}, "
                     f"$l_2$ = {d['rel']:.1e}", fontsize=11, pad=-10)

    # The bars are placed in figure coordinates rather than attached to the
    # panels. A colour bar attached to a list of 3-D axes steals its room
    # from their bounding boxes, which for a sphere are mostly empty space,
    # and the bar ends up drawn across the globes.
    # Both bars on one line at the foot of the figure. A bar between the rows
    # runs into the lower row's titles: a 3-D axes has a bounding box far
    # larger than the sphere drawn inside it, so the gap between the rows is
    # not where it appears to be.
    fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.10,
                        wspace=0.0, hspace=0.16)
    for x, cmap, n, lab in ((0.10, DIVERGING, n_top,
                             "top row: $h$ minus its mean (m)"),
                            (0.57, SEQUENTIAL, n_bot,
                             "bottom row: $|h-h_{\\rm exact}|$ (m)")):
        cax = fig.add_axes([x, 0.055, 0.33, 0.014])
        cb = fig.colorbar(cm.ScalarMappable(norm=n, cmap=cmap), cax=cax,
                          orientation="horizontal")
        cb.set_label(lab + ", one scale across the row", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_edgecolor(RULE)
    save(fig, "fig_rotated_tc2")


# ----------------------------------------------------------------------
# Figure 2: the Galewsky instability, the frame the literature compares
# ----------------------------------------------------------------------
def fig_galewsky():
    print("fig_galewsky")
    got = need("galewsky_day6") or need("galewsky_day1")
    if got is None:
        return
    (u, v, h), meta = got[0]
    day = int(meta.get("days", 6))
    z = vorticity(u, v, meta)
    nlat = z.shape[0]
    lat = 90 - np.arange(nlat) * 180 / nlat
    lon = np.arange(z.shape[1]) * 360 / z.shape[1]

    # The published reference is shown as a strip over the unstable band, so
    # this is drawn the same way and can be laid beside it.
    band = (lat <= 80) & (lat >= 10)
    fig, (ax, axg) = plt.subplots(
        1, 2, figsize=(12.6, 3.6), width_ratios=[2.1, 1.0],
        subplot_kw=None)
    axg.remove()
    axg = fig.add_subplot(1, 2, 2, projection="3d")

    zs, unit = si_scale(z)
    n = sym_norm(zs[band], 99.5)
    cs = ax.contourf(lon, lat[band], zs[band],
                     levels=np.linspace(n.vmin, n.vmax, 25),
                     cmap=DIVERGING, norm=n, extend="both")
    cs.set_rasterized(True)
    flatten_contours(cs)
    # No zero contour. Outside the jet the field is near zero and its zero
    # line is the contour of rounding, which draws wandering curves through
    # the quiet half of the panel that a reader takes for structure. The
    # neutral grey midpoint of the colour map already marks zero.
    ax.set_xlabel("longitude (degrees)"); ax.set_ylabel("latitude (degrees)")
    ax.set_title(f"relative vorticity, day {day}, "
                 f"$L$ = {meta.get('LMAX', '?')}", fontsize=11)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    cb = fig.colorbar(cm.ScalarMappable(norm=n, cmap=DIVERGING), ax=ax,
                      fraction=0.035, pad=0.015)
    cb.set_label(f"$\\zeta$ ({unit})", fontsize=9)
    cb.ax.tick_params(labelsize=8); cb.outline.set_edgecolor(RULE)

    globe(axg, zs, DIVERGING, sym_norm(zs, 99.5), mask_rows=0, elev=42.0)
    axg.set_title("the same field on the sphere", fontsize=11, pad=-6)

    fig.subplots_adjust(wspace=0.02)
    save(fig, "fig_galewsky")


# ----------------------------------------------------------------------
# Figure 3: what the three regimes look like
# ----------------------------------------------------------------------
def fig_regimes():
    print("fig_regimes")
    names = [n for n in ("waves", "one_jet", "two_jets")
             if load(f"regime_{n}_f0") is not None]
    if not names:
        print("  skip: no regime snapshots")
        return
    fig = plt.figure(figsize=(4.3 * len(names), 4.6))
    for j, name in enumerate(names):
        (u, v, h), meta = load(f"regime_{name}_f0")
        k = meta.get("n_pole_rows", 10)
        z, unit = si_scale(vorticity(u, v, meta))
        sl = slice(k, z.shape[0] - k) if k else slice(None)
        n = sym_norm(z[sl], 99.0)
        ax = fig.add_subplot(1, len(names), j + 1, projection="3d")
        globe(ax, z, DIVERGING, n, mask_rows=k)
        # The peak in the title, because the three panels are on three
        # scales. They have to be: waves is an order of magnitude weaker
        # than the forced regimes, and a shared scale would render it
        # blank. But three panels of similar colour at different scales
        # invite the reader to compare amplitudes they cannot see, so the
        # number is stated rather than left to the colour bar.
        ax.set_title(f"{name.replace('_', ' ')}\n"
                     f"$|\\zeta|_{{\\max}}$ = {np.nanmax(np.abs(z[sl])):.2g}"
                     f" {unit}", fontsize=11, pad=-4)
        colorbar(fig, ax, DIVERGING, n, f"$\\zeta$ ({unit})")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.04, wspace=0.0)
    save(fig, "fig_regimes")


# ----------------------------------------------------------------------
# Figure 4: the pole, with the sponge and without it
# ----------------------------------------------------------------------
def fig_sponge_poles():
    print("fig_sponge_poles")
    got = need("sponge_rows10", "sponge_rows0")
    if got is None:
        return
    raw = [vorticity(f[0], f[1], m) for f, m in got]
    _, unit = si_scale(raw[0])
    sc = 1.0 if unit.startswith("s") else 10.0 ** int(
        unit.split("{")[1].split("}")[0])
    fields = [z / sc for z in raw]

    # ONE colour scale for both panels. The figure exists to be compared, and
    # two panels scaled independently cannot be: the run with more polar noise
    # would get a wider scale and come out looking calmer than the quiet one,
    # which is the exact opposite of what the figure is for. The scale is set
    # on the midlatitudes of the default run, where both are well behaved, so
    # anything that saturates at the pole is saturating against a yardstick
    # taken from the part of the flow we trust.
    mid = fields[0].shape[0] // 2
    n = sym_norm(fields[0][mid - 8:mid + 8], 99.0)
    days = got[0][1].get("days", 30)

    fig = plt.figure(figsize=(9.0, 5.4))
    for j, (rows, z) in enumerate(zip((10, 0), fields)):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        # Straight down on the north pole, so the polar cap is the picture
        # rather than a sliver at the limb.
        globe(ax, z, DIVERGING, n, mask_rows=0, elev=74.0, azim=-60.0)
        pk = float(np.nanmax(np.abs(z[:max(rows, 6)]))) / max(n.vmax, 1e-30)
        ax.set_title(("default sponge, 10 rows" if rows else "no sponge")
                     + f"\npolar peak {pk:.1f}$\\times$ the midlatitude scale",
                     fontsize=11, pad=-2)

    cb = fig.colorbar(cm.ScalarMappable(norm=n, cmap=DIVERGING),
                      ax=fig.axes, orientation="horizontal", fraction=0.04,
                      pad=0.02, aspect=34, shrink=0.55)
    cb.set_label(f"$\\zeta$ ({unit}), one scale for both panels, "
                 f"set on the midlatitudes", fontsize=9)
    cb.ax.tick_params(labelsize=8); cb.outline.set_edgecolor(RULE)

    fig.suptitle(f"the north pole after {days:g} day{'' if days == 1 else 's'}"
                 f" of two_jets", fontsize=12, y=1.02)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.10, wspace=0.0)
    save(fig, "fig_sponge_poles")


# ----------------------------------------------------------------------
# Figure 5: the Rossby-Haurwitz wave travelling
# ----------------------------------------------------------------------
def fig_rh_propagation():
    print("fig_rh_propagation")
    days, rows, meta = [], [], None
    for d in range(0, 15):
        got = load(f"rh_day{d}")
        if got is None:
            continue
        (u, v, h), meta = got
        nlat = h.shape[0]
        lat = 90 - np.arange(nlat) * 180 / nlat
        j = int(np.argmin(np.abs(lat - 30.0)))     # the wave's own latitude
        rows.append(h[j] - h[j].mean()); days.append(d)
    if len(days) < 3:
        print("  skip: need at least three rh snapshots")
        return
    M = np.array(rows)
    lon = np.arange(M.shape[1]) * 360 / M.shape[1]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                  width_ratios=[1.35, 1.0])
    n = sym_norm(M, 99.5)
    # gouraud, not nearest: with one row per day, flat shading draws thick
    # horizontal bands and the eye reads them as a standing pattern. The
    # wave moves 11 degrees a day against a 90-degree wavelength, so the
    # tilt is real but shallow, and blocky rows hide exactly the thing the
    # panel exists to show.
    ax.pcolormesh(lon, days, M, cmap=DIVERGING, norm=n, shading="gouraud",
                  rasterized=True)

    # Both phase lines, the measured one and the analytic one, because the
    # gap between them is the result and drawing only the theory would hide
    # it. The analytic speed is the NONDIVERGENT BAROTROPIC one, and the
    # shallow-water system is divergent, so part of any gap is physical
    # rather than numerical: EXP-02 runs three resolutions, and whether the
    # ratio converges to a constant or keeps moving is what separates the
    # two. Measured here from the complex phase of zonal mode R, which a
    # pattern cos(R(lam - lam0)) carries as -R*lam0, hence the sign.
    Om = meta.get("Omega", 7.292e-5); R = 4
    om = 7.848e-6
    nu = (R * (3.0 + R) * om - 2.0 * Om) / ((1.0 + R) * (2.0 + R))
    analytic = np.degrees(nu * 86400.0)
    t = np.array(days, dtype=float)

    phase = np.unwrap([np.angle(np.fft.rfft(r)[R]) for r in M])
    lam_meas = np.degrees(-phase / R)
    measured = float(np.polyfit(t, lam_meas, 1)[0]) if len(t) > 2 else analytic

    lam0 = lon[int(np.argmax(M[0]))]
    for sp, c, ls, lab in ((analytic, AMBER, "--",
                            f"barotropic limit, {analytic:.1f}$^\\circ$/day"),
                           (measured, GREEN, "-",
                            f"measured, {measured:.1f}$^\\circ$/day")):
        x = (lam0 + sp * t) % 360.0
        x[np.r_[False, np.abs(np.diff(x)) > 180.0]] = np.nan   # no wrap line
        ax.plot(x, t, lw=2.0, color=c, ls=ls, label=lab)
    ax.set_xlabel("longitude (degrees)"); ax.set_ylabel("day")
    ax.set_title("depth anomaly at 30$^\\circ$N", fontsize=11)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              fontsize=8)
    cb = fig.colorbar(cm.ScalarMappable(norm=n, cmap=DIVERGING), ax=ax,
                      fraction=0.04, pad=0.02)
    cb.set_label("$h'$ (m)", fontsize=9); cb.outline.set_edgecolor(RULE)

    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    (u, v, h), m2 = load(f"rh_day{days[-1]}")
    anom = h - h.mean()
    globe(ax2, anom, DIVERGING, sym_norm(anom), mask_rows=0, relief=0.06)
    ax2.set_title(f"day {days[-1]}", fontsize=11, pad=-6)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig_rh_propagation")


def main():
    if not SNAPS.exists():
        print(f"no snapshots in {SNAPS}; run exp11_gallery.py first")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_rotated_tc2, fig_galewsky, fig_regimes, fig_sponge_poles,
              fig_rh_propagation):
        try:
            f()
        except Exception as exc:                      # one bad panel is not
            print(f"  {f.__name__} failed: {exc!r}")  # a reason to lose the rest
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
