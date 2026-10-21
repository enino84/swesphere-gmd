# -*- coding: utf-8 -*-
"""EXP-11. The model states behind every figure in the paper.

This experiment computes nothing new. It re-runs the cases the other
experiments already cover and saves the raw ``(u, v, h)`` states, so the
figures can be drawn, redrawn and restyled without touching the physics
again.

That separation is the whole point. A figure goes through twenty revisions
before it is publishable, and a reviewer will ask for a twenty-first. If
drawing is coupled to integrating, every revision costs a day at the
resolutions that look good in print, and the honest consequence is that the
figures stop improving. Here the expensive half runs once, on the server,
and writes ``.npy``; the cheap half reads those files and runs in seconds on
a laptop.

The five sets, and what each one is for:

``tc2``
    Williamson test case 2 at three rotation angles, with the sponge and
    without it, the exact initial state and the state after five days. The
    solution is steady, so the difference between them is entirely the
    model's error, and showing it on a sphere makes the central argument of
    the paper visible in one picture: at zero degrees the error hides in the
    polar caps where nothing is happening, and at ninety it sits in the flow.

``galewsky``
    The barotropic instability, daily to day 7, at the highest resolution we
    can afford. Day 6 is the frame the literature compares, so it is the one
    frame that has to be right.

``regimes``
    Each forced preset after spin-up, plus a short sequence, for the figure
    that shows what the three regimes actually look like.

``sponge``
    ``two_jets`` run with the default sponge and with none, viewed from
    above the pole. This is the figure for the uncomfortable section: the
    sponge-free run survives, and the polar cap fills with grid-scale noise
    that a global view would hide.

``rossby_haurwitz``
    Fourteen daily frames of the R = 4 wave, for the figure that shows the
    pattern travelling east at the analytic rate.

Every file is skipped if it already exists, so an interrupted run resumes and
a figure that needs one more frame costs one more frame.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from swesphere import SWEModel, diagnostics, presets

from common import DAY, QUICK, RESULTS, Table, log, with_coriolis
from common import TC2_H0, TC2_U0, WILLIAMSON
from testcases import tc2_coriolis, williamson_tc2

#: Where the states go. Kept apart from the CSVs because these are large and
#: a user may want to fetch the tables without the arrays.
SNAPS = RESULTS / "snapshots"

# Resolutions. The Galewsky case gets the highest one it can, because it is
# the figure that is compared against a published reference and detail is the
# point of it; the others are drawn as smooth fields and gain little above 64.
#
# Each is overridable, because this is the one experiment whose cost is
# chosen rather than dictated: it produces pictures, and how good they have
# to look is a decision. GALEWSKY_LMAX=64 turns eleven minutes into one and
# the figure is still publishable; 128 is what we would submit.


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


LMAX_TC2 = _env_int("TC2_LMAX", 16 if QUICK else 64)
LMAX_GALEWSKY = _env_int("GALEWSKY_LMAX", 32 if QUICK else 128)
LMAX_REGIMES = _env_int("REGIMES_LMAX", 16 if QUICK else 64)
LMAX_SPONGE = _env_int("SPONGE_LMAX", 16 if QUICK else 64)
LMAX_RH = _env_int("RH_LMAX", 16 if QUICK else 64)

TC2_ANGLES = [0.0, 90.0] if QUICK else [0.0, 45.0, 90.0]
TC2_DAYS = 0.5 if QUICK else 5.0

GALEWSKY_DAYS = [0, 1] if QUICK else [0, 1, 2, 3, 4, 5, 6, 7]

REGIME_PRESETS = ["two_jets"] if QUICK else ["waves", "one_jet", "two_jets"]
REGIME_SPINUP = 1.0 if QUICK else 40.0
REGIME_FRAMES = 2 if QUICK else 5
REGIME_EVERY = 0.5 if QUICK else 2.0

SPONGE_ROWS = [10, 0]
SPONGE_DAYS = 1.0 if QUICK else 30.0

RH_DAYS = [0, 1] if QUICK else list(range(0, 15))


def dt_for(L: int) -> float:
    """The step that holds the Courant number at the value the defaults give."""
    return 120.0 * (32.0 / L)


def store(name: str, model, x, **meta) -> bool:
    """Save one state as float32, with the metadata a renderer needs.

    Returns True if it wrote, False if the file was already there. float32
    halves the files and is far below the precision any figure resolves;
    nothing is computed from these, they are only drawn.
    """
    SNAPS.mkdir(parents=True, exist_ok=True)
    path = SNAPS / f"{name}.npy"
    if path.exists():
        return False
    np.save(path, np.asarray(x, dtype=np.float32))
    meta = dict(meta)
    meta.update(LMAX=model.grid.LMAX, Nlat=model.Nlat, Nlon=model.Nlon,
                n_pole_rows=model.n_pole_rows, a=model.grid.a,
                Omega=model.grid.Omega, g=model.grid.g)
    (SNAPS / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return True


def have(name: str) -> bool:
    return (SNAPS / f"{name}.npy").exists()


# ----------------------------------------------------------------------
# The five sets
# ----------------------------------------------------------------------
def set_tc2(table) -> None:
    for alpha in TC2_ANGLES:
        for rows in (10, 0):
            tag = f"tc2_a{int(alpha):02d}_rows{rows}"
            if have(f"{tag}_final"):
                log(f"skip {tag}")
                continue
            log(f"{tag}: LMAX={LMAX_TC2}, {TC2_DAYS} days")
            t0 = time.time()
            model = SWEModel(LMAX=LMAX_TC2, dt=dt_for(LMAX_TC2),
                             n_pole_rows=rows, **WILLIAMSON)
            model.grid = with_coriolis(model.grid,
                                       tc2_coriolis(model.grid, alpha))
            x0 = model.pack(*williamson_tc2(model.grid, U0=TC2_U0, H0=TC2_H0,
                                            alpha_deg=alpha))
            n = int(round(TC2_DAYS * DAY / model.dt))
            x = model.propagate(x0, [0.0, n * model.dt])
            store(f"{tag}_exact", model, x0, kind="tc2", alpha_deg=alpha,
                  sponge_rows=rows, days=0.0, U0=TC2_U0, H0=TC2_H0)
            store(f"{tag}_final", model, x, kind="tc2", alpha_deg=alpha,
                  sponge_rows=rows, days=TC2_DAYS, U0=TC2_U0, H0=TC2_H0)
            table.add(set="tc2", tag=tag, LMAX=LMAX_TC2, days=TC2_DAYS,
                      frames=2, seconds=round(time.time() - t0, 1))


def set_galewsky(table) -> None:
    if all(have(f"galewsky_day{d}") for d in GALEWSKY_DAYS):
        log("skip galewsky")
        return
    log(f"galewsky: LMAX={LMAX_GALEWSKY}, to day {max(GALEWSKY_DAYS)}")
    t0 = time.time()
    model = SWEModel(LMAX=LMAX_GALEWSKY, dt=dt_for(LMAX_GALEWSKY),
                     n_pole_rows=10, **WILLIAMSON)
    x = model.initial_condition("galewsky")
    store("galewsky_day0", model, x, kind="galewsky", days=0)
    t = 0.0
    for d in range(1, max(GALEWSKY_DAYS) + 1):
        n = int(round(DAY / model.dt))
        x = model.propagate(x, [t, t + n * model.dt])
        t += n * model.dt
        if d in GALEWSKY_DAYS:
            store(f"galewsky_day{d}", model, x, kind="galewsky", days=d)
            log(f"  day {d}")
    table.add(set="galewsky", tag="galewsky", LMAX=LMAX_GALEWSKY,
              days=max(GALEWSKY_DAYS), frames=len(GALEWSKY_DAYS),
              seconds=round(time.time() - t0, 1))


def set_regimes(table) -> None:
    for name in REGIME_PRESETS:
        if all(have(f"regime_{name}_f{i}") for i in range(REGIME_FRAMES)):
            log(f"skip regime {name}")
            continue
        log(f"regime {name}: LMAX={LMAX_REGIMES}, spin-up {REGIME_SPINUP} d")
        t0 = time.time()
        model, x = presets.make(name, LMAX=LMAX_REGIMES)
        n = int(round(REGIME_SPINUP * DAY / model.dt))
        x = model.propagate(x, [0.0, n * model.dt])
        t = n * model.dt
        for i in range(REGIME_FRAMES):
            store(f"regime_{name}_f{i}", model, x, kind="regime", preset=name,
                  frame=i, days=REGIME_SPINUP + i * REGIME_EVERY)
            if i < REGIME_FRAMES - 1:
                m = int(round(REGIME_EVERY * DAY / model.dt))
                x = model.propagate(x, [t, t + m * model.dt])
                t += m * model.dt
        table.add(set="regimes", tag=f"regime_{name}", LMAX=LMAX_REGIMES,
                  days=REGIME_SPINUP + (REGIME_FRAMES - 1) * REGIME_EVERY,
                  frames=REGIME_FRAMES, seconds=round(time.time() - t0, 1))


def set_sponge(table) -> None:
    """The same flow with the sponge and without it, for the polar view.

    Both start from the same spun-up state, which is built with the default
    sponge; otherwise the two runs would differ from the first step for a
    reason that has nothing to do with the question.
    """
    if all(have(f"sponge_rows{r}") for r in SPONGE_ROWS):
        log("skip sponge")
        return
    log(f"sponge: LMAX={LMAX_SPONGE}, {SPONGE_DAYS} days, rows {SPONGE_ROWS}")
    t0 = time.time()
    base, x0 = presets.make("two_jets", LMAX=LMAX_SPONGE)
    n = int(round((1.0 if QUICK else 40.0) * DAY / base.dt))
    x0 = base.propagate(x0, [0.0, n * base.dt])
    store("sponge_start", base, x0, kind="sponge", sponge_rows=10, days=0.0)

    for rows in SPONGE_ROWS:
        if have(f"sponge_rows{rows}"):
            continue
        model, _ = presets.make("two_jets", LMAX=LMAX_SPONGE, n_pole_rows=rows)
        m = int(round(SPONGE_DAYS * DAY / model.dt))
        try:
            x = model.propagate(x0, [0.0, m * model.dt])
        except RuntimeError:
            log(f"  rows={rows} blew up; not saved")
            continue
        store(f"sponge_rows{rows}", model, x, kind="sponge",
              sponge_rows=rows, days=SPONGE_DAYS)
        log(f"  rows={rows} done")
    table.add(set="sponge", tag="sponge", LMAX=LMAX_SPONGE, days=SPONGE_DAYS,
              frames=len(SPONGE_ROWS) + 1, seconds=round(time.time() - t0, 1))


def set_rossby_haurwitz(table) -> None:
    if all(have(f"rh_day{d}") for d in RH_DAYS):
        log("skip rossby_haurwitz")
        return
    log(f"rossby_haurwitz: LMAX={LMAX_RH}, to day {max(RH_DAYS)}")
    t0 = time.time()
    model = SWEModel(LMAX=LMAX_RH, dt=dt_for(LMAX_RH), n_pole_rows=10,
                     **WILLIAMSON)
    x = model.initial_condition("rossby_haurwitz", R=4)
    store("rh_day0", model, x, kind="rossby_haurwitz", days=0, R=4)
    t = 0.0
    for d in range(1, max(RH_DAYS) + 1):
        n = int(round(DAY / model.dt))
        x = model.propagate(x, [t, t + n * model.dt])
        t += n * model.dt
        if d in RH_DAYS:
            store(f"rh_day{d}", model, x, kind="rossby_haurwitz", days=d, R=4)
    table.add(set="rossby_haurwitz", tag="rh", LMAX=LMAX_RH,
              days=max(RH_DAYS), frames=len(RH_DAYS),
              seconds=round(time.time() - t0, 1))


def run() -> None:
    table = Table(
        "exp11_gallery",
        fields=["set", "tag", "LMAX", "days", "frames", "seconds"],
        key_fields=["set", "tag"],
    )
    # Cheapest first, so a figure is drawable early even if the long ones are
    # still going.
    set_tc2(table)
    set_rossby_haurwitz(table)
    set_sponge(table)
    set_regimes(table)
    set_galewsky(table)
    table.close()
    n = len(list(SNAPS.glob("*.npy"))) if SNAPS.exists() else 0
    mb = sum(p.stat().st_size for p in SNAPS.glob("*.npy")) / 1e6 if n else 0.0
    log(f"EXP-11 done -> {n} snapshots, {mb:.0f} MB in {SNAPS}")


if __name__ == "__main__":
    run()
