# -*- coding: utf-8 -*-
"""EXP-13. Is the first-order polar end difference the cause of the plateau?

The alpha = 90 sweep without a sponge stops improving around 4e-3 whatever
the resolution (Sect. 5.1). Two candidates were tested and rejected
elsewhere: the floor on |cos(phi)| and the filter protocol. The third, and
the one we expected to be the answer, is the pair of one-sided, FIRST-order
meridional differences at the rows adjacent to each pole. A first-order term
inside a second-order scheme is the usual explanation for a rate that is not
two.

This repeats the sweep with those two rows replaced by the second-order
one-sided stencil and changes nothing else.

Why it monkey-patches rather than adding an option to the package
-----------------------------------------------------------------
``swesphere.dynamics`` does ``from .grid import d_lat``, so the name it calls
is its own module-level binding and that is what is rebound here. The
alternative would be to ship the alternative stencil as a model option, but
an option exists to be used and this one should not be: the result below is
that it does not fix what it was meant to fix. Putting the experiment in the
experiment repository, where it is archived and reproducible but not part of
the published API, is the honest place for a change that was tried and
rejected.

The first block of the output reproduces ``exp01_convergence.csv`` with the
shipped stencil. That is not ceremony. A comparison between a patched run
and a published number is only a comparison if the unpatched run reproduces
the published number first, and the agreement here is better than one
percent.

Takes about 20 minutes, almost all of it at L = 96.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

import swesphere.dynamics as dyn
from swesphere import SWEModel
from swesphere.grid import SphereGrid

from common import (DAY, QUICK, TC2_H0, TC2_U0, WILLIAMSON, Table, interior,
                    l2_error, log, with_coriolis)
from testcases import tc2_coriolis, williamson_tc2

warnings.filterwarnings("ignore")

RESOLUTIONS = [16, 32] if QUICK else [16, 32, 64, 96]
ALPHA = 90.0
DAYS = 0.5 if QUICK else 5.0
SCORE_LATITUDE = 62.7

#: The published values this must reproduce before the comparison means
#: anything. From exp01_convergence.csv, alpha = 90, no sponge, 5 days.
#:
#: Under QUICK the runs are half a day rather than five, so
#: ``ratio_to_baseline`` compares two different integrations and is
#: meaningless. Only the ratio BETWEEN the two stencils is meaningful there,
#: and QUICK exists to check that the code runs, not to measure anything.
BASELINE = {16: 2.88e-2, 32: 9.33e-3, 64: 4.68e-3, 96: 4.04e-3}


def d_lat_first_order(q: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """What the package ships: centred inside, first order at both ends."""
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2 * grid.dlat)
    dq[0] = (q[1] - q[0]) / grid.dlat
    dq[-1] = (q[-1] - q[-2]) / grid.dlat
    return -dq


def d_lat_second_order(q: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Centred inside, second-order one-sided at both ends."""
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2 * grid.dlat)
    dq[0] = (-3.0 * q[0] + 4.0 * q[1] - q[2]) / (2 * grid.dlat)
    dq[-1] = (3.0 * q[-1] - 4.0 * q[-2] + q[-3]) / (2 * grid.dlat)
    return -dq


STENCILS = (("first_order", d_lat_first_order),
            ("second_order", d_lat_second_order))


def run() -> None:
    table = Table(
        "exp13_polar_stencil",
        fields=["stencil", "LMAX", "Nlat", "alpha_deg", "days", "dt",
                "l2_h", "baseline_l2_h", "ratio_to_baseline", "seconds"],
        key_fields=["stencil", "LMAX"],
    )
    original = dyn.d_lat
    try:
        for name, stencil in STENCILS:
            dyn.d_lat = stencil
            for L in RESOLUTIONS:
                if table.has(stencil=name, LMAX=L):
                    log(f"skip {name} LMAX={L}")
                    continue
                t0 = time.time()
                dt = 120.0 * (32.0 / L)
                n_lat = 2 * (L + 1)
                dlat_deg = 180.0 / n_lat
                log(f"{name} LMAX={L} dt={dt:.0f}s")

                model = SWEModel(LMAX=L, dt=dt, n_pole_rows=0, **WILLIAMSON)
                model.grid = with_coriolis(model.grid,
                                           tc2_coriolis(model.grid, ALPHA))
                u0, v0, h0 = williamson_tc2(model.grid, U0=TC2_U0,
                                            H0=TC2_H0, alpha_deg=ALPHA)
                x0 = model.pack(u0, v0, h0)
                n_steps = int(round(DAYS * DAY / dt))
                x = model.propagate(x0, [0.0, n_steps * dt])
                _, _, h = model.unpack(x)

                k = max(1, int(round((90.0 - SCORE_LATITUDE) / dlat_deg)))
                err = l2_error(interior(h, k), interior(h0, k))
                base = BASELINE.get(L, float("nan"))
                table.add(
                    stencil=name, LMAX=L, Nlat=model.Nlat, alpha_deg=ALPHA,
                    days=DAYS, dt=dt, l2_h=err, baseline_l2_h=base,
                    ratio_to_baseline=round(err / base, 4),
                    seconds=round(time.time() - t0, 1))
                log(f"  l2={err:.3e}  published={base:.3e}  "
                    f"ratio={err / base:.3f}")
    finally:
        dyn.d_lat = original
    table.close()
    log("EXP-13 done ->", table.path)


if __name__ == "__main__":
    run()
