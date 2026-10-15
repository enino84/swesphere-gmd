# -*- coding: utf-8 -*-
"""EXP-01. Does the model converge in the configuration people actually run?

Answers reviewer 2's first objection and reviewer 4's minor point 1.

The submitted manuscript showed second-order convergence on Williamson test
case 2 with the flow axis on the rotation axis and the polar sponge switched
off, and separately reported that with the sponge on the error sits at about
1e-2 at both resolutions. The reviewer's complaint was exact: convergence
measured after removing the component that dominates the operational error
does not establish the accuracy of the method people use.

This experiment measures the full grid:

  * three rotation angles. ``alpha = 0`` is the grid-aligned case of the
    manuscript; at 90 degrees the flow runs straight over both poles, so the
    solution passes through the sponge and through the converging meridians.
  * four resolutions, each a factor of two apart, so the observed order is
    read off directly.
  * sponge on and off, the same case twice, which separates the error of the
    discretization from the error of the polar treatment instead of reporting
    only the favourable half.

Test case 2 is an exact steady solution, so the state after five days should
equal the initial state, and everything else is error. Note that the test
prescribes its own Coriolis field: it tilts the planetary rotation together
with the flow. With the untilted ``f`` the tilted flow is not a solution of
anything, which ``common.with_coriolis`` and ``testcases.tc2_coriolis``
explain.
"""

from __future__ import annotations

import numpy as np
from swesphere import SWEModel, diagnostics

from common import (DAY, QUICK, TC2_H0, TC2_U0, WILLIAMSON, Table, interior,
                    l2_error, linf_error, log, with_coriolis)
from testcases import tc2_coriolis, williamson_tc2

#: Polar rows the sponge acts on. The model's own value is used for
#: scoring; this is what the model is built with.
POLE_ROWS = 10

RESOLUTIONS = [16, 32] if QUICK else [16, 32, 64, 96]
ANGLES = [0.0, 90.0] if QUICK else [0.0, 45.0, 90.0]
DAYS = 0.5 if QUICK else 5.0

#: Latitude poleward of which the sponge acts, for the ``fixed_latitude``
#: sweep. 62.7 degrees is what the shipped ``n_pole_rows = 10`` gives at the
#: default LMAX = 32, so that configuration is a point on both sweeps.
SPONGE_LATITUDE = 62.7


def run() -> None:
    table = Table(
        "exp01_convergence",
        fields=["LMAX", "Nlat", "Nlon", "alpha_deg", "sponge", "sweep",
                "pole_rows", "sponge_lat_deg", "days", "dt", "l2_h", "linf_h",
                "l2_u", "l2_v", "mass_drift", "energy_drift", "seconds"],
        key_fields=["LMAX", "alpha_deg", "sponge", "sweep"],
    )
    for L in RESOLUTIONS:
        # The step has to shrink with the grid or the time error contaminates
        # the space error; this keeps the Courant number fixed across the sweep.
        dt = 120.0 * (32.0 / L)
        n_lat = 2 * (L + 1)
        d_lat = 180.0 / n_lat
        # Two sweeps, because they answer different questions and the
        # manuscript accidentally ran a mixture of them.
        #
        #   fixed_rows     n_pole_rows = 10 at every resolution, which is what
        #                  a user gets from the defaults. But 10 rows is a
        #                  different latitude at every resolution: it reaches
        #                  37 degrees at LMAX 16 and 83 at LMAX 128. Refining
        #                  the grid therefore also retreats the sponge, so this
        #                  is an operational curve, not a convergence test.
        #   fixed_latitude the sponge covers the same band at every resolution,
        #                  so only the discretization changes. This is the one
        #                  that measures an order of convergence.
        sweeps = {
            "fixed_rows": POLE_ROWS,
            "fixed_latitude": max(1, int(round((90.0 - SPONGE_LATITUDE) / d_lat))),
        }
        for alpha in ANGLES:
            for sponge in (True, False):
                for sweep, rows in sweeps.items():
                    # Without a sponge the two sweeps are the same run.
                    if not sponge and sweep != "fixed_rows":
                        continue
                    if table.has(LMAX=L, alpha_deg=alpha, sponge=sponge,
                                 sweep=sweep):
                        log(f"skip LMAX={L} alpha={alpha} sponge={sponge} {sweep}")
                        continue
                    log(f"LMAX={L} alpha={alpha} sponge={sponge} {sweep} "
                        f"rows={rows} dt={dt:.0f}s")
                    import time as _t
                    t0 = _t.time()

                    model = SWEModel(LMAX=L, dt=dt,
                                     n_pole_rows=rows if sponge else 0,
                                     **WILLIAMSON)
                    model.grid = with_coriolis(model.grid,
                                               tc2_coriolis(model.grid, alpha))
                    u0, v0, h0 = williamson_tc2(model.grid, U0=TC2_U0,
                                                H0=TC2_H0, alpha_deg=alpha)
                    x0 = model.pack(u0, v0, h0)

                    inv0 = diagnostics.invariants(model, x0)
                    n_steps = int(round(DAYS * DAY / dt))
                    x = model.propagate(x0, [0.0, n_steps * dt])
                    inv1 = diagnostics.invariants(model, x)
                    u, v, h = model.unpack(x)

                    # Scored on the same latitude band at every resolution,
                    # whatever the sponge is doing, so the rows compare.
                    k = max(1, int(round((90.0 - SPONGE_LATITUDE) / d_lat)))
                    table.add(
                        LMAX=L, Nlat=model.Nlat, Nlon=model.Nlon,
                        alpha_deg=alpha, sponge=int(sponge), sweep=sweep,
                        pole_rows=rows if sponge else 0,
                        sponge_lat_deg=round(90.0 - rows * d_lat, 2) if sponge else 90.0,
                        days=DAYS, dt=dt,
                        l2_h=l2_error(interior(h, k), interior(h0, k)),
                        linf_h=linf_error(interior(h, k), interior(h0, k)),
                        l2_u=l2_error(interior(u, k), interior(u0, k)),
                        l2_v=float(np.sqrt(np.mean((interior(v, k)
                                                    - interior(v0, k)) ** 2))),
                        mass_drift=abs(inv1["mass"] / inv0["mass"] - 1.0),
                        energy_drift=abs(inv1["energy"] / inv0["energy"] - 1.0),
                        seconds=round(_t.time() - t0, 1),
                    )
    table.close()
    log("EXP-01 done ->", table.path)


if __name__ == "__main__":
    run()
