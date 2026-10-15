# -*- coding: utf-8 -*-
"""EXP-03. The Galewsky instability, and what the filter costs it.

Reviewer 2 again: "At the default resolution, filtering substantially reduces
the jet's peak velocity and delays the instability. Qualitative resemblance
to the reference solution at higher resolution does not resolve this
deficiency."

That is a fair reading of what the manuscript showed, and the honest answer
is a number, not a picture. Three things are measured at each resolution:

  * **what the filter takes before anything happens**: the peak wind of the
    initial jet after a single application of the spectral filter, against
    the analytic profile. This is the damage the manuscript acknowledged
    without quantifying.
  * **when the instability arrives**: the first time the relative vorticity
    maximum exceeds a fixed multiple of its initial value. A model that
    delays the onset is not reproducing the reference, however similar the
    final picture looks.
  * **how much the answer still moves with resolution**: the same diagnostics
    at the highest resolution serve as the reference, so the reader sees
    whether the default has converged or merely looks plausible.

The perturbation is the one the test prescribes and is already in swesphere's
``galewsky`` initial condition.
"""

from __future__ import annotations

import time

import numpy as np
from swesphere import SWEModel, diagnostics
from swesphere.dynamics import relative_vorticity

from common import WILLIAMSON, DAY, QUICK, Table, interior, log

#: Polar rows the sponge acts on. The model's own value is used for
#: scoring; this is what the model is built with.
POLE_ROWS = 10

RESOLUTIONS = [32] if QUICK else [32, 64, 128]
DAYS = 0.5 if QUICK else 8.0
ONSET_FACTOR = 1.5          # vorticity max this many times its initial value


def run() -> None:
    table = Table(
        "exp03_galewsky",
        fields=["LMAX", "Nlat", "dt", "days", "umax_analytic", "umax_filtered",
                "umax_loss_pct", "onset_day", "zeta_max_final",
                "zeta_max_initial", "mass_drift", "energy_drift",
                "enstrophy_drift", "seconds"],
        key_fields=["LMAX", "days"],
    )
    for L in RESOLUTIONS:
        if table.has(LMAX=L, days=DAYS):
            log(f"skip LMAX={L}")
            continue
        dt = 120.0 * (32.0 / L)
        log(f"LMAX={L} dt={dt:.0f}s for {DAYS} days")
        t0 = time.time()

        model = SWEModel(LMAX=L, dt=dt, n_pole_rows=POLE_ROWS, **WILLIAMSON,
                         filter_every=3)
        x0 = model.initial_condition("galewsky")
        u0, v0, h0 = model.unpack(x0)
        umax_analytic = float(np.abs(interior(u0, model.n_pole_rows)).max())

        # One filter interval of integration, then read the peak wind: this
        # is what the model actually starts from, not what the test specifies.
        x_f = model.propagate(x0, [0.0, 3 * dt])
        u_f = model.unpack(x_f)[0]
        umax_filtered = float(np.abs(interior(u_f, model.n_pole_rows)).max())

        inv0 = diagnostics.invariants(model, x0)
        zeta0 = float(np.abs(interior(relative_vorticity(u0, v0, model.grid),
                                      model.n_pole_rows)).max())

        # March day by day and record when the instability takes off.
        onset = float("nan")
        x = x0
        steps_per_day = int(round(DAY / dt))
        n_days = int(round(DAYS))
        sub = max(1, steps_per_day // 8)          # sample eight times a day
        t = 0.0
        zmax = zeta0
        for _ in range(int(round(DAYS * DAY / (sub * dt)))):
            x = model.propagate(x, [t, t + sub * dt])
            t += sub * dt
            u, v, _ = model.unpack(x)
            z = float(np.abs(interior(relative_vorticity(u, v, model.grid),
                                      model.n_pole_rows)).max())
            zmax = max(zmax, z)
            if np.isnan(onset) and z > ONSET_FACTOR * zeta0:
                onset = t / DAY
        inv1 = diagnostics.invariants(model, x)

        table.add(
            LMAX=L, Nlat=model.Nlat, dt=dt, days=DAYS,
            umax_analytic=umax_analytic, umax_filtered=umax_filtered,
            umax_loss_pct=100.0 * (1.0 - umax_filtered / umax_analytic),
            onset_day=onset, zeta_max_final=zmax, zeta_max_initial=zeta0,
            mass_drift=abs(inv1["mass"] / inv0["mass"] - 1.0),
            energy_drift=abs(inv1["energy"] / inv0["energy"] - 1.0),
            enstrophy_drift=abs(inv1["enstrophy"] / inv0["enstrophy"] - 1.0),
            seconds=round(time.time() - t0, 1),
        )
    table.close()
    log("EXP-03 done ->", table.path)


if __name__ == "__main__":
    run()
