# -*- coding: utf-8 -*-
"""EXP-07. Separating the integrator from the dissipative model.

Reviewer 2's third objection: "comparisons that change the time step must
distinguish integration errors from changes in the dissipative model itself.
The claim that different integrators are assessed on the same terms therefore
needs a clearly specified comparison protocol and appropriate limits."

The manuscript had found that the eddy amplitude of a regime is set by the
interval between filter applications rather than by the step, and that runs
sharing an interval still differ by 2 to 12%. That residual was attributed to
the polar sponge acting once per step. Both halves of that need to be
measured rather than asserted, because the filter and the sponge are two
different dissipations on two different schedules.

The protocol this experiment defines and measures, in three blocks:

  A. **step varies, filter interval fixed.** Changing dt alone. What is left
     is integration error plus the sponge, which still fires every step.
  B. **step varies, filter interval fixed, sponge off.** The same sweep
     without the polar sponge, which isolates how much of block A's residual
     the sponge is responsible for. This is the measurement the manuscript
     did not have.
  C. **step fixed, filter interval varies.** The dissipative model changes
     and the integrator does not, which is the dependence the manuscript
     reported.

A comparison of two integrators is fair only within a block. Reporting the
three together is what lets a reader see which effect dominates.

Note this experiment is only meaningful on swesphere >= 0.4.0: before that,
the filter schedule restarted on every ``propagate`` call, so the interval a
run actually used depended on how the driver chopped up time.
"""

from __future__ import annotations

import time

import numpy as np
from swesphere import presets

from common import DAY, QUICK, Table, interior, log

PRESETS = ["two_jets"] if QUICK else ["waves", "two_jets"]
DAYS = 0.5 if QUICK else 30.0
LMAX = 16 if QUICK else 32
FIXED_INTERVAL = 360.0
STEPS_A = [120.0, 180.0] if QUICK else [60.0, 90.0, 120.0, 180.0, 360.0]
INTERVALS_C = [360.0, 720.0] if QUICK else [120.0, 240.0, 360.0, 720.0,
                                            1440.0, 2880.0]
FIXED_STEP = 120.0


def eddy_amplitude(model, x: np.ndarray) -> float:
    h = model.unpack(x)[2]
    return float(interior(h - h.mean(axis=1, keepdims=True),
                          model.n_pole_rows).std())


def _one(name, dt, interval, sponge, block):
    every = max(1, int(round(interval / dt)))
    kw = dict(LMAX=LMAX, dt=dt, filter_every=every)
    if not sponge:
        kw["n_pole_rows"] = 0
    model, x0 = presets.make(name, **kw)
    n = int(round(DAYS * DAY / dt))
    t0 = time.time()
    x = model.propagate(x0, [0.0, n * dt])
    # Scored on a fixed number of rows whether or not the sponge is on, so the
    # sponge-off rows are comparable with the sponge-on ones.
    h = model.unpack(x)[2]
    amp = float(interior(h - h.mean(axis=1, keepdims=True), 10).std())
    return dict(preset=name, block=block, dt=dt, filter_every=every,
                filter_interval_s=every * dt, sponge=int(sponge), days=DAYS,
                eddy_amplitude=amp, seconds=round(time.time() - t0, 1))


def run() -> None:
    table = Table(
        "exp07_filter_protocol",
        fields=["preset", "block", "dt", "filter_every", "filter_interval_s",
                "sponge", "days", "eddy_amplitude", "seconds"],
        key_fields=["preset", "block", "dt", "filter_interval_s", "sponge"],
    )
    for name in PRESETS:
        for dt in STEPS_A:                       # A: sponge on
            if not table.has(preset=name, block="A", dt=dt,
                             filter_interval_s=FIXED_INTERVAL, sponge=1):
                log(f"A {name} dt={dt:g} interval={FIXED_INTERVAL:g} sponge=on")
                table.add(**_one(name, dt, FIXED_INTERVAL, True, "A"))
        for dt in STEPS_A:                       # B: sponge off
            if not table.has(preset=name, block="B", dt=dt,
                             filter_interval_s=FIXED_INTERVAL, sponge=0):
                log(f"B {name} dt={dt:g} interval={FIXED_INTERVAL:g} sponge=off")
                table.add(**_one(name, dt, FIXED_INTERVAL, False, "B"))
        for iv in INTERVALS_C:                   # C: interval varies
            if not table.has(preset=name, block="C", dt=FIXED_STEP,
                             filter_interval_s=iv, sponge=1):
                log(f"C {name} dt={FIXED_STEP:g} interval={iv:g}")
                table.add(**_one(name, FIXED_STEP, iv, True, "C"))
    table.close()
    log("EXP-07 done ->", table.path)


if __name__ == "__main__":
    run()
