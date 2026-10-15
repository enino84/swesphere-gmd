# -*- coding: utf-8 -*-
"""EXP-06. Per-preset time-step limits, and the difference between
"it finished" and "it is right".

Reviewer 4's minor point 3: the manuscript reported that the forced presets
complete 100 days at dt = 1440 s but fail at 1800 s, while ``waves``
completes at 1800 s, and then said "the practical limit is therefore 1800 s",
which is ambiguous at best and wrong for two of the three presets.

Two numbers per preset, because they are not the same question:

  * the largest step at which the integration **completes** without the
    solution blowing up, which is what the manuscript measured;
  * the largest step at which the answer is still **accurate**, defined
    against a reference run at a small step, which is what a user needs.

A scheme can be stable and useless. Reporting only the first number invites
exactly the misreading the reviewer objected to.

The accuracy threshold is applied to the depth field after a fixed physical
time, so the comparison is between states at the same instant and not
between different numbers of steps.
"""

from __future__ import annotations

import time

import numpy as np
from swesphere import presets

from common import DAY, QUICK, Table, interior, l2_error, log

PRESETS = ["two_jets"] if QUICK else ["waves", "one_jet", "two_jets"]
STEPS = [120.0, 240.0] if QUICK else [60.0, 120.0, 240.0, 360.0, 480.0,
                                      720.0, 900.0, 1200.0, 1440.0, 1800.0,
                                      2400.0, 2880.0]
TEST_DAYS = 0.5 if QUICK else 20.0
REFERENCE_DT = 60.0
ACCURACY_THRESHOLD = 0.05        # 5% relative l2 in the interior
LMAX = 16 if QUICK else 32


def run() -> None:
    table = Table(
        "exp06_stability",
        fields=["preset", "dt", "days", "filter_every", "filter_interval_s",
                "completed", "l2_vs_reference", "accurate", "seconds"],
        key_fields=["preset", "dt"],
    )
    for name in PRESETS:
        log(f"{name}: reference run at dt={REFERENCE_DT:g}s")
        ref_model, x0 = presets.make(name, LMAX=LMAX, dt=REFERENCE_DT)
        n_ref = int(round(TEST_DAYS * DAY / REFERENCE_DT))
        x_ref = ref_model.propagate(x0, [0.0, n_ref * REFERENCE_DT])
        h_ref = interior(ref_model.unpack(x_ref)[2], ref_model.n_pole_rows)

        for dt in STEPS:
            if table.has(preset=name, dt=dt):
                log(f"skip {name} dt={dt:g}")
                continue
            t0 = time.time()
            # Keep the physical filter interval fixed while dt varies, so this
            # measures the integrator and not a different dissipative model.
            every = max(1, int(round(360.0 / dt)))
            model, x0 = presets.make(name, LMAX=LMAX, dt=dt, filter_every=every)
            n = int(round(TEST_DAYS * DAY / dt))
            completed, err = True, float("nan")
            try:
                x = model.propagate(x0, [0.0, n * dt])
                h = interior(model.unpack(x)[2], model.n_pole_rows)
                err = l2_error(h, h_ref)
            except (RuntimeError, FloatingPointError) as exc:
                completed = False
                log(f"  {name} dt={dt:g}: {type(exc).__name__}")
            log(f"  {name} dt={dt:6.0f}s  completed={completed}  "
                f"l2 vs ref={err:.3e}")
            table.add(
                preset=name, dt=dt, days=TEST_DAYS, filter_every=every,
                filter_interval_s=every * dt, completed=int(completed),
                l2_vs_reference=err,
                accurate=int(completed and np.isfinite(err)
                             and err < ACCURACY_THRESHOLD),
                seconds=round(time.time() - t0, 1),
            )
    table.close()
    log("EXP-06 done ->", table.path)


if __name__ == "__main__":
    run()
