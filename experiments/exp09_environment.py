# -*- coding: utf-8 -*-
"""EXP-09. What "about two seconds per model day" actually means.

Reviewer 4's minor point 4: "The manuscript repeatedly cites approximately
two seconds per model day on one core. Please report the processor, software
environment, model settings, and timing procedure so that readers can
interpret and compare this cost estimate."

A cost figure with no machine attached is not reproducible, and the number in
the manuscript was measured on one machine with whatever BLAS threading
happened to be configured. This records the whole context next to the timing,
and states the procedure: a warm-up integration that is discarded, then
several timed repetitions of a fixed number of model days, reported as the
median rather than the mean so one scheduling hiccup does not set the number.

It also times each resolution, so a reader can scale to the configuration
they intend to run rather than extrapolating from one point.
"""

from __future__ import annotations

import os
import time

import numpy as np
from swesphere import presets

from common import DAY, QUICK, Table, describe_environment, log

RESOLUTIONS = [16] if QUICK else [16, 32, 64]
PRESETS = ["two_jets"] if QUICK else ["waves", "two_jets"]
WARMUP_DAYS = 0.25 if QUICK else 1.0
TIMED_DAYS = 0.25 if QUICK else 2.0
REPEATS = 2 if QUICK else 5


def run() -> None:
    env = describe_environment()
    log("environment:")
    for k, v in env.items():
        log(f"  {k:18s} {v}")

    env_table = Table("exp09_environment", fields=sorted(env) + ["recorded"],
                      key_fields=["recorded"])
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not env_table.has(recorded=stamp):
        env_table.add(recorded=stamp, **env)
    env_table.close()

    table = Table(
        "exp09_timing",
        fields=["preset", "LMAX", "Nlat", "Nlon", "dim", "dt", "scheme",
                "filter_every", "timed_days", "repeats", "median_s_per_day",
                "min_s_per_day", "max_s_per_day", "steps_per_second",
                "blas_threads", "cpu"],
        key_fields=["preset", "LMAX"],
    )
    for name in PRESETS:
        for L in RESOLUTIONS:
            if table.has(preset=name, LMAX=L):
                log(f"skip {name} LMAX={L}")
                continue
            model, x0 = presets.make(name, LMAX=L)
            n_warm = int(round(WARMUP_DAYS * DAY / model.dt))
            n_timed = int(round(TIMED_DAYS * DAY / model.dt))

            # Warm-up: pyshtools builds its plans on first use, and the first
            # call would otherwise be charged to the measurement.
            x = model.propagate(x0, [0.0, n_warm * model.dt])
            t = n_warm * model.dt

            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                x = model.propagate(x, [t, t + n_timed * model.dt])
                times.append((time.perf_counter() - t0) / TIMED_DAYS)
                t += n_timed * model.dt
            times = np.array(times)
            log(f"{name} LMAX={L}: median {np.median(times):.2f} s per model day")

            table.add(
                preset=name, LMAX=L, Nlat=model.Nlat, Nlon=model.Nlon,
                dim=model.dim, dt=model.dt, scheme=model.scheme,
                filter_every=model.filter_every, timed_days=TIMED_DAYS,
                repeats=REPEATS,
                median_s_per_day=float(np.median(times)),
                min_s_per_day=float(times.min()), max_s_per_day=float(times.max()),
                steps_per_second=float(n_timed / (np.median(times) * TIMED_DAYS)),
                blas_threads=env["blas_threads"], cpu=env["cpu"],
            )
    table.close()
    log("EXP-09 done ->", table.path)


if __name__ == "__main__":
    run()
