# -*- coding: utf-8 -*-
"""EXP-12. How fast a small error grows, and what that means for a window.

This exists because of a number that surprised us while debugging EXP-08: a
1.9 m perturbation of the depth was still 1.9 m six hours later. It had been
assumed, without measurement, that a six-hour window was short relative to
the error growth of this flow. That assumption is the first thing anyone
designing a cycled experiment on this testbed will make, so it should be a
measured curve in the paper rather than a habit inherited from operational
practice on a very different model.

The measurement is deliberately filter-free. Two trajectories that differ by
a small, balanced perturbation are propagated side by side and the distance
between them is recorded against lead time. A perturbation built as a scaled
difference of two states of the EXP-04 record is dynamically consistent by
construction, so what is measured is error growth and not the radiation of
an imbalance that was put in by hand.

Three perturbation sizes are used, because growth in a nonlinear system is
only exponential while the error is small, and the point at which it stops
being so is itself worth reporting.

What it is for
--------------
Two statements in Sect. 7 rest on this experiment. A six-hour cycle sits
well inside the linear regime, so the background error contracts from cycle
to cycle. A twenty-four hour cycle does not: the error regrows between
analyses about as fast as the analysis removes it, and the system settles at
a plateau instead of converging. Neither is a defect; they are different
experiments, and a user should be able to choose between them from a figure
rather than by trial.

Needs the records from EXP-04.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from common import DAY, HOUR, QUICK, RESULTS, SHARD_COUNT, SHARD_INDEX, \
    Table, log, shard

warnings.filterwarnings("ignore")

#: Must match EXP-04, whose record this reads.
LMAX = 16 if QUICK else 32
PRESETS = ["two_jets"] if QUICK else ["waves", "one_jet", "two_jets"]
SEEDS = [0] if QUICK else [0, 1, 2]

#: Perturbation sizes, as multiples of a difference between two record
#: states. 0.05 is the size EXP-08 uses, so the middle row of this table is
#: directly comparable with that experiment.
SCALES = [0.05] if QUICK else [0.01, 0.05, 0.25]

#: Lead times at which the separation is recorded, in hours. The first two
#: bracket the window choices of EXP-08.
LEADS_H = [6.0, 24.0] if QUICK else [3.0, 6.0, 12.0, 24.0, 48.0, 96.0,
                                     192.0, 384.0]

#: Independent pairs per configuration, averaged.
PAIRS = 1 if QUICK else 4


def load_record(preset: str, seed: int):
    path = RESULTS / f"record_{preset}_seed{seed}_L{LMAX}.npy"
    if not path.exists():
        log(f"MISSING {path.name}: run exp04_regimes.py first")
        return None
    return np.load(path).astype(float)


def run() -> None:
    from pyteda.models import SWEModel

    table = Table(
        "exp12_predictability",
        fields=["preset", "seed", "LMAX", "scale", "pair", "lead_h",
                "rms_h_0", "rms_h", "growth_h", "rms_wind_0", "rms_wind",
                "growth_wind", "completed", "seconds"],
        key_fields=["preset", "seed", "scale", "pair", "lead_h"],
    )

    cases = [(preset, seed, scale)
             for preset in PRESETS for seed in SEEDS for scale in SCALES]
    mine = shard(cases)
    if SHARD_COUNT > 1:
        log(f"shard {SHARD_INDEX} of {SHARD_COUNT}: "
            f"{len(mine)} of {len(cases)} configurations")

    records: dict = {}
    for preset, seed, scale in mine:
        if table.has(preset=preset, seed=seed, scale=scale, pair=PAIRS - 1,
                     lead_h=LEADS_H[-1]):
            log(f"skip {preset} seed={seed} scale={scale}")
            continue

        key = (preset, seed)
        if key not in records:
            records[key] = load_record(preset, seed)
        S = records[key]
        if S is None:
            continue

        model = SWEModel(preset=preset, LMAX=LMAX, sponge_pole_rows=0)
        fs = model.field_size
        h_sl, w_sl = slice(2 * fs, 3 * fs), slice(0, 2 * fs)
        rng = np.random.default_rng(90000 + 17 * seed)

        for pair in range(PAIRS):
            t0 = time.time()
            a = S[rng.integers(len(S))].copy()
            i, j = rng.choice(len(S), 2, replace=False)
            b = a + scale * (S[i] - S[j])

            rms = lambda sl: float(np.sqrt(np.mean((a[sl] - b[sl]) ** 2)))
            rms_h_0, rms_wind_0 = rms(h_sl), rms(w_sl)
            log(f"{preset} seed={seed} scale={scale} pair={pair}: "
                f"initial separation {rms_h_0:.2f} m, "
                f"{rms_wind_0:.3f} m/s")

            t = 0.0
            blew_up = False
            for lead in LEADS_H:
                if blew_up:
                    break
                t_next = lead * HOUR
                try:
                    a = model.propagate(a, [t, t_next])
                    b = model.propagate(b, [t, t_next])
                except (RuntimeError, FloatingPointError) as exc:
                    log(f"  lead {lead:.0f} h: blew up, {type(exc).__name__}")
                    nan = float("nan")
                    table.add(preset=preset, seed=seed, LMAX=LMAX,
                              scale=scale, pair=pair, lead_h=lead,
                              rms_h_0=round(rms_h_0, 6), rms_h=nan,
                              growth_h=nan,
                              rms_wind_0=round(rms_wind_0, 6), rms_wind=nan,
                              growth_wind=nan, completed=0,
                              seconds=round(time.time() - t0, 1))
                    blew_up = True
                    break
                t = t_next
                rms_h, rms_wind = rms(h_sl), rms(w_sl)
                table.add(
                    preset=preset, seed=seed, LMAX=LMAX, scale=scale,
                    pair=pair, lead_h=lead,
                    rms_h_0=round(rms_h_0, 6), rms_h=round(rms_h, 6),
                    growth_h=round(rms_h / rms_h_0, 6),
                    rms_wind_0=round(rms_wind_0, 6),
                    rms_wind=round(rms_wind, 6),
                    growth_wind=round(rms_wind / rms_wind_0, 6),
                    completed=1, seconds=round(time.time() - t0, 1))
    table.close()
    log("EXP-12 done ->", table.path)


if __name__ == "__main__":
    run()
