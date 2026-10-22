# -*- coding: utf-8 -*-
"""EXP-04. Long records of each regime, from several independent starts.

This is the expensive one, and the one everything in EXP-05 is computed from.
Start it first.

The submitted manuscript characterized each preset from a single record and
compared the two halves of it. Both reviewers objected: comparing two window
averages of one diagnostic from one realization says very little about
whether the statistics are stable, and it gives no uncertainty at all. The
fix has two parts, and this is the first.

Several **independent realizations** per preset, each from a different seed,
each spun up separately. Agreement between realizations is evidence that a
regime is a property of the model rather than of one trajectory; disagreement
is the honest answer. The records are the raw material, written as float32
``.npy`` so EXP-05 can compute uncertainties without re-running anything.

Sampling is deliberately denser than the manuscript's, so that the
autocorrelation of the record can be estimated rather than assumed, and the
record is long enough that the integrated autocorrelation time is a small
fraction of it.
"""

from __future__ import annotations

import time

import numpy as np
from swesphere import climatology, diagnostics, presets

from common import DAY, QUICK, RESULTS, Table, interior, log, SHARD_COUNT, SHARD_INDEX, shard

PRESETS = ["two_jets"] if QUICK else ["waves", "one_jet", "two_jets"]
SEEDS = [0] if QUICK else [0, 1, 2]
SPINUP_DAYS = 2.0 if QUICK else 60.0
N_SNAPSHOTS = 8 if QUICK else 400
EVERY_DAYS = 0.5 if QUICK else 2.0
LMAX = 16 if QUICK else 32


def eddy_std(model, x: np.ndarray) -> float:
    """Standard deviation of the depth about its zonal mean, interior only."""
    h = model.unpack(x)[2]
    return float(interior(h - h.mean(axis=1, keepdims=True), model.n_pole_rows).std())


def run() -> None:
    table = Table(
        "exp04_regimes",
        fields=["preset", "seed", "LMAX", "spinup_days", "n_snapshots",
                "every_days", "record_days", "eddy_std_mean", "eddy_std_sd",
                "umax_mean", "mass_drift", "energy_drift", "record_file",
                "seconds"],
        key_fields=["preset", "seed", "LMAX"],
    )
    cases = [(n, s) for n in PRESETS for s in SEEDS]
    mine = shard(cases)
    if SHARD_COUNT > 1:
        log(f"shard {SHARD_INDEX} of {SHARD_COUNT}: "
            f"{len(mine)} of {len(cases)} records")
    for name, seed in mine:
        if table.has(preset=name, seed=seed, LMAX=LMAX):
            log(f"skip {name} seed={seed}")
            continue
        log(f"{name} seed={seed} LMAX={LMAX}: spin-up {SPINUP_DAYS:g} d, "
            f"then {N_SNAPSHOTS} states every {EVERY_DAYS:g} d")
        t0 = time.time()

        model, x0 = presets.make(name, LMAX=LMAX)
        # A different start per realization. Seed 0 is the preset's own
        # state, so one record is always the documented one.
        if seed != 0:
            x0 = model.initial_condition("rossby", seed=seed, perturbed=True) \
                if name == "waves" else \
                model.pack(*[f * (1.0 + 1e-3 * np.sin(seed * model.grid.LON_ns))
                             for f in model.unpack(x0)])

        inv0 = diagnostics.invariants(model, x0)
        S = climatology.build_climatology(
            model, x0, spinup_days=SPINUP_DAYS, n_snapshots=N_SNAPSHOTS,
            every_days=EVERY_DAYS, verbose=True, label=f"{name}/{seed}")
        inv1 = diagnostics.invariants(model, np.asarray(S[-1], dtype=float))

        rec = RESULTS / f"record_{name}_seed{seed}_L{LMAX}.npy"
        np.save(rec, S)

        eddy = np.array([eddy_std(model, np.asarray(s, dtype=float))
                         for s in S])
        umax = np.array([float(np.abs(np.hypot(*model.unpack(
            np.asarray(s, dtype=float))[:2])).max()) for s in S])

        table.add(
            preset=name, seed=seed, LMAX=LMAX, spinup_days=SPINUP_DAYS,
            n_snapshots=N_SNAPSHOTS, every_days=EVERY_DAYS,
            record_days=N_SNAPSHOTS * EVERY_DAYS,
            eddy_std_mean=float(eddy.mean()), eddy_std_sd=float(eddy.std()),
            umax_mean=float(umax.mean()),
            mass_drift=abs(inv1["mass"] / inv0["mass"] - 1.0),
            energy_drift=abs(inv1["energy"] / inv0["energy"] - 1.0),
            record_file=rec.name, seconds=round(time.time() - t0, 1),
        )
    table.close()
    log("EXP-04 done ->", table.path)


if __name__ == "__main__":
    run()
