# -*- coding: utf-8 -*-
"""EXP-10. Is the polar sponge necessary at all?

Not a reviewer's question. It came out of answering theirs, and it may be the
most consequential thing in this repository, so it gets measured properly
instead of being argued about.

The sponge is the dominant error of the shipped configuration. EXP-01 shows
that with the sponge covering a fixed latitude band the solution barely
converges at all: refining the grid by a factor of two reduces the error by a
few percent rather than a factor of four, because damping the wind in a fixed
band is an order-one perturbation that does not shrink with resolution. The
apparent convergence that the manuscript reported came from holding
``n_pole_rows`` fixed at 10, which is a different latitude at every
resolution: 37 degrees at LMAX 16, 63 at LMAX 32, 83 at LMAX 128. Refining
the grid was quietly retreating the sponge toward the poles.

So the question is whether the model needs it. Spot checks on ``two_jets``
say no: at LMAX 32 and 64 it runs 20 days without any sponge, at the same
cost per model day, reaching a slightly larger peak wind because nothing is
removing it. If that holds for every preset, over the record lengths the
characterization uses, and up to the resolutions people might want, then the
sponge is not a necessary evil, it is a removable one, and the model becomes
second-order accurate on the whole sphere instead of first-order-ish on a
band.

What this measures, for each preset and resolution, with the sponge at
several widths down to none:

  * whether the integration survives, and for how long;
  * the cost per model day, so "we need it for stability" cannot hide a
    performance argument;
  * the peak wind and the eddy amplitude, which say how much of the flow the
    sponge was removing;
  * the invariants, which say whether a run that survives is also sane.

The honest failure mode to look for is a run that completes while quietly
accumulating grid-scale noise at the poles. ``polar_roughness`` below is the
amplitude of the highest resolved zonal wavenumbers in the polar rows: it
grows long before an integration actually blows up, so a model that is
heading for trouble shows it here first.
"""

from __future__ import annotations

import time

import numpy as np
from swesphere import diagnostics, presets

from common import DAY, QUICK, SHARD_COUNT, SHARD_INDEX, Table, log, shard

PRESETS = ["two_jets"] if QUICK else ["waves", "one_jet", "two_jets"]
RESOLUTIONS = [16] if QUICK else [32, 64, 96]
POLE_ROWS = [10, 0] if QUICK else [10, 6, 3, 1, 0]
DAYS = 1.0 if QUICK else 120.0
CHECK_EVERY_DAYS = 0.5 if QUICK else 5.0


def polar_roughness(model, x: np.ndarray, n_rows: int = 6) -> float:
    """Amplitude of the shortest resolved zonal waves in the polar rows.

    Relative to the amplitude of the same wavenumbers in midlatitudes, so it
    is a shape measure and not a units measure. Grid-scale noise piling up at
    the pole shows here well before the integration fails.
    """
    u = model.unpack(x)[0]
    spec = np.abs(np.fft.rfft(u, axis=1))
    high = slice(spec.shape[1] * 3 // 4, None)
    polar = np.concatenate([spec[:n_rows, high], spec[-n_rows:, high]]).mean()
    mid = spec[spec.shape[0] // 2 - n_rows:spec.shape[0] // 2 + n_rows, high].mean()
    return float(polar / mid) if mid > 0 else float("nan")


def run() -> None:
    table = Table(
        "exp10_sponge_necessity",
        fields=["preset", "LMAX", "pole_rows", "sponge_lat_deg", "survived",
                "days_completed", "days_requested", "umax", "eddy_std_h",
                "polar_roughness", "mass_drift", "energy_drift",
                "seconds_per_day"],
        key_fields=["preset", "LMAX", "pole_rows"],
    )
    cases = [(name, L, rows) for name in PRESETS for L in RESOLUTIONS
             for rows in POLE_ROWS]
    mine = shard(cases)
    if SHARD_COUNT > 1:
        log(f"shard {SHARD_INDEX} of {SHARD_COUNT}: "
            f"{len(mine)} of {len(cases)} cases")
    for name, L, rows in mine:
        d_lat = 180.0 / (2 * (L + 1))
        if table.has(preset=name, LMAX=L, pole_rows=rows):
            log(f"skip {name} LMAX={L} rows={rows}")
            continue
        log(f"{name} LMAX={L} rows={rows} "
            f"(sponge from {90 - rows * d_lat:.1f} deg)")
        t0 = time.time()
        model, x0 = presets.make(name, LMAX=L, n_pole_rows=rows)
        inv0 = diagnostics.invariants(model, x0)

        x, t, done, survived = x0, 0.0, 0.0, True
        step = CHECK_EVERY_DAYS * DAY
        while done < DAYS - 1e-9:
            try:
                x = model.propagate(x, [t, t + step])
            except RuntimeError:
                survived = False
                break
            t += step
            done += CHECK_EVERY_DAYS
            if not np.isfinite(x).all():
                survived = False
                break

        elapsed = time.time() - t0
        if survived:
            u, v, h = model.unpack(x)
            inv1 = diagnostics.invariants(model, x)
            k = max(1, int(round(27.3 / d_lat)))   # score poleward of 63 deg
            table.add(
                preset=name, LMAX=L, pole_rows=rows,
                sponge_lat_deg=round(90.0 - rows * d_lat, 2),
                survived=1, days_completed=done, days_requested=DAYS,
                umax=float(np.hypot(u, v).max()),
                eddy_std_h=float((h - h.mean(axis=1, keepdims=True))[k:-k].std()),
                polar_roughness=polar_roughness(model, x),
                mass_drift=abs(inv1["mass"] / inv0["mass"] - 1.0),
                energy_drift=abs(inv1["energy"] / inv0["energy"] - 1.0),
                seconds_per_day=round(elapsed / max(done, 1e-9), 2),
            )
        else:
            log(f"  blew up after {done:g} days")
            table.add(
                preset=name, LMAX=L, pole_rows=rows,
                sponge_lat_deg=round(90.0 - rows * d_lat, 2),
                survived=0, days_completed=done, days_requested=DAYS,
                umax=float("nan"), eddy_std_h=float("nan"),
                polar_roughness=float("nan"), mass_drift=float("nan"),
                energy_drift=float("nan"),
                seconds_per_day=round(elapsed / max(done, 1e-9), 2),
            )
    table.close()
    log("EXP-10 done ->", table.path)


if __name__ == "__main__":
    run()
