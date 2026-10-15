# -*- coding: utf-8 -*-
"""EXP-08. What the characterization is for: a cycled assimilation experiment.

Reviewer 2's fourth objection, which is the one that decides whether the
paper is a model description or a contribution: "A representative,
reproducible algorithm experiment would help demonstrate how the supplied
characterization supports a concrete research question and what practical
benefit the package offers relative to an appropriate existing testbed."

The concrete research question this answers is one that Lorenz-96 cannot
pose at all, and it is the reason a multivariate testbed exists: **when you
observe only the mass field, does a localized ensemble filter recover the
wind?**

The answer is not automatic. A localized filter restricts each analysis to a
neighbourhood, and whether that neighbourhood reaches across variables is a
choice. With the blocks kept separate the analysis is univariate and the wind
increment is identically zero however good the depth observations are. With
them joined, the observation reaches the wind through the ensemble cross
covariances, at the cost of trusting a correlation estimated from a few dozen
members. Which is better is an empirical question about a particular flow,
and answering it needs exactly what this package supplies: a documented
regime, a climatological record to size the perturbations and the
observation error, and a cheap forecast.

The experiment cycles each filter over the same scenario and reports the
analysis error separately for the observed variable and for the unobserved
ones, which is where the two settings differ.

Needs ``pip install pyteda[swe]``.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from common import DAY, HOUR, QUICK, Table, log

warnings.filterwarnings("ignore")

LMAX = 8 if QUICK else 16
ENSEMBLE = 20 if QUICK else 40
CYCLES = 3 if QUICK else 40
WINDOW_H = 6.0
OBS_SPACING = 4
OBS_STD = 2.0                 # metres on the depth field
SEEDS = [0] if QUICK else [0, 1, 2]
METHODS = (
    ("letkf", dict(r=2)),
    ("enkf-modified-cholesky", dict(r=2)),
)


def run() -> None:
    from pyteda.models import SWEModel
    from pyteda.observation.grid_masks import strided_indices
    from pyteda.observation.noise import IsotropicDiagonal
    from pyteda.observation.operators import LinearSelection
    from pyteda.observation import Observation
    from pyteda.background import Background
    from pyteda.analysis.analysis_factory import AnalysisFactory

    table = Table(
        "exp08_assimilation",
        fields=["method", "cross", "seed", "cycle", "LMAX", "ensemble",
                "window_h", "rmse_h_b", "rmse_h_a", "rmse_wind_b",
                "rmse_wind_a", "spread_h_a", "incr_h", "incr_wind", "seconds"],
        key_fields=["method", "cross", "seed", "cycle"],
    )

    for seed in SEEDS:
        for method, kw in METHODS:
            for cross in (False, True):
                if table.has(method=method, cross=int(cross), seed=seed,
                             cycle=CYCLES - 1):
                    log(f"skip {method} cross={cross} seed={seed}")
                    continue
                log(f"{method} cross={cross} seed={seed}: {CYCLES} cycles")
                t0 = time.time()
                rng = np.random.default_rng(1000 + seed)

                model = SWEModel(preset="two_jets", LMAX=LMAX)
                fs, dim = model.field_size, model.dim
                truth = model.get_initial_condition(seed=7777 + seed)
                Xb = np.column_stack([
                    model.get_initial_condition(seed=10_000 + 100 * seed + e)
                    for e in range(ENSEMBLE)])

                idx, m_obs = strided_indices(model, spacing=OBS_SPACING,
                                             variables=["h"])
                op = LinearSelection(m_obs, dim, indices=idx, rng=rng)
                noise = IsotropicDiagonal(OBS_STD, dim=m_obs)

                bg = Background(model, ensemble_size=ENSEMBLE)
                t = 0.0
                for cycle in range(CYCLES):
                    # Forecast every member and the truth over the window.
                    t_next = t + WINDOW_H * HOUR
                    truth = model.propagate(truth, [t, t_next])
                    for e in range(ENSEMBLE):
                        Xb[:, e] = model.propagate(Xb[:, e], [t, t_next])
                    t = t_next

                    obs = Observation.from_arrays(
                        op.apply(truth) + noise.sample(rng), op, noise)
                    bg.Xb = Xb

                    xb_mean = Xb.mean(axis=1)
                    analysis = AnalysisFactory(
                        method=method, model=SWEModel(preset="two_jets",
                                                      LMAX=LMAX),
                        **dict(kw, cross=cross)).create_analysis()
                    Xa = np.asarray(analysis.perform_assimilation(bg, obs))
                    xa_mean = Xa.mean(axis=1)

                    rmse = lambda x, sl: float(np.sqrt(
                        np.mean((x[sl] - truth[sl]) ** 2)))
                    h_sl, w_sl = slice(2 * fs, 3 * fs), slice(0, 2 * fs)
                    table.add(
                        method=method, cross=int(cross), seed=seed,
                        cycle=cycle, LMAX=LMAX, ensemble=ENSEMBLE,
                        window_h=WINDOW_H,
                        rmse_h_b=rmse(xb_mean, h_sl), rmse_h_a=rmse(xa_mean, h_sl),
                        rmse_wind_b=rmse(xb_mean, w_sl),
                        rmse_wind_a=rmse(xa_mean, w_sl),
                        spread_h_a=float(Xa[h_sl].std(axis=1).mean()),
                        # The point of the experiment: with the blocks kept
                        # apart this is exactly zero, however good the depth
                        # observations are. Recorded so the claim is in the
                        # data and not in the prose.
                        incr_h=float(np.abs(Xa[h_sl] - Xb[h_sl]).mean()),
                        incr_wind=float(np.abs(Xa[w_sl] - Xb[w_sl]).mean()),
                        seconds=round(time.time() - t0, 1),
                    )
                    Xb = Xa
    table.close()
    log("EXP-08 done ->", table.path)


if __name__ == "__main__":
    run()
