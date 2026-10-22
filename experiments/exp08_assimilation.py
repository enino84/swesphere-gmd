# -*- coding: utf-8 -*-
"""EXP-08. What the characterization is for: a cycled assimilation experiment.

Reviewer 2's fourth objection, which is the one that decides whether the
paper is a model description or a contribution: "A representative,
reproducible algorithm experiment would help demonstrate how the supplied
characterization supports a concrete research question and what practical
benefit the package offers relative to an appropriate existing testbed."

The concrete research question is one that Lorenz 96 cannot pose at all, and
it is the reason a multivariate testbed exists: **when you observe only the
mass field, does a localized ensemble filter recover the wind?**

The answer is not automatic. A localized filter restricts each analysis to a
neighbourhood, and whether that neighbourhood reaches across variables is a
choice. With the blocks kept separate the analysis is univariate and the wind
increment is identically zero however good the depth observations are. With
them joined, the observation reaches the wind through the ensemble cross
covariances, at the cost of trusting a correlation estimated from a few dozen
members. Which is better is an empirical question about a particular flow,
and it depends on how much of that flow you observe, which is why the
observation density is swept here rather than fixed.

How the characterization is used, which is the half the first version of this
experiment was missing
--------------------------------------------------------------------------
The earlier version built the truth and the ensemble with
``get_initial_condition(seed=...)``, which perturbs the preset's initial
state. The method's own documentation warns against exactly that: "since that
perturbation is unbalanced, pass ``T`` or propagate before sampling". Nothing
was propagated, so every member started off the attractor, the first cycles
measured gravity waves settling rather than assimilation, and the ensemble
spread had no relation to the climatological spread it is supposed to
approximate. Worse for the paper, it meant Sects. 5 and 6 built a
characterization that Sect. 7 then ignored.

This version takes the truth and all the members from the record EXP-04
writes: 400 states sampled every 2 days after a 60-day spin-up, which is the
climatology. Members are spread evenly over the record, so consecutive
members are 18 model days apart, and EXP-05 reports the integrated
autocorrelation time against which that spacing should be read. The
observation error is set as a fraction of the climatological depth anomaly
measured from the same record rather than as a round number in metres.

Inflation
---------
pyteda's filters take no inflation parameter, so it is applied here, to the
background anomalies, before the analysis. Without any, a cycled ensemble
this small loses spread within a few cycles and the filter stops listening to
the observations; the symptom is in ``spread_h_a`` falling while
``rmse_h_a`` rises.

Needs ``pip install pyteda[swe]`` and the records from EXP-04.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from common import (DAY, HOUR, QUICK, RESULTS, SHARD_COUNT, SHARD_INDEX,
                    Table, log, shard)

warnings.filterwarnings("ignore")

PRESET = "two_jets"
#: Must match EXP-04, whose record this reads.
LMAX = 16 if QUICK else 32
ENSEMBLE = 4 if QUICK else 40
CYCLES = 3 if QUICK else 40
WINDOW_H = 6.0

#: Observation error as a fraction of the climatological standard deviation
#: of the depth anomaly, so it means the same thing at every resolution and
#: in every regime rather than being a number of metres chosen once.
OBS_ERROR_FRACTION = 0.02

#: Spacings of the observation grid. The fraction of points each one
#: observes is measured and recorded rather than assumed.
SPACINGS = [4] if QUICK else [2, 4, 8]

#: Multiplicative inflation of the background anomalies.
INFLATION = 1.02

SEEDS = [0] if QUICK else [0, 1, 2]
METHODS = (
    ("letkf", dict(r=2)),
    ("enkf-modified-cholesky", dict(r=2)),
)


def load_record(seed: int):
    """The climatological record EXP-04 wrote, or ``None`` with an explanation.

    This experiment cannot be run before EXP-04, and failing clearly here is
    better than silently falling back to perturbed initial states, which is
    the mistake this version exists to correct.
    """
    path = RESULTS / f"record_{PRESET}_seed{seed}_L{LMAX}.npy"
    if not path.exists():
        log(f"MISSING {path.name}: run exp04_regimes.py first "
            f"(this experiment draws its ensemble from the climatology)")
        return None
    return np.load(path).astype(float)


def draw_from_climatology(S: np.ndarray, n: int, rng):
    """One truth and ``n`` members, spread evenly over the record.

    Even spacing rather than random sampling, so the distance between
    members is known and reported rather than merely probable. With 400
    snapshots two days apart and 41 states to draw, consecutive draws are 18
    model days apart.
    """
    k = S.shape[0]
    if k < n + 1:
        raise ValueError(
            f"the record has {k} snapshots and {n + 1} states were asked for "
            f"(one truth and {n} members). Drawing more states than the "
            f"record holds would repeat snapshots, which would give the "
            f"ensemble a spread smaller than the climatological one and "
            f"quietly flatter every filter. Lengthen the record in EXP-04 "
            f"(N_SNAPSHOTS) or reduce ENSEMBLE.")
    idx = np.linspace(0, k - 1, n + 1).round().astype(int)
    rng.shuffle(idx)
    spacing_days = (k - 1) / (n + 1) * 2.0
    return S[idx[0]].copy(), np.column_stack([S[j] for j in idx[1:]]), spacing_days


def inflate(Xb: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return Xb
    m = Xb.mean(axis=1, keepdims=True)
    return m + factor * (Xb - m)


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
        fields=["method", "cross", "seed", "spacing", "obs_fraction", "cycle",
                "LMAX", "ensemble", "window_h", "obs_std", "inflation",
                "member_spacing_days", "rmse_h_b", "rmse_h_a", "rmse_wind_b",
                "rmse_wind_a", "spread_h_a", "incr_h", "incr_wind", "seconds"],
        key_fields=["method", "cross", "seed", "spacing", "cycle"],
    )

    cases = [(seed, spacing, method, kw, cross)
             for seed in SEEDS for spacing in SPACINGS
             for method, kw in METHODS for cross in (False, True)]
    mine = shard(cases)
    if SHARD_COUNT > 1:
        log(f"shard {SHARD_INDEX} of {SHARD_COUNT}: "
            f"{len(mine)} of {len(cases)} configurations")

    records = {}
    for seed, spacing, method, kw, cross in mine:
        if table.has(method=method, cross=int(cross), seed=seed,
                     spacing=spacing, cycle=CYCLES - 1):
            log(f"skip {method} cross={cross} seed={seed} spacing={spacing}")
            continue

        if seed not in records:
            records[seed] = load_record(seed)
        S = records[seed]
        if S is None:
            continue

        t0 = time.time()
        rng = np.random.default_rng(1000 + seed)

        # sponge_pole_rows=0 explicitly. pyteda's adapter carries its own
        # default of 10, so leaving it out would give this experiment a
        # sponge that swesphere 0.6.0 no longer applies anywhere else.
        def new_model():
            return SWEModel(preset=PRESET, LMAX=LMAX, sponge_pole_rows=0)

        model = new_model()
        fs, dim = model.field_size, model.dim
        h_sl, w_sl = slice(2 * fs, 3 * fs), slice(0, 2 * fs)

        truth, Xb, member_spacing = draw_from_climatology(S, ENSEMBLE, rng)

        # The observation error, from the climatology rather than from habit.
        clim_anomaly = float(
            (S[:, h_sl] - S[:, h_sl].mean(axis=0, keepdims=True)).std())
        obs_std = OBS_ERROR_FRACTION * clim_anomaly

        idx, m_obs = strided_indices(model, spacing=spacing, variables=["h"])
        obs_fraction = m_obs / fs
        op = LinearSelection(m_obs, dim, indices=idx, rng=rng)
        noise = IsotropicDiagonal(obs_std, dim=m_obs)

        log(f"{method} cross={cross} seed={seed} spacing={spacing}: "
            f"{m_obs} sites ({100 * obs_fraction:.1f}% of a field), "
            f"obs_std={obs_std:.2f} m, members {member_spacing:.0f} d apart")

        bg = Background(model, ensemble_size=ENSEMBLE)
        analysis = AnalysisFactory(method=method, model=new_model(),
                                   **dict(kw, cross=cross)).create_analysis()

        t = 0.0
        for cycle in range(CYCLES):
            t_next = t + WINDOW_H * HOUR
            truth = model.propagate(truth, [t, t_next])
            for e in range(ENSEMBLE):
                Xb[:, e] = model.propagate(Xb[:, e], [t, t_next])
            t = t_next

            Xb_inf = inflate(Xb, INFLATION)
            bg.Xb = Xb_inf
            obs = Observation.from_arrays(
                op.apply(truth) + noise.sample(rng), op, noise)
            Xa = np.asarray(analysis.perform_assimilation(bg, obs))

            xb_mean, xa_mean = Xb_inf.mean(axis=1), Xa.mean(axis=1)
            rmse = lambda x, sl: float(np.sqrt(np.mean((x[sl] - truth[sl]) ** 2)))
            table.add(
                method=method, cross=int(cross), seed=seed, spacing=spacing,
                obs_fraction=round(obs_fraction, 5), cycle=cycle, LMAX=LMAX,
                ensemble=ENSEMBLE, window_h=WINDOW_H,
                obs_std=round(obs_std, 4), inflation=INFLATION,
                member_spacing_days=round(member_spacing, 1),
                rmse_h_b=rmse(xb_mean, h_sl), rmse_h_a=rmse(xa_mean, h_sl),
                rmse_wind_b=rmse(xb_mean, w_sl),
                rmse_wind_a=rmse(xa_mean, w_sl),
                spread_h_a=float(Xa[h_sl].std(axis=1).mean()),
                # The point of the experiment: with the blocks kept apart
                # this is exactly zero, however good the depth observations
                # are. Recorded so the claim is in the data, not the prose.
                incr_h=float(np.abs(Xa[h_sl] - Xb_inf[h_sl]).mean()),
                incr_wind=float(np.abs(Xa[w_sl] - Xb_inf[w_sl]).mean()),
                seconds=round(time.time() - t0, 1),
            )
            Xb = Xa
    table.close()
    log("EXP-08 done ->", table.path)


if __name__ == "__main__":
    run()
