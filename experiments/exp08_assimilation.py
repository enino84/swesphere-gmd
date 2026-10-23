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

This version takes the truth and the members from the record EXP-04 writes:
400 states sampled every 2 days after a 60-day spin-up, which is the
climatology. The observation error is set as a fraction of the climatological
depth anomaly measured from that same record rather than as a round number in
metres.

The second version, and why
---------------------------
Drawing members directly from the record was also wrong, in a way that only
showed up when the results were examined rather than merely produced. It gave
the ensemble the climatological spread, roughly 310 m in the depth anomaly,
and a six-hour forecast error in this model is three orders of magnitude
smaller: a 1.9 m perturbation was measured to be 1.9 m again six hours later.
Handed a background wrong by 185 m and good observations, both filters
answered with increments of 223 m, the size of the signal. The depth analysis
was excellent, about 3 m from the truth, but the wind increment came from
cross covariances estimated with 40 climatological members, so the analysis
had roughly the right mass field and the wrong winds, out of geostrophic
balance, with peak speeds of 88 to 91 m/s against a true 80. Six hours later
the error was back to 161 m, and the analysis spread had collapsed to 0.9 m
against it.

None of that was a property of either filter. Both received the same
impossible background, which is why one change repairs both. The members are
now the truth displaced by scaled differences of record states, which are
balanced by construction, with the spread set to about 19 m, and the
background is centred on a separate draw so the ensemble is not accidentally
centred on the truth. Measured over the first ten cycles, the spread to error
ratio is 0.86 to 1.07, the background error falls from 23.6 to 12.9 m, the
analysis holds at 2.9 m, and the peak wind stays near 77 m/s. The first
``SPINUP_CYCLES`` cycles are run but not scored.

EXP-05 reports the integrated autocorrelation time of this record, which is
what says whether two states drawn from it are independent enough to form a
perturbation.

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

#: Size of the initial ensemble perturbations, as a multiple of a difference
#: between two states of the climatological record.
#:
#: This is the parameter the first version of this experiment got wrong, so it
#: is worth stating why it is here. Drawing members directly from the record
#: gives the ensemble the CLIMATOLOGICAL spread, about 310 m in the depth
#: anomaly, which is three orders of magnitude larger than a six-hour forecast
#: error. Both filters then answer with increments the size of the signal
#: itself. The analysis does fit the observations, to about 3 m, but the wind
#: increments come from cross-covariances estimated with 40 climatological
#: members, which carry no information about a short-range error. The result
#: is a state with roughly the right depth and the wrong winds, out of
#: geostrophic balance, whose error is back to 160 m six hours later. The
#: analysis spread collapses to under 1 m against that 160 m error.
#:
#: At 0.05 the initial spread is about 19 m, six percent of climatological,
#: and the measured spread-to-error ratio over the first ten cycles is 0.86 to
#: 1.07. That is the regime this experiment is supposed to be in. The failure
#: was never in either filter, which is why one change fixes both.
PERTURBATION_SCALE = 0.05

#: Cycles run before scoring starts, to let the perturbations take on the
#: structure of real forecast errors. Measured: the background error settles
#: into its decay after about three cycles.
SPINUP_CYCLES = 1 if QUICK else 4

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


def draw_from_climatology(S: np.ndarray, n: int, rng,
                          scale: float = PERTURBATION_SCALE):
    """One truth and ``n`` members, as an observing-system experiment.

    The members are the truth displaced by scaled DIFFERENCES of states from
    the record, not states from the record. A difference of two solutions of
    the same equations is a dynamically consistent perturbation, so the
    members stay in balance, and scaling it sets the spread independently of
    the climatology. See ``PERTURBATION_SCALE`` for why that matters.

    The background centre is itself one such displacement, drawn from the
    same distribution as the member perturbations. That detail is not
    cosmetic. If the ensemble is centred on the truth, the ensemble mean is
    accurate to sigma/sqrt(n) by construction, the background error comes out
    about six times smaller than the spread, and the experiment reports a
    wildly overdispersive ensemble that is an artifact of how it was built.
    Centring on a separate draw gives a spread-to-error ratio near one, which
    is what a correctly specified ensemble has.
    """
    k = S.shape[0]
    if k < 4:
        raise ValueError(
            f"the record has {k} snapshots; at least 4 are needed to form "
            f"independent differences. Run EXP-04 first.")

    def displacement():
        i, j = rng.choice(k, 2, replace=False)
        return scale * (S[i] - S[j])

    truth = S[rng.integers(k)].copy()
    centre = truth + displacement()
    Xb = np.column_stack([centre + displacement() for _ in range(n)])
    #: Mean separation in model days between the two states of a difference,
    #: reported so the perturbation structure is traceable in the CSV.
    spacing_days = k / 3.0 * 2.0
    return truth, Xb, spacing_days


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
                "member_spacing_days", "completed", "rmse_h_b", "rmse_h_a",
                "rmse_wind_b", "rmse_wind_a", "spread_h_b", "spread_h_a", "incr_h",
                "incr_wind", "seconds"],
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
        class CrossLocalized(SWEModel):
            """A model whose localization reaches across variables.

            This subclass exists because of a gap between pyteda's analyses
            and its models. Both ``AnalysisLETKF`` and the local modified
            Cholesky filter call ``self.model.get_ngb(i, r)`` with no
            ``cross`` argument, and their docstrings say "the model handles
            the dispatch in ``get_ngb``". ``SWEModel.get_ngb`` does accept
            ``cross``, but it defaults to ``False`` and ``SWEModel.__init__``
            takes no setting that would change it, so the default always
            wins.

            The consequence is that passing ``cross=True`` to
            ``AnalysisFactory`` does nothing at all: the keyword lands in
            ``**kwargs`` and is never read. The first run of this experiment
            did exactly that, and its 18 cross-variable configurations were
            identical to the 18 univariate ones, down to the last digit of
            the RMSE. The wind increment was zero in every row of the table,
            including the rows that existed to show it was not.

            Overriding the default is the only way to make the request
            through the published API. At radius 2 it enlarges each local
            domain from 25 indices to 75, which is what lets a depth
            observation reach the wind.
            """

            def get_ngb(self, i, r, cross=True):
                return super().get_ngb(i, r, cross=True)

            def get_pre(self, i, r, cross=True):
                return super().get_pre(i, r, cross=True)

        def new_model(localized_across_variables=False):
            cls = CrossLocalized if localized_across_variables else SWEModel
            return cls(preset=PRESET, LMAX=LMAX, sponge_pole_rows=0)

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
        # ``cross`` is expressed through the model, not through the factory.
        # Passing it to the factory would be silently ignored; see
        # CrossLocalized.
        analysis = AnalysisFactory(
            method=method, model=new_model(localized_across_variables=cross),
            **kw).create_analysis()

        t = 0.0
        # The first SPINUP_CYCLES cycles are run but not scored, so the
        # perturbations can take on the structure of a real forecast error
        # before any number is reported. Their index is negative, which makes
        # a spin-up failure visible in the CSV rather than indistinguishable
        # from a first-cycle one.
        for step in range(SPINUP_CYCLES + CYCLES):
            cycle = step - SPINUP_CYCLES
            scored = cycle >= 0
            t_next = t + WINDOW_H * HOUR
            # A cycle that cannot be integrated is a row of this table, not a
            # crash. The first run of this experiment against swesphere 0.6.0
            # lost four of eight shards to one unhandled RuntimeError each,
            # and with them 14 configurations that had nothing wrong with
            # them. It is also a result in its own right: an analysis the
            # model cannot carry forward is a fact about the filter and the
            # model together, and silence about it would be the worst of the
            # three options.
            try:
                truth = model.propagate(truth, [t, t_next])
                for e in range(ENSEMBLE):
                    Xb[:, e] = model.propagate(Xb[:, e], [t, t_next])
            except (RuntimeError, FloatingPointError) as exc:
                log(f"  cycle {cycle}: forecast blew up, {type(exc).__name__}")
                nan = float("nan")
                table.add(
                    method=method, cross=int(cross), seed=seed,
                    spacing=spacing, obs_fraction=round(obs_fraction, 5),
                    cycle=cycle, LMAX=LMAX, ensemble=ENSEMBLE,
                    window_h=WINDOW_H, obs_std=round(obs_std, 4),
                    inflation=INFLATION,
                    member_spacing_days=round(member_spacing, 1),
                    completed=0, rmse_h_b=nan, rmse_h_a=nan, rmse_wind_b=nan,
                    rmse_wind_a=nan, spread_h_b=nan, spread_h_a=nan, incr_h=nan,
                    incr_wind=nan, seconds=round(time.time() - t0, 1))
                break
            t = t_next

            Xb_inf = inflate(Xb, INFLATION)
            bg.Xb = Xb_inf
            obs = Observation.from_arrays(
                op.apply(truth) + noise.sample(rng), op, noise)
            Xa = np.asarray(analysis.perform_assimilation(bg, obs))

            xb_mean, xa_mean = Xb_inf.mean(axis=1), Xa.mean(axis=1)
            rmse = lambda x, sl: float(np.sqrt(np.mean((x[sl] - truth[sl]) ** 2)))
            if not scored:
                Xb = Xa
                continue
            table.add(
                method=method, cross=int(cross), seed=seed, spacing=spacing,
                obs_fraction=round(obs_fraction, 5), cycle=cycle, LMAX=LMAX,
                ensemble=ENSEMBLE, window_h=WINDOW_H,
                obs_std=round(obs_std, 4), inflation=INFLATION,
                member_spacing_days=round(member_spacing, 1), completed=1,
                rmse_h_b=rmse(xb_mean, h_sl), rmse_h_a=rmse(xa_mean, h_sl),
                rmse_wind_b=rmse(xb_mean, w_sl),
                rmse_wind_a=rmse(xa_mean, w_sl),
                # Background spread beside background error, so the
                # calibration of the ensemble is in the data. A ratio near one
                # is the point; the version of this experiment that drew
                # members straight from the climatology produced 0.006.
                spread_h_b=float(Xb_inf[h_sl].std(axis=1).mean()),
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
