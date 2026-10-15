# -*- coding: utf-8 -*-
"""EXP-05. Is the regime stationary, and how well do we know?

The second half of the answer to the reviewers on stationarity. This reads
the records EXP-04 wrote and computes nothing new from the model, so it is
cheap and can be re-run as the records grow.

What the manuscript did: split one record in half, compare two means, call a
2% difference stationary. What was wrong with it, in the reviewers' words:
"comparing two window averages of one diagnostic provides limited evidence
for the stability of the multivariate statistics... The standard deviation of
individual samples is also different from the uncertainty of a time average."

Both are right, and the second is the sharper point. Successive states two
days apart are correlated, so a record of 400 of them does not carry 400
independent pieces of information, and dividing the sample standard deviation
by sqrt(400) would understate the uncertainty of the mean several-fold.

So for each diagnostic this computes:

  * the integrated autocorrelation time ``tau`` of the series, and the
    effective sample size ``n / (2 tau + 1)``;
  * a moving-block bootstrap confidence interval for the mean, with blocks
    long enough to carry the correlation;
  * the same statistics over several splits, not only halves, because a
    quantity can look stable across halves and drift within them;
  * the spread **between** independent realizations, which is the thing the
    manuscript never had.

A regime is reported as stationary when the intervals of its sub-windows
overlap, which is a statement with an error bar attached rather than a
percentage.
"""

from __future__ import annotations

import numpy as np
from swesphere import presets

from common import (QUICK, RESULTS, Table, block_bootstrap_ci,
                    effective_sample_size, integral_autocorrelation_time,
                    interior, log)

N_WINDOWS = [2, 3, 4]


def series_from_record(model, S: np.ndarray) -> dict:
    """The diagnostics each record is summarized by."""
    out = {"eddy_std_h": [], "umax": [], "ke": []}
    for s in S:
        u, v, h = model.unpack(np.asarray(s, dtype=float))
        out["eddy_std_h"].append(
            float(interior(h - h.mean(axis=1, keepdims=True), model.n_pole_rows).std()))
        out["umax"].append(float(np.hypot(u, v).max()))
        out["ke"].append(float(np.mean(interior(0.5 * (u ** 2 + v ** 2),
                                                model.n_pole_rows))))
    return {k: np.asarray(v) for k, v in out.items()}


def run() -> None:
    records = sorted(RESULTS.glob("record_*.npy"))
    if not records:
        log("no records found; run exp04_regimes.py first")
        return

    per_series = Table(
        "exp05_stationarity",
        fields=["preset", "seed", "LMAX", "quantity", "n", "mean", "sd",
                "tau", "n_effective", "sd_of_mean_naive", "sd_of_mean_corrected",
                "ci_lo", "ci_hi"],
        key_fields=["preset", "seed", "LMAX", "quantity"],
    )
    per_window = Table(
        "exp05_windows",
        fields=["preset", "seed", "LMAX", "quantity", "n_windows", "window",
                "mean", "ci_lo", "ci_hi"],
        key_fields=["preset", "seed", "LMAX", "quantity", "n_windows", "window"],
    )

    for path in records:
        stem = path.stem.split("_")          # record_<preset..>_seed<k>_L<lmax>
        lmax = int(stem[-1][1:])
        seed = int(stem[-2][4:])
        name = "_".join(stem[1:-2])
        log(f"{path.name}: preset={name} seed={seed} LMAX={lmax}")

        model, _ = presets.make(name, LMAX=lmax)
        S = np.load(path, mmap_mode="r")
        series = series_from_record(model, S)

        for q, x in series.items():
            n = x.size
            tau = integral_autocorrelation_time(x)
            n_eff = effective_sample_size(x)
            lo, hi = block_bootstrap_ci(x)
            if not per_series.has(preset=name, seed=seed, LMAX=lmax, quantity=q):
                per_series.add(
                    preset=name, seed=seed, LMAX=lmax, quantity=q, n=n,
                    mean=float(x.mean()), sd=float(x.std(ddof=1)), tau=tau,
                    n_effective=n_eff,
                    sd_of_mean_naive=float(x.std(ddof=1) / np.sqrt(n)),
                    sd_of_mean_corrected=float(x.std(ddof=1) / np.sqrt(max(n_eff, 1.0))),
                    ci_lo=lo, ci_hi=hi,
                )
            for k in N_WINDOWS:
                edges = np.linspace(0, n, k + 1).astype(int)
                for w in range(k):
                    seg = x[edges[w]:edges[w + 1]]
                    if seg.size < 4 or per_window.has(
                            preset=name, seed=seed, LMAX=lmax, quantity=q,
                            n_windows=k, window=w):
                        continue
                    slo, shi = block_bootstrap_ci(seg)
                    per_window.add(preset=name, seed=seed, LMAX=lmax,
                                   quantity=q, n_windows=k, window=w,
                                   mean=float(seg.mean()), ci_lo=slo, ci_hi=shi)

    per_series.close()
    per_window.close()
    log("EXP-05 done ->", per_series.path, "and", per_window.path)


if __name__ == "__main__":
    run()
