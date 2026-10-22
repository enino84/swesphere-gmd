# swesphere-gmd

The experiments for the GMD model description paper on `swesphere`.

Each one answers a specific objection from the SoftwareX review, and the
mapping is deliberate: the reviewers did not ask for a different paper, they
asked for the verification a model description paper is expected to carry.

| Experiment | Answers | Cost |
|---|---|---|
| `exp01_convergence` | R2.1, R4-m1: does it converge in the configuration people run? | hours |
| `exp02_rossby_haurwitz` | R2.1: a non-zonal, time-dependent benchmark | hours |
| `exp03_galewsky` | R2.1: what the filter costs the instability | hours |
| `exp04_regimes` | R2.2: several independent realizations per regime | **longest, start first** |
| `exp05_stationarity` | R2.2, R4.3: uncertainty with the correlation accounted for | seconds |
| `exp06_stability` | R4-m3: per-preset step limits, stable vs accurate | hours |
| `exp07_filter_protocol` | R2.3: separating the integrator from the dissipation | hours |
| `exp08_assimilation` | R2.4: a worked algorithm experiment | hours |
| `exp09_environment` | R4-m4: what the cost figure means | seconds |
| `exp10_sponge_necessity` | not asked for: is the polar sponge needed at all? | hours |
| `exp11_gallery` | the model states behind every field figure | hours |

R2 is reviewer 2, R4-m1 is reviewer 4's first minor point, and so on.

## Running

```bash
git clone <this repo> && cd swesphere-gmd
export GIT_COMMIT=$(git rev-parse --short HEAD)   # recorded in every result
docker compose build
docker compose run --rm smoke          # QUICK=1, everything, a few minutes
docker compose run -d exp04            # start the long one first
docker compose run -d exp01
docker compose run -d exp06
```

`GIT_COMMIT` is not decoration. The image copies `experiments/` and
`figures/` and not `.git`, so inside the container there is nothing to ask,
and without the variable EXP-09 records the commit as `unknown`. For a paper
whose claim is that the numbers are reproducible, that column has to be
right.

The smoke service has been run end to end against the released
`swesphere 0.5.0` and `pyteda 0.6.0`: all ten experiments complete, two
cores, about five minutes, with `exp08` taking a minute of it and everything
else seconds.

`exp05` reads the records `exp04` writes, so run it afterwards, and re-run it
as often as you like while `exp04` is still going: it costs seconds and the
intervals tighten as the records grow.

Every script honours `QUICK=1` for a reduced version, which is what `smoke`
runs. **Run the smoke service on a new machine before starting anything
long.** Results go to `./results` as CSV, flushed after every row, so an
interrupted run keeps what it had and a restart skips the finished cases.

## Running everything (swesphere 0.6.0)

One script, nothing to run by hand afterwards:

    git pull
    nohup ./run_all.sh > run_all.log 2>&1 & disown
    tail -f run_all.log

It rebuilds the images, removes the results that 0.6.0 invalidated, and runs
the campaign in three stages: exp04 in four shards alongside exp06, exp07 and
exp11; then exp05 and exp08 in eight shards, which cannot start earlier
because exp08 draws its ensemble from exp04's records; then exp09 alone, so
its cost measurement is not taken while seven containers compete, and the
figures. About four to six hours on eight cores.

**exp01, exp02, exp03 and exp10 are not rerun.** They pass `n_pole_rows`
explicitly at every call, or sweep it deliberately, so the new default does
not change their numbers; rerunning them would cost fifteen hours to
reproduce the same CSVs.

## How long it all takes

Measured cost per model day at one BLAS thread on a 2.8 GHz Xeon core, times
the model days each experiment asks for. Your machine will differ; the
*ratios* are what matters when deciding what to parallelise.

| experiment | hours |
|---|---|
| `exp10_sponge_necessity` | **39** |
| `exp04_regimes` | 7.0 |
| `exp08_assimilation` | 3.9 |
| `exp11_gallery` | 1.5 |
| `exp01_convergence` | 1.0 |
| `exp03_galewsky` | 0.5 |
| `exp07_filter_protocol` | 0.4 |
| `exp02`, `exp06` | 0.3 each |
| `exp05`, `exp09` | seconds |
| **total, one at a time** | **54 h** |

exp10 alone is 72 % of it, and running the others alongside it does not help:
the wall clock is whatever exp10 takes. So shard it.

    for i in 0 1 2 3; do
      SHARD_INDEX=$i SHARD_COUNT=4 docker compose run -d --name exp10-$i exp10
    done

Four shards bring exp10 to about ten hours and the whole set to roughly
twelve. Each shard writes its own CSV; merge them when they are done:

    python -c "import pandas,glob; pandas.concat([pandas.read_csv(f) \
      for f in glob.glob('results/exp10_sponge_necessity_shard*.csv')]) \
      .to_csv('results/exp10_sponge_necessity.csv', index=False)"

Do not start more containers than the machine has physical cores. Every
experiment reports a cost per model day, and a timing taken while the cores
were oversubscribed compares with nothing.

## Figures

Two services, and the split matters.

    docker compose run -d exp11      # integrates, hours, writes .npy
    docker compose run --rm gallery  # draws, seconds, writes .pdf and .png

`exp11` re-runs cases the other experiments already cover and saves the raw
`(u, v, h)` states to `results/snapshots/`. `gallery` reads those and draws.
Nothing is integrated while drawing, so a figure can be restyled twenty times
for free, which is what actually happens to a figure between a draft and a
published paper.

`figures` is the other half: the line plots and tables, straight from the
CSVs.

Five field figures come out of `gallery`:

| figure | what it shows |
|---|---|
| `fig_rotated_tc2` | the exact steady solution and the error after five days, at three rotation angles, on the sphere. The paper's central claim in one picture |
| `fig_galewsky` | day-6 relative vorticity against the published reference |
| `fig_regimes` | what the three forced regimes look like |
| `fig_sponge_poles` | the north pole with the sponge and without it, **on one shared colour scale** |
| `fig_rh_propagation` | the Rossby-Haurwitz wave travelling, against its analytic phase speed |

Resolutions are overridable, because this is the one experiment whose cost is
a choice rather than a requirement:

    GALEWSKY_LMAX=64 docker compose run -d exp11   # 1 minute instead of 11

The defaults are what we would submit: LMAX 128 for Galewsky, 64 for the
rest.

On the colour maps, since a reviewer looks before reading. Signed fields get
a two-hue diverging map through a neutral grey; unsigned magnitudes get one
hue from light to dark. Nothing is a rainbow, and the categorical colours in
the line plots were checked for colour vision deficiency rather than chosen
by eye.

## Constants

The verification experiments (exp01, exp02, exp03) build the model with the
constants the test-case literature specifies, `a = 6.37122e6 m`,
`g = 9.80616 m/s2`, `Omega = 7.292e-5 s-1`, and run test case 2 at its
specified `u0 = 2 pi a / (12 days) = 38.611 m/s` and
`g h0 = 2.94e4 m2 s-2`, that is `h0 = 2998.1 m`. These differ from swesphere's
own defaults (`a = 6.371e6`, `g = 9.81`, `U0 = 38`, `H0 = 2800`) in the fourth
digit and by 1.6 and 6.6 percent respectively. Nothing physical changes, since
the flow is an exact steady solution for any `U0` and `H0`, but a case run with
other constants is not the published case. They live in `common.WILLIAMSON`,
`common.TC2_U0` and `common.TC2_H0`.

The characterization experiments (exp04 onward) keep swesphere's defaults,
because they characterize the package as users get it.

## The test cases

`experiments/testcases.py` re-exports two standard test cases from
`swesphere` 0.5.0, which is where they were added for this paper, and keeps a
local copy as a fallback for 0.4.0. The experiments therefore run released
code, which is the point: a reviewer can reproduce the numbers from PyPI.

The first is **test case 2 at a rotation angle**. Up to 0.4.0 swesphere
shipped it only in the grid-aligned form, which is the easy case: the
solution is zonally symmetric, every meridional derivative is taken along a
grid line, and the polar caps carry no flow. That is precisely the
configuration in which this discretization looks best, and it is what the
submitted manuscript measured.

There is a subtlety worth stating, because it is the reason the model needed
a change. The rotated test does not tilt the flow and leave the planet alone.
It tilts the whole problem, planetary rotation included, so it prescribes its
own Coriolis field, `f = 2 Omega sin(phi')` with `phi'` the latitude in the
tilted frame. With the untilted `f` the tilted flow is not a steady solution
of anything: the balance

    dB/dlam = a cos(phi) (f + zeta) v,    dB/dphi = -a (f + zeta) u

fails its own integrability condition by `-2 Omega U0 a sin(alpha) sin(lam)
cos^2(phi)`, so **no height field whatsoever holds it steady**. Running the
test that way would measure the drift of a problem that was never in balance
and call it truncation error. Verified with sympy; with the tilted `f` both
residuals are identically zero.

So the model has to accept a Coriolis field. swesphere gained
`SphereGrid.with_coriolis` and `SWEModel.set_coriolis` in 0.5.0 for this, and
`presets.williamson_tc2(alpha_deg=...)` installs the right field so the
mistake cannot be made; `common.with_coriolis` falls back to
`dataclasses.replace` on 0.4.0.

The second is the **Rossby-Haurwitz wave**, test case 6: non-zonal and
time-dependent, with an analytic phase speed, so it tests where the wave is
as well as whether it survived.

## Two early results

**The sponge does not converge, and the manuscript's convergence was partly an
artifact.** `n_pole_rows` counts rows, not degrees, so the same default of 10
covers everything poleward of 37 degrees at LMAX 16, 63 at LMAX 32 and 83 at
LMAX 128. Refining the grid was quietly retreating the sponge. `exp01`
therefore runs two sweeps. Observed order of the normalized l2 error of the
depth, in `QUICK` mode:

| | no sponge | sponge, rows fixed | sponge, latitude fixed |
|---|---|---|---|
| alpha = 0 | 2.05 | 2.39 | 0.32 |
| alpha = 90 | 1.21 | 0.52 | 0.06 |

The last column is the honest one: hold the sponge over the same band and
refining the grid buys almost nothing, because damping the wind in a fixed
region is an order-one perturbation that does not shrink. The middle column
is what the manuscript measured.

**The sponge may not be necessary.** `two_jets` completes 20 days with no
sponge at all, at LMAX 32 and 64, at the same cost per model day and with a
slightly larger peak wind. But it is not free of consequences: the amplitude
of the shortest resolved zonal waves in the polar rows goes from 0.02 to 2.8
times its midlatitude value. The run survives and the poles fill with
grid-scale noise. Whether that matters depends on whether the polar band is
used, which for an assimilation testbed scored on `interior_mask()` it is
not. `exp10` measures this across presets, resolutions and sponge widths so
the decision comes from data.

## Notes

- One BLAS thread by default, set in the image. Every experiment reports a
  cost per model day, and a number measured while the library quietly used
  every core compares with nothing. Override if you want speed rather than a
  reportable timing.
- `exp08` needs `pyteda[swe]`, which `requirements.txt` installs. In the
  smoke run its `incr_wind` column is **identically zero** for every
  `cross=0` row and nonzero for every `cross=1` row, which is the whole point
  of the cross-variable work: without it a localized filter observing only
  the depth leaves the wind untouched. Watch the size of that increment in
  the full run. At QUICK settings `enkf-modified-cholesky` with `cross=1`
  moves the wind by 2e-3 in the first cycle and then by almost nothing, which
  may be the 20-member, LMAX-8 setup rather than the method.
- `exp04` writes the records as float32 `.npy` next to the CSV. They are the
  expensive part; keep them.
