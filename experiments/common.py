# -*- coding: utf-8 -*-
"""Shared machinery for the GMD experiments.

Every experiment writes its CSV row by row, so a run that is interrupted
still leaves usable results, and re-running skips the cases already on disk.
Set ``QUICK=1`` for a reduced version that finishes in minutes; that is what
the smoke test uses, and it is the first thing to run on a new machine.
"""

from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

DAY = 86400.0
HOUR = 3600.0

QUICK = os.environ.get("QUICK", "0") not in ("0", "", "false", "False")
RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))


# ----------------------------------------------------------------------
# swesphere compatibility
# ----------------------------------------------------------------------
def with_coriolis(grid, f_cor: np.ndarray):
    """``grid.with_coriolis(f)``, also on swesphere 0.4.0 which lacks it.

    The rotated test case needs the planetary rotation tilted along with
    everything else. swesphere gained ``SphereGrid.with_coriolis`` in 0.5.0;
    on 0.4.0 the grid is an ordinary dataclass, so ``replace`` does the same
    thing and these experiments run against the released package.
    """
    f = np.asarray(f_cor, dtype=float)
    if f.shape != np.shape(grid.sinL):
        raise ValueError(f"Coriolis field must have shape {np.shape(grid.sinL)}")
    if hasattr(grid, "with_coriolis"):
        return grid.with_coriolis(f)
    return replace(grid, f_cor=f)


#: Physical constants of the standard test-case literature. Williamson et al.
#: (1992) and Galewsky et al. (2004) both specify these, and they differ from
#: swesphere's defaults in the fourth digit (a = 6.371e6, g = 9.81). The
#: difference changes nothing physically, but a case run with other constants
#: is not the published case, and that is the kind of thing a reviewer checks.
WILLIAMSON = dict(a=6.37122e6, Omega=7.292e-5, g=9.80616)

#: Williamson test case 2 as specified: u0 = 2 pi a / (12 days) and
#: g h0 = 2.94e4 m2 s-2. swesphere's own defaults, U0 = 38 m/s and H0 = 2800 m,
#: are 1.6 and 6.6 percent away from these.
TC2_U0 = 2.0 * np.pi * WILLIAMSON["a"] / (12.0 * 86400.0)   # 38.611 m/s
TC2_H0 = 2.94e4 / WILLIAMSON["g"]                            # 2998.1 m


#: Which slice of a case list this process takes, from the environment.
#: ``SHARD_INDEX`` in ``0 .. SHARD_COUNT-1``.
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = max(1, int(os.environ.get("SHARD_COUNT", "1")))


def shard(cases: list) -> list:
    """The cases this process is responsible for.

    EXP-10 is 45 independent runs and about 39 hours in one process, which
    makes the sponge decision the slowest thing in the repository by a wide
    margin. The runs do not talk to each other, so there is no reason to do
    them one after another on a machine with cores to spare: four shards
    bring it to a morning.

    Round robin rather than contiguous blocks, because the cases are
    ordered by cost (resolution varies down the list) and contiguous blocks
    would hand one process all the expensive ones.

    Each shard appends to its own CSV, which `Table` already names after
    the shard, so two processes never write the same file.
    """
    if SHARD_COUNT == 1:
        return list(cases)
    return [c for i, c in enumerate(cases) if i % SHARD_COUNT == SHARD_INDEX]


# ----------------------------------------------------------------------
# Incremental CSV
# ----------------------------------------------------------------------
class Table:
    """A CSV that is flushed after every row and can resume.

    ``key_fields`` names the columns that identify a case. On a second run,
    cases already present are skipped, so a long experiment can be stopped
    and restarted without losing or repeating work.
    """

    def __init__(self, name: str, fields: List[str], key_fields: List[str]):
        RESULTS.mkdir(parents=True, exist_ok=True)
        # Per shard, so parallel processes never write the same file.
        # Merge them afterwards with
        #     python -c "import pandas,glob; pandas.concat(
        #         [pandas.read_csv(f) for f in glob.glob('results/NAME*.csv')]
        #         ).to_csv('results/NAME.csv', index=False)"
        suffix = "" if SHARD_COUNT == 1 else f"_shard{SHARD_INDEX}of{SHARD_COUNT}"
        self.path = RESULTS / f"{name}{suffix}.csv"
        self.fields = list(fields)
        self.key_fields = list(key_fields)
        self._done = set()
        write_header = not self.path.exists()
        if not write_header:
            with self.path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    self._done.add(self._key_of(row))
        self._fh = self.path.open("a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fields)
        if write_header:
            self._w.writeheader()
            self._fh.flush()

    def _key_of(self, row: Dict) -> tuple:
        return tuple(str(row[k]) for k in self.key_fields)

    def has(self, **key) -> bool:
        return tuple(str(key[k]) for k in self.key_fields) in self._done

    def add(self, **row) -> None:
        missing = set(self.fields) - set(row)
        if missing:
            raise KeyError(f"row is missing {sorted(missing)}")
        self._w.writerow({k: row[k] for k in self.fields})
        self._fh.flush()
        self._done.add(self._key_of({k: row[k] for k in self.key_fields}))

    def close(self) -> None:
        self._fh.close()


def log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


# ----------------------------------------------------------------------
# Diagnostics the experiments share
# ----------------------------------------------------------------------
def interior(field: np.ndarray, n_pole_rows: int) -> np.ndarray:
    """Drop the polar rows the sponge acts on.

    Raises rather than returning an empty array: a silently empty interior
    turns every later statistic into NaN, which is much harder to notice than
    a traceback. It happens as soon as the grid is coarse enough that the
    sponge covers it, which is easy to do by accident in a reduced run.
    """
    k = int(n_pole_rows)
    if k <= 0:
        return field
    if 2 * k >= field.shape[0]:
        raise ValueError(
            f"the sponge covers the whole grid: {field.shape[0]} rows with "
            f"{k} removed at each pole leaves nothing. Use a finer grid or "
            f"fewer polar rows."
        )
    return field[k:-k]


def l2_error(x: np.ndarray, ref: np.ndarray) -> float:
    """Normalized l2 error, the norm Williamson et al. report."""
    return float(np.sqrt(np.mean((x - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


def linf_error(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


def area_weights(grid) -> np.ndarray:
    w = np.abs(grid.cosL)
    return w / w.sum()


def observed_order(errors: Iterable[float], refinements: Iterable[float]) -> List[float]:
    """log2 of successive error ratios, for a factor-2 refinement."""
    e = list(errors)
    return [float(np.log2(e[i] / e[i + 1])) if e[i + 1] > 0 else float("nan")
            for i in range(len(e) - 1)]


def integral_autocorrelation_time(series: np.ndarray, max_lag: int = None) -> float:
    """Integrated autocorrelation time, in units of the sampling interval.

    The effective number of independent samples in a record of length ``n``
    is ``n / (2 tau + 1)``, not ``n``. Reporting a mean of a correlated series
    with ``sigma / sqrt(n)`` understates its uncertainty, which is the
    criticism the reviewers made of the stationarity claim.

    Summation is truncated at the first non-positive autocorrelation, the
    standard automatic window.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 4 or np.allclose(x, 0):
        return 0.0
    max_lag = min(max_lag or n // 2, n - 2)
    var = np.dot(x, x) / n
    if var <= 0:
        return 0.0
    tau = 0.0
    for lag in range(1, max_lag + 1):
        rho = np.dot(x[:-lag], x[lag:]) / (n * var)
        if rho <= 0.0:
            break
        tau += rho
    return float(tau)


def effective_sample_size(series: np.ndarray) -> float:
    tau = integral_autocorrelation_time(series)
    return float(len(series) / (2.0 * tau + 1.0))


def block_bootstrap_ci(series: np.ndarray, block: int = None, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0):
    """Confidence interval for the mean of a correlated series.

    A moving-block bootstrap: resample contiguous blocks long enough to carry
    the correlation, so the interval reflects the real information in the
    record rather than its raw length.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 4:
        return float("nan"), float("nan")
    if block is None:
        tau = integral_autocorrelation_time(x)
        block = int(max(2, min(n // 4, np.ceil(2.0 * (2.0 * tau + 1.0)))))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ----------------------------------------------------------------------
# Environment, so a runtime number means something
# ----------------------------------------------------------------------
def describe_environment() -> Dict[str, str]:
    """Everything needed to interpret a timing: reviewer 4's minor point 4."""
    import numpy

    def _cpu() -> str:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or "unknown"

    def _ver(mod: str) -> str:
        """Version of an installed distribution.

        The package metadata first, because a package is not obliged to carry
        a ``__version__`` attribute and pyteda 0.6.0 does not: asking for the
        attribute alone reported it as missing when it was installed and in
        use, which is exactly the kind of wrong line this table must not
        print. The attribute is the fallback, for anything installed outside
        a distribution.
        """
        try:
            return importlib.metadata.version(mod)
        except importlib.metadata.PackageNotFoundError:
            pass
        except Exception:
            pass
        try:
            return str(__import__(mod).__version__)
        except AttributeError:
            return "installed, version unknown"
        except ImportError:
            return "not installed"
        except Exception as exc:
            return f"unavailable ({type(exc).__name__})"

    def _git() -> str:
        """The commit these results came from.

        ``$GIT_COMMIT`` first: the Docker image copies ``experiments/`` and
        ``figures/`` and not ``.git``, so inside the container there is no
        repository to ask and ``git rev-parse`` would always say "unknown".
        Compose forwards the variable from the host, and the README says to
        export it.
        """
        env = os.environ.get("GIT_COMMIT", "").strip()
        if env:
            return env
        try:
            out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            return out or "unknown"
        except Exception:
            return "unknown"

    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 0

    return {
        "cpu": _cpu(),
        "cores_available": str(n_cpu),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": _ver("scipy"),
        "pyshtools": _ver("pyshtools"),
        "swesphere": _ver("swesphere"),
        "pyteda": _ver("pyteda"),
        "blas_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "commit": _git(),
    }
