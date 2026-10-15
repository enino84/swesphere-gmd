# -*- coding: utf-8 -*-
"""EXP-02. A non-zonal, time-dependent benchmark: the Rossby-Haurwitz wave.

The second half of reviewer 2's first objection. Test case 2, even rotated,
is steady: it asks whether the model can leave a solution alone. Williamson
et al.'s test case 6 asks something harder, whether the model can carry a
structure. A wavenumber-4 pattern propagates eastward and should keep its
shape, at an angular rate that is known in closed form,

    nu = (R (3 + R) omega - 2 Omega) / ((1 + R) (2 + R)),

about 2.46e-6 rad/s for the standard R = 4, a pattern that circles the globe
in roughly 29.5 days. So there are two independent things to measure, and a
model can get one right and the other wrong:

  * **phase**: where the wave is after 14 days, against the analytic rate.
    Measured from the complex phase of zonal Fourier mode 4 of the depth
    field, averaged over the latitudes where the wave has amplitude, which is
    far more precise than tracking a maximum on the grid.
  * **shape**: how much of the pattern survives. The initial field is shifted
    by the measured phase and correlated with the final one, so a wave that
    arrives in the right place but has been smeared by the filter scores low.

The shallow-water Rossby-Haurwitz wave is not an exact solution, it breaks
down after several weeks even in reference models, so 14 days is the window
where a disagreement is the model's fault rather than physics.
"""

from __future__ import annotations

import time

import numpy as np
from swesphere import SWEModel, diagnostics

from common import WILLIAMSON, DAY, QUICK, Table, interior, log
from testcases import rh_angular_phase_speed, rossby_haurwitz

#: Polar rows the sponge acts on. The model's own value is used for
#: scoring; this is what the model is built with.
POLE_ROWS = 10

RESOLUTIONS = [32] if QUICK else [32, 64, 96]
DAYS = 0.5 if QUICK else 14.0
R = 4


def wave_phase(h: np.ndarray, grid, R: int = R,
               pole_rows: int = POLE_ROWS) -> float:
    """Longitude of the wave crest, from zonal Fourier mode ``R``.

    Averaged over latitude with the amplitude as weight, so quiet rows do not
    add noise, and unwrapped against the first weighted row.
    """
    coef = np.fft.rfft(h, axis=1)[:, R]
    k = int(pole_rows)
    coef = coef[k:-k]
    w = np.abs(coef)
    if w.sum() <= 0:
        return float("nan")
    # Average the unit vectors, not the angles, so the mean is circular.
    z = (coef / np.where(w > 0, w, 1.0) * w).sum() / w.sum()
    # A pattern cos(R(lam - lam0)) has Fourier phase -R*lam0, hence the sign.
    return float(-np.angle(z) / R)


def run() -> None:
    table = Table(
        "exp02_rossby_haurwitz",
        fields=["LMAX", "Nlat", "dt", "days", "phase_measured_deg",
                "phase_analytic_deg", "phase_error_deg", "speed_ratio",
                "shape_correlation", "l2_h_vs_shifted", "amplitude_ratio",
                "mass_drift", "energy_drift", "enstrophy_drift", "seconds"],
        key_fields=["LMAX", "days"],
    )
    for L in RESOLUTIONS:
        if table.has(LMAX=L, days=DAYS):
            log(f"skip LMAX={L}")
            continue
        dt = 120.0 * (32.0 / L)
        log(f"LMAX={L} dt={dt:.0f}s for {DAYS} days")
        t0 = time.time()

        model = SWEModel(LMAX=L, dt=dt, n_pole_rows=POLE_ROWS, **WILLIAMSON)
        u0, v0, h0 = rossby_haurwitz(model.grid, R=R)
        x0 = model.pack(u0, v0, h0)
        inv0 = diagnostics.invariants(model, x0)

        n_steps = int(round(DAYS * DAY / dt))
        x = model.propagate(x0, [0.0, n_steps * dt])
        inv1 = diagnostics.invariants(model, x)
        u, v, h = model.unpack(x)

        nu = rh_angular_phase_speed(model.grid, R=R)
        t_end = n_steps * dt
        phase_analytic = nu * t_end
        d_phase = wave_phase(h, model.grid) - wave_phase(h0, model.grid)
        # Unwrap into the branch nearest the analytic prediction.
        period = 2.0 * np.pi / R
        d_phase += period * np.round((phase_analytic - d_phase) / period)

        # Shape: shift the initial field by the measured phase and compare.
        shift = d_phase * model.Nlon / (2.0 * np.pi)
        coef0 = np.fft.rfft(h0, axis=1)
        kx = np.arange(coef0.shape[1])
        h0_shifted = np.fft.irfft(
            coef0 * np.exp(-2j * np.pi * kx[None, :] * shift / model.Nlon),
            n=model.Nlon, axis=1)
        a, b = interior(h, model.n_pole_rows).ravel(), interior(h0_shifted, model.n_pole_rows).ravel()
        corr = float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])

        amp = lambda f: float(np.abs(np.fft.rfft(interior(f, model.n_pole_rows), axis=1)[:, R]).mean())

        table.add(
            LMAX=L, Nlat=model.Nlat, dt=dt, days=DAYS,
            phase_measured_deg=np.degrees(d_phase),
            phase_analytic_deg=np.degrees(phase_analytic),
            phase_error_deg=np.degrees(d_phase - phase_analytic),
            speed_ratio=d_phase / phase_analytic if phase_analytic else float("nan"),
            shape_correlation=corr,
            l2_h_vs_shifted=float(np.sqrt(np.mean((a - b) ** 2))
                                  / np.sqrt(np.mean(b ** 2))),
            amplitude_ratio=amp(h) / amp(h0),
            mass_drift=abs(inv1["mass"] / inv0["mass"] - 1.0),
            energy_drift=abs(inv1["energy"] / inv0["energy"] - 1.0),
            enstrophy_drift=abs(inv1["enstrophy"] / inv0["enstrophy"] - 1.0),
            seconds=round(time.time() - t0, 1),
        )
    table.close()
    log("EXP-02 done ->", table.path)


if __name__ == "__main__":
    run()
