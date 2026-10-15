# -*- coding: utf-8 -*-
"""The standard test cases of Williamson et al. (1992).

They live in ``swesphere`` from 0.5.0 on, and this module re-exports them so
the experiments run the released code rather than a copy of it, which is what
makes the paper's numbers reproducible from PyPI alone. The local
implementations below are kept as a fallback for 0.4.0, which was the newest
release when these experiments were written; they are the same code that was
contributed upstream.

The released 0.4.0 package implements test case 2 only in its grid-aligned
form, with the flow axis on the rotation axis. That is the easy case: the
solution is zonally symmetric, every meridional derivative is taken along a
grid line, and the polar caps carry no flow. A solver can look second-order
accurate on it and still be wrong where the flow crosses a pole.

Two cases fix that:

``williamson_tc2(grid, alpha_deg=...)``
    The same steady solid-body rotation, but with the flow axis tilted by
    ``alpha`` from the rotation axis. At 90 degrees the flow runs straight
    over both poles, through the sponge, which is the hardest configuration
    for this discretization and the one the reviewers asked for. The solution
    stays exact and steady for every alpha, so the error after N days is
    entirely numerical. The test tilts the planet too, so it prescribes its
    own Coriolis field: see ``tc2_coriolis``.

``rossby_haurwitz(grid)``
    Test case 6, a wavenumber-4 pattern that is neither zonal nor steady. It
    propagates eastward without changing shape at an angular rate that is
    known analytically, so it tests phase as well as amplitude.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["tc2_sin_rotated_latitude", "tc2_coriolis", "williamson_tc2",
           "rossby_haurwitz", "rh_angular_phase_speed"]

try:                                    # swesphere >= 0.5.0
    from swesphere.initial_conditions import (  # noqa: F401
        rh_angular_phase_speed, rossby_haurwitz, tc2_coriolis,
        tc2_sin_rotated_latitude, williamson_tc2)
    FROM_PACKAGE = True
except ImportError:                     # swesphere 0.4.0, the local copies
    FROM_PACKAGE = False


if not FROM_PACKAGE:
    def tc2_sin_rotated_latitude(grid, alpha_deg: float) -> np.ndarray:
        """``sin`` of the latitude measured in the frame tilted by ``alpha_deg``."""
        a = np.radians(float(alpha_deg))
        return (-np.cos(grid.LON_ns) * grid.cosL * np.sin(a)
                + grid.sinL * np.cos(a))


    def tc2_coriolis(grid, alpha_deg: float = 0.0) -> np.ndarray:
        """The Coriolis field test case 2 prescribes at angle ``alpha_deg``.

        ``f = 2 Omega sin(phi')``, the planetary rotation tilted with everything
        else. This is not a detail: with the untilted ``f = 2 Omega sin(phi)``
        the tilted flow below is **not** a steady solution, and not because of any
        discretization. The balance

            dB/dlam = a cos(phi) (f + zeta) v,    dB/dphi = -a (f + zeta) u

        then fails its own integrability condition by
        ``-2 Omega U0 a sin(alpha) sin(lam) cos^2(phi)``, so no height field
        whatsoever balances it. Checked with sympy; with the tilted ``f`` both
        residuals are identically zero.

        Pass the result to ``grid.with_coriolis(...)``, which needs
        swesphere >= 0.5.0.
        """
        return 2.0 * grid.Omega * tc2_sin_rotated_latitude(grid, alpha_deg)


    def williamson_tc2(
        grid,
        U0: float = 38.0,
        H0: float = 2800.0,
        alpha_deg: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Steady solid-body rotation at an angle ``alpha_deg`` to the pole.

        Williamson et al. (1992), test case 2. With ``alpha_deg = 0`` this
        reproduces, to the bit, the zonally symmetric case swesphere ships.

            u  =  U0 (cos phi cos a + sin phi cos lam sin a)
            v  = -U0 sin lam sin a
            gh = gH0 - (a Omega U0 + U0^2 / 2) sin^2(phi')

        Exact and steady for every ``alpha``, **provided the model uses the
        Coriolis field of** :func:`tc2_coriolis`. Whatever the model then does to
        it over a few days is error and nothing else.
        """
        a = np.radians(float(alpha_deg))
        lam = grid.LON_ns
        cos_phi, sin_phi = grid.cosL, grid.sinL

        u = U0 * (cos_phi * np.cos(a) + sin_phi * np.cos(lam) * np.sin(a))
        v = -U0 * np.sin(lam) * np.sin(a)

        rot = tc2_sin_rotated_latitude(grid, alpha_deg)
        h = H0 - (grid.a * grid.Omega * U0 + 0.5 * U0 ** 2) / grid.g * rot ** 2
        return u, v, h


    def rossby_haurwitz(
        grid,
        R: int = 4,
        omega: float = 7.848e-6,
        K: float = 7.848e-6,
        h0: float = 8000.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Williamson et al. (1992), test case 6: a Rossby-Haurwitz wave.

        A wavenumber-``R`` pattern superposed on a zonal flow. In the
        non-divergent barotropic limit it propagates eastward without changing
        shape; in the shallow-water system it is not exact, but it holds its form
        for weeks and its phase speed is close to the analytic value, which is
        what :func:`rh_angular_phase_speed` returns.

        The defaults are the ones the test case prescribes.
        """
        lam, phi = grid.LON_ns, grid.LAT_ns
        c, s = grid.cosL, grid.sinL
        a, Om, g = grid.a, grid.Omega, grid.g

        u = a * omega * c + a * K * c ** (R - 1) * (R * s ** 2 - c ** 2) * np.cos(R * lam)
        v = -a * K * R * c ** (R - 1) * s * np.sin(R * lam)

        # A, B, C of the test case. The cos^(-2) term of A is folded into the
        # power so nothing is divided by cos(phi) at the poles.
        A = (
            0.5 * omega * (2.0 * Om + omega) * c ** 2
            + 0.25 * K ** 2 * c ** (2 * R) * ((R + 1) * c ** 2 + (2 * R ** 2 - R - 2))
            - 0.5 * K ** 2 * R ** 2 * c ** (2 * R - 2)
        )
        B = (
            2.0 * (Om + omega) * K / ((R + 1) * (R + 2))
            * c ** R
            * ((R ** 2 + 2 * R + 2) - (R + 1) ** 2 * c ** 2)
        )
        C = 0.25 * K ** 2 * c ** (2 * R) * ((R + 1) * c ** 2 - (R + 2))

        h = h0 + (a ** 2 / g) * (A + B * np.cos(R * lam) + C * np.cos(2 * R * lam))
        return u, v, h


    def rh_angular_phase_speed(grid, R: int = 4, omega: float = 7.848e-6) -> float:
        """Analytic eastward angular phase speed of the Rossby-Haurwitz wave.

            nu = (R (3 + R) omega - 2 Omega) / ((1 + R)(2 + R))

        in radians per second. For the standard R = 4 this is about
        2.46e-6 rad/s, a pattern that circles the globe in roughly 29.5 days.
        """
        return (R * (3.0 + R) * omega - 2.0 * grid.Omega) / ((1.0 + R) * (2.0 + R))
