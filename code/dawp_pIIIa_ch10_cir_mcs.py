#
# Cox-Ingersoll-Ross (1985): Simulation and Monte Carlo Valuation
# dawp_pIIIa_ch10_cir_mcs.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from __future__ import annotations

import math  # elementary math functions

import numpy as np  # numerical arrays
from numpy.random import default_rng  # random number generator


def cir_generate_paths_exact(
    r0: float,
    kappa_r: float,
    theta_r: float,
    sigma_r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int=10,
) -> np.ndarray:
    # Simulates the CIR short-rate process using its exact transition law.
    dt = T / n_steps  # time step length
    rng = default_rng(seed)  # random number generator

    r = np.empty((n_steps + 1, n_paths), dtype=float)  # path matrix
    r[0] = r0  # initial short rate

    d = 4.0 * kappa_r * theta_r / (sigma_r * sigma_r)  # degrees of freedom
    # Scale parameter of the chi-square representation.
    c = (sigma_r * sigma_r * (1.0 - math.exp(-kappa_r * dt))) / (4.0 * kappa_r)
    exp_kdt = math.exp(-kappa_r * dt)  # mean reversion factor per step
    denom = sigma_r * sigma_r * (1.0 - exp_kdt)  # denominator for non-centrality

    for t in range(1, n_steps + 1):
        # Non-centrality parameter per path.
        lam = 4.0 * kappa_r * exp_kdt * r[t - 1] / denom
        chi = rng.noncentral_chisquare(d, lam, size=n_paths)
        r[t] = c * chi

    return r


def cir_generate_paths_full_truncation(
    r0: float,
    kappa_r: float,
    theta_r: float,
    sigma_r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int=10,
) -> np.ndarray:
    # Simulates the CIR short-rate process via the full truncation Euler scheme.
    dt = T / n_steps  # time step length
    sdt = math.sqrt(dt)  # square root of dt
    rng = default_rng(seed)  # random number generator

    r = np.empty((n_steps + 1, n_paths), dtype=float)  # path matrix
    rh = np.empty_like(r)  # helper process (can become negative)
    r[0] = r0  # initial short rate
    rh[0] = r0

    Z = rng.standard_normal((n_steps, n_paths))  # Gaussian shocks

    for t in range(1, n_steps + 1):
        rt = np.maximum(0.0, rh[t - 1])  # truncated rate
        rh[t] = rh[t - 1] + kappa_r * (theta_r - rt) * dt
        rh[t] += sigma_r * np.sqrt(rt) * sdt * Z[t - 1]
        r[t] = np.maximum(0.0, rh[t])

    return r


def zcb_value_mcs(
    r_paths: np.ndarray,
    T: float,
) -> np.ndarray:
    # Returns Monte Carlo estimates B_t(T) for all grid times t along the simulation.
    n_steps = r_paths.shape[0] - 1
    dt = T / n_steps  # time step length

    zcb = np.ones_like(r_paths)  # initialize with terminal value 1.0
    for t in range(n_steps, 0, -1):
        disc = np.exp(-0.5 * (r_paths[t] + r_paths[t - 1]) * dt)
        zcb[t - 1] = zcb[t] * disc

    return zcb.mean(axis=1)
