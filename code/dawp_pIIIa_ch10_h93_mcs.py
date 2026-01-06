#
# Heston (1993): Monte Carlo Valuation Building Blocks
# dawp_pIIIa_ch10_h93_mcs.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from __future__ import annotations

import math  # elementary math functions

import numpy as np  # numerical arrays
from numpy.random import default_rng  # random number generator


def h93_generate_paths_full_truncation(
    S0: float,
    v0: float,
    r: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int=10,
    antithetic: bool=True,
) -> tuple[np.ndarray, np.ndarray]:
    # Simulates (S_t, v_t) under Heston dynamics with full truncation Euler.
    dt = T / n_steps
    sdt = math.sqrt(dt)

    rng = default_rng(seed)

    if antithetic:
        n_half = (n_paths + 1) // 2
        Z = rng.standard_normal((2, n_steps, n_half))
        Z = np.concatenate((Z, -Z), axis=2)
        Z = Z[:, :, :n_paths]
    else:
        Z = rng.standard_normal((2, n_steps, n_paths))

    Z1, Z2 = Z[0], Z[1]
    # Correlates the Brownian motions dW^S and dW^v.
    Z2 = rho * Z1 + math.sqrt(1.0 - rho * rho) * Z2

    S = np.empty((n_steps + 1, n_paths), dtype=float)
    v = np.empty_like(S)
    S[0] = S0
    v[0] = v0

    for t in range(1, n_steps + 1):
        v_prev = np.maximum(0.0, v[t - 1])
        v[t] = v[t - 1] + kappa_v * (theta_v - v_prev) * dt
        v[t] += sigma_v * np.sqrt(v_prev) * sdt * Z2[t - 1]
        v[t] = np.maximum(0.0, v[t])

        expo = (r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * sdt * Z1[t - 1]
        S[t] = S[t - 1] * np.exp(expo)

    return S, v


def h93_call_value_mcs(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    v0: float,
    n_steps: int,
    n_paths: int,
    seed: int=10,
) -> float:
    # European call value via Monte Carlo simulation under the Heston model.
    S, _ = h93_generate_paths_full_truncation(
        S0,
        v0,
        r,
        kappa_v,
        theta_v,
        sigma_v,
        rho,
        T,
        n_steps,
        n_paths,
        seed=seed,
        antithetic=True,
    )
    payoff = np.maximum(S[-1] - K, 0.0)
    return float(math.exp(-r * T) * payoff.mean())
