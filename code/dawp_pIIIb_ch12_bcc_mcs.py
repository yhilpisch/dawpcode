#
# Monte Carlo Simulation and LSM in the BCC (1997) Model
# dawp_pIIIb_ch12_bcc_mcs.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from __future__ import annotations

import math  # elementary math functions
from typing import Dict, Tuple  # type hints

import numpy as np  # numerical arrays

from numpy.random import default_rng  # random number generator

from dawp_pIIIa_ch09_cir_zcb import cir_zcb_value  # CIR discount factors


def generate_cholesky(rho: float) -> np.ndarray:
    """Generates the Cholesky matrix for correlated BCC innovations."""
    rho_rs = 0.0
      # correlation between index level and short rate
    cov = np.zeros((4, 4), dtype=float)
    cov[0] = [1.0, rho_rs, 0.0, 0.0]
    cov[1] = [rho_rs, 1.0, rho, 0.0]
    cov[2] = [0.0, rho, 1.0, 0.0]
    cov[3] = [0.0, 0.0, 0.0, 1.0]
    return np.linalg.cholesky(cov)


def random_number_generator(
    M: int,
    I: int,
    anti_paths: bool,
    moment_matching: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generates pseudo-random numbers for the BCC simulation engine."""
    if anti_paths:
        rand = rng.standard_normal((4, M + 1, int(I / 2)))
        rand = np.concatenate((rand, -rand), axis=2)
    else:
        rand = rng.standard_normal((4, M + 1, I))

    if moment_matching:
        for a in range(4):
            arr = rand[a]
            arr = arr / np.std(arr)
            arr = arr - np.mean(arr)
            rand[a] = arr
    return rand


def srd_generate_paths(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    M: int,
    I: int,
    rand: np.ndarray,
    row: int,
    cho_matrix: np.ndarray,
) -> np.ndarray:
    """Simulates square-root diffusion paths via full truncation Euler.

    The function can be used for both the short-rate and variance process.
    """
    dt = T / M
    sdt = math.sqrt(dt)

    x = np.zeros((M + 1, I), dtype=float)
    x[0] = x0
    xh = np.zeros_like(x)
    xh[0] = x0

    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        prev = np.maximum(0.0, xh[t - 1])
        drift = kappa * (theta - prev) * dt
        diff = sigma * np.sqrt(prev) * ran[row] * sdt
        xh[t] = xh[t - 1] + drift + diff
        x[t] = np.maximum(0.0, xh[t])

    return x


def b96_generate_paths(
    S0: float,
    r: np.ndarray,
    v: np.ndarray,
    lamb: float,
    mu_j: float,
    delta: float,
    rand: np.ndarray,
    row1: int,
    row2: int,
    cho_matrix: np.ndarray,
    T: float,
    M: int,
    I: int,
    moment_matching: bool,
) -> np.ndarray:
    """Simulates the Bates (1996) index process in the BCC framework."""
    S = np.zeros((M + 1, I), dtype=float)
    S[0] = S0

    dt = T / M
    sdt = math.sqrt(dt)

    ran_poi = np.random.poisson(lamb * dt, size=(M + 1, I))
      # jump counts per step and path

    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t, :])

        if moment_matching:
            bias = np.mean(np.sqrt(v[t]) * ran[row1] * sdt)
        else:
            bias = 0.0

        diffusion = ((r[t] + r[t - 1]) * 0.5 - 0.5 * v[t]) * dt
        diffusion += np.sqrt(v[t]) * ran[row1] * sdt - bias

        jumps = (np.exp(mu_j + delta * ran[row2]) - 1.0) * ran_poi[t]

        S[t] = S[t - 1] * (np.exp(diffusion) + jumps)

    return S


def simulate_bcc_paths(
    S0: float,
    r0: float,
    kappa_r: float,
    theta_r: float,
    sigma_r: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    v0: float,
    lamb: float,
    mu_J: float,
    delta: float,
    T: float,
    M: int,
    I: int,
    anti_paths: bool=True,
    moment_matching: bool=True,
    seed: int | None=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generates BCC paths for index level, short rate, and variance."""
    rng = default_rng(seed)
    cho_matrix = generate_cholesky(rho)
    rand = random_number_generator(M, I, anti_paths, moment_matching, rng)

    r = srd_generate_paths(
        x0=r0,
        kappa=kappa_r,
        theta=theta_r,
        sigma=sigma_r,
        T=T,
        M=M,
        I=I,
        rand=rand,
        row=0,
        cho_matrix=cho_matrix,
    )

    v = srd_generate_paths(
        x0=v0,
        kappa=kappa_v,
        theta=theta_v,
        sigma=sigma_v,
        T=T,
        M=M,
        I=I,
        rand=rand,
        row=2,
        cho_matrix=cho_matrix,
    )

    S = b96_generate_paths(
        S0=S0,
        r=r,
        v=v,
        lamb=lamb,
        mu_j=mu_J,
        delta=delta,
        rand=rand,
        row1=1,
        row2=3,
        cho_matrix=cho_matrix,
        T=T,
        M=M,
        I=I,
        moment_matching=moment_matching,
    )

    dt = T / M
    return S, r, v, dt


def _discount_factors_from_short_rate(r: np.ndarray, dt: float) -> np.ndarray:
    """Computes Monte Carlo discount factors from short-rate paths."""
    avg_r = 0.5 * (r[1:] + r[:-1])
    integral = np.cumsum(avg_r * dt, axis=0)
    B = np.exp(-integral)
    return B


def european_call_mcs(
    params: Dict[str, float],
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    anti_paths: bool=True,
    moment_matching: bool=True,
    seed: int | None=None,
) -> float:
    """Values a European call option in the BCC model via Monte Carlo."""
    S, r, v, dt = simulate_bcc_paths(
        S0=params["S0"],
        r0=params["r0"],
        kappa_r=params["kappa_r"],
        theta_r=params["theta_r"],
        sigma_r=params["sigma_r"],
        kappa_v=params["kappa_v"],
        theta_v=params["theta_v"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        lamb=params["lamb"],
        mu_J=params["mu_J"],
        delta=params["delta"],
        T=T,
        M=n_steps,
        I=n_paths,
        anti_paths=anti_paths,
        moment_matching=moment_matching,
        seed=seed,
    )

    h = np.maximum(S[-1] - K, 0.0)
      # inner values at maturity

    # Analytical CIR discount factor B_0(T) for consistency with the
    # transform-based valuations that rely on the same short-rate
    # parameters.
    B0T = cir_zcb_value(
        params["r0"],
        params["kappa_r"],
        params["theta_r"],
        params["sigma_r"],
        T=T,
        t=0.0,
    )
    V0 = float(B0T * np.mean(h))
    return V0


def american_put_lsm(
    params: Dict[str, float],
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    basis_degree: int=3,
    anti_paths: bool=True,
    moment_matching: bool=True,
    seed: int | None=None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, float]:
    """Values an American put option via LSM in the BCC model.

    The basis consists of polynomial functions in the state variables
    (index level, variance, short rate) up to the specified degree.
    """
    S, r, v, dt = simulate_bcc_paths(
        S0=params["S0"],
        r0=params["r0"],
        kappa_r=params["kappa_r"],
        theta_r=params["theta_r"],
        sigma_r=params["sigma_r"],
        kappa_v=params["kappa_v"],
        theta_v=params["theta_v"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        lamb=params["lamb"],
        mu_J=params["mu_J"],
        delta=params["delta"],
        T=T,
        M=n_steps,
        I=n_paths,
        anti_paths=anti_paths,
        moment_matching=moment_matching,
        seed=seed,
    )

    M = n_steps
    I = n_paths
    h = np.maximum(K - S, 0.0)
      # inner value matrix
    V = h.copy()
      # value/cash flow matrix
    ex = np.zeros_like(V)
      # exercise matrix

    D = 10
      # number of regression coefficients (including constant)
    rg = np.zeros((M + 1, D + 1), dtype=float)
      # regression coefficient matrix

    for t in range(M - 1, 0, -1):
        df = np.exp(-(r[t] + r[t + 1]) * 0.5 * dt)
          # discount factors between t and t+1

        itm = h[t] > 0.0
          # in-the-money paths at time t
        if not np.any(itm):
            continue

        idx = np.where(itm)[0]
        rel_S = S[t, idx]
        rel_v = v[t, idx]
        rel_r = r[t, idx]
        rel_V = V[t + 1, idx] * df[idx]

        X = np.zeros((D + 1, len(idx)), dtype=float)
          # design matrix (features in rows)
        X[10] = rel_S * rel_v * rel_r
        X[9] = rel_S * rel_v
        X[8] = rel_S * rel_r
        X[7] = rel_v * rel_r
        X[6] = rel_S ** 2
        X[5] = rel_v ** 2
        X[4] = rel_r ** 2
        X[3] = rel_S
        X[2] = rel_v
        X[1] = rel_r
        X[0] = 1.0

        coef, *_ = np.linalg.lstsq(X.T, rel_V, rcond=None)
        rg[t] = coef

        cont = np.zeros(I, dtype=float)
        cont[idx] = np.dot(rg[t], X)
          # continuation values for ITM paths

        exercise_now = h[t] > cont
        V[t] = np.where(exercise_now, h[t], V[t + 1] * df)
        ex[t] = np.where(exercise_now, 1.0, 0.0)

    df0 = np.exp(-(r[0] + r[1]) * 0.5 * dt)
    V0 = max(float(np.mean(V[1] * df0)), float(h[0, 0]))
      # LSM estimator

    return V0, S, r, v, ex, rg, h, dt


def main() -> None:
    """Simple smoke test for the BCC Monte Carlo engine."""
    from dawp_pIIIb_ch11_bcc_calibration import load_bcc_parameters

    params = load_bcc_parameters()
    K = params["S0"]
    T = 1.0

    V0, S, r, v, ex, rg, h, dt = american_put_lsm(
        params,
        K=K,
        T=T,
        n_steps=25,
        n_paths=20_000,
        basis_degree=3,
        seed=12345,
    )

    print(f"American put value (LSM, BCC): {V0:.4f}")
    print(f"Average exercise count per path: "
          f"{float(ex.sum(axis=0).mean()):.3f}")


if __name__ == "__main__":
    main()
