#
# Dynamic Hedging of an American Put in the BSM Model
# dawp_pIIIb_ch13_bsm_hedging.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from __future__ import annotations

import math  # elementary math functions
from typing import Tuple  # type hints

import numpy as np  # numerical arrays


def bsm_lsm_put_value(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None=None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Values an American put in the BSM model via the LSM algorithm."""
    if seed is not None:
        np.random.seed(seed)

    M = n_steps
    I = n_paths

    rand = np.random.standard_normal((M + 1, I))
      # Gaussian shocks
    dt = T / M
      # time step length
    df = math.exp(-r * dt)
      # per-step discount factor

    S = np.zeros((M + 1, I), dtype=float)
      # stock price matrix
    S[0] = S0

    for t in range(1, M + 1):
        drift = (r - 0.5 * sigma * sigma) * dt
        diff = sigma * math.sqrt(dt) * rand[t]
        S[t] = S[t - 1] * np.exp(drift + diff)

    h = np.maximum(K - S, 0.0)
      # inner value matrix
    V = h.copy()
      # value/cash flow matrix
    ex = np.zeros_like(V)
      # exercise matrix
    C = np.zeros_like(V)
      # continuation value matrix

    D = 9
      # polynomial degree (number of regression coefficients - 1)
    rg = np.zeros((M + 1, D + 1), dtype=float)
      # regression coefficient matrix

    for t in range(M - 1, 0, -1):
        target = V[t + 1] * df
          # discounted continuation cash flows
        coef = np.polyfit(S[t], target, D)
          # regression on current stock prices
        rg[t] = coef

        C[t] = np.polyval(coef, S[t])
        C[t] = np.where(C[t] < 0.0, 0.0, C[t])
          # truncate negative continuation values

        exercise_now = h[t] >= C[t]
        V[t] = np.where(exercise_now, h[t], V[t + 1] * df)
        ex[t] = np.where(exercise_now, 1.0, 0.0)

    V0 = float(np.mean(V[1]) * df)
      # LSM estimator
    return V0, S, ex, rg, h, dt


def bsm_american_put_hedge_run(
    S0: float,
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    sigma: float,
    r: float,
    dis: float=0.01,
    a: float=1.0,
    seed: int | None=None,
    path_index: int=0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Implements discrete-time delta hedging for a single BSM path.

    The hedge is based on LSM American put values and finite-difference
    approximations of the delta along a selected price path.
    """
    if seed is not None:
        np.random.seed(seed)

    M = n_steps
    I = n_paths
    p = path_index

    ds = dis * S0
      # perturbation size for finite-difference deltas
    V_1, S, ex, rg, h, dt = bsm_lsm_put_value(
        S0 + ds,
        K,
        T,
        r,
        sigma,
        n_steps=M,
        n_paths=I,
        seed=None,
    )
    V_2, _, _, _, _, _ = bsm_lsm_put_value(
        S0,
        K,
        T,
        r,
        sigma,
        n_steps=M,
        n_paths=I,
        seed=None,
    )
    del_0 = (V_1 - V_2) / ds
      # initial delta

    delt = np.zeros(M + 1, dtype=float)
      # delta path
    delt[0] = del_0

    print("DYNAMIC HEDGING OF AMERICAN PUT (BSM)")
    print("-------------------------------------")

    po = np.zeros(M + 1, dtype=float)
      # portfolio value path
    vt = np.zeros(M + 1, dtype=float)
      # option value path

    vt[0] = V_1
    po[0] = V_1
    bo = V_1 - delt[0] * S0
      # initial bond position value

    print("Initial hedge")
    print(f"Stocks             {delt[0]:8.3f}")
    print(f"Bonds              {bo:8.3f}")
    print(f"Cost               {delt[0] * S0 + bo:8.3f}")

    print()
    print("Regular re-hedges")
    print(68 * "-")
    print("step|" + 7 * " %7s|" % ("S_t", "Port", "Put",
                                   "Diff", "Stock", "Bond", "Cost"))

    t_ex = M
      # exercise time index
    for t in range(1, M + 1):
        if ex[t, p] == 0:
            vt[t] = bsm_lsm_put_value(
                S[t, p],
                K,
                T - t * dt,
                r,
                sigma,
                n_steps=M - t,
                n_paths=I,
                seed=None,
            )[0]
              # re-valued option
            po[t] = delt[t - 1] * S[t, p] + bo * math.exp(r * dt)
              # portfolio revaluation

            ds_t = dis * S[t, p]
              # perturbation at time t
            up_val = bsm_lsm_put_value(
                S[t, p] + (2.0 - a) * ds_t,
                K,
                T - t * dt,
                r,
                sigma,
                n_steps=M - t,
                n_paths=I,
                seed=None,
            )[0]
            dn_val = bsm_lsm_put_value(
                S[t, p] - a * ds_t,
                K,
                T - t * dt,
                r,
                sigma,
                n_steps=M - t,
                n_paths=I,
                seed=None,
            )[0]
            delt[t] = (up_val - dn_val) / (2.0 * ds_t)
              # updated delta
            bo = po[t] - delt[t] * S[t, p]
              # updated bond position

            print("%4d|" % t + 7 * " %7.3f|" %
                  (S[t, p], po[t], vt[t],
                   (po[t] - vt[t]), delt[t],
                   bo, delt[t] * S[t, p] + bo))
        else:
            vt[t] = h[t, p]
              # inner value at exercise
            po[t] = delt[t - 1] * S[t, p] + bo * math.exp(r * dt)
            t_ex = t
            break

    errs = po[: t_ex + 1] - vt[: t_ex + 1]
      # hedge errors
    print(f"MSE             {float(np.mean(errs * errs)):7.3f}")
    print(f"Average Error   {float(np.mean(errs)):7.3f}")
    print(f"Total P&L       {float(np.sum(errs)):7.3f}")

    return S[:, p], po[: t_ex + 1], vt[: t_ex + 1], errs, t_ex


if __name__ == "__main__":
    S, po, vt, errs, t_ex = bsm_american_put_hedge_run(
        S0=36.0,
        K=40.0,
        T=1.0,
        n_steps=50,
        n_paths=50_000,
        sigma=0.20,
        r=0.06,
        dis=0.01,
        a=1.0,
        seed=50_000,
        path_index=0,
    )

