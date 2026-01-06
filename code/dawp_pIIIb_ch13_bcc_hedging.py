#
# Dynamic Hedging of an American Put in the Calibrated BCC Model
# dawp_pIIIb_ch13_bcc_hedging.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from __future__ import annotations

import math  # elementary math functions
from typing import Dict, Tuple  # type hints

import numpy as np  # numerical arrays

from dawp_pIIIb_ch12_bcc_mcs import american_put_lsm


def bcc_american_put_hedge_run(
    params: Dict[str, float],
    K: float,
    T: float,
    n_steps: int,
    n_paths: int,
    dis: float=0.01,
    a: float=1.0,
    seed: int | None=None,
    path_index: int=0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Implements delta hedging for an American put in the BCC model.

    The routine relies on an initial LSM valuation run that provides
    simulated paths, inner values, exercise decisions, and regression
    coefficients for the continuation values. Delta estimates are based
    on revaluations of the regression-based continuation function at
    perturbed index levels along a single hedging path.
    """
    M = n_steps
    I = n_paths
    p = path_index

    if seed is None:
        seed_val = 50_000
    else:
        seed_val = seed

    ds = dis * params["S0"]
      # perturbation of the initial index level

    # LSM run for S0 + (2 - a) * ds: basis for delta hedge
    params_up = dict(params)
    params_up["S0"] = params["S0"] + (2.0 - a) * ds
    V_up, S, r, v, ex, rg, h, dt = american_put_lsm(
        params_up,
        K=K,
        T=T,
        n_steps=M,
        n_paths=I,
        basis_degree=3,
        seed=seed_val,
    )

    # LSM run for S0 - a * ds
    params_dn = dict(params)
    params_dn["S0"] = params["S0"] - a * ds
    V_dn = american_put_lsm(
        params_dn,
        K=K,
        T=T,
        n_steps=M,
        n_paths=I,
        basis_degree=3,
        seed=seed_val,
    )[0]

    delt = np.zeros(M + 1, dtype=float)
      # delta path
    delt[0] = (V_up - V_dn) / (2.0 * ds)
      # initial delta

    print("DYNAMIC HEDGING OF AMERICAN PUT (BCC97)")
    print("---------------------------------------")

    po = np.zeros(M + 1, dtype=float)
      # portfolio value path
    vt = np.zeros(M + 1, dtype=float)
      # option value path

    vt[0] = V_up
    po[0] = V_up
    bo = V_up - delt[0] * params["S0"]
      # initial bond position

    print("Initial hedge")
    print(f"Stocks             {delt[0]:8.3f}")
    print(f"Bonds              {bo:8.3f}")
    print(f"Cost               {delt[0] * params['S0'] + bo:8.3f}")

    print()
    print("Regular re-hedges")
    print(82 * "-")
    print("step|" + 7 * " %9s|" % ("S_t", "Port", "Put",
                                   "Diff", "Stock", "Bond", "Cost"))

    t_ex = M
      # exercise time index
    for t in range(1, M + 1):
        if ex[t, p] == 0:
            df = math.exp((r[t, p] + r[t - 1, p]) * 0.5 * dt)
              # discount factor between t-1 and t

            if t != M:
                po[t] = delt[t - 1] * S[t, p] + bo * df
                  # portfolio revaluation

                # state-dependent continuation value revaluation
                ds_t = dis * S[t, p]
                  # local perturbation
                sd_a = S[t, p] + (2.0 - a) * ds_t
                  # disturbed index level (up)
                state_a = [
                    sd_a * v[t, p] * r[t, p],
                    sd_a * v[t, p],
                    sd_a * r[t, p],
                    v[t, p] * r[t, p],
                    sd_a ** 2,
                    v[t, p] ** 2,
                    r[t, p] ** 2,
                    sd_a,
                    v[t, p],
                    r[t, p],
                    1.0,
                ]
                state_a.reverse()
                  # basis order as in american_put_lsm
                V_a = max(0.0, float(np.dot(rg[t], state_a)))

                sd_b = S[t, p] - a * ds_t
                  # disturbed index level (down)
                state_b = [
                    sd_b * v[t, p] * r[t, p],
                    sd_b * v[t, p],
                    sd_b * r[t, p],
                    v[t, p] * r[t, p],
                    sd_b ** 2,
                    v[t, p] ** 2,
                    r[t, p] ** 2,
                    sd_b,
                    v[t, p],
                    r[t, p],
                    1.0,
                ]
                state_b.reverse()
                V_b = max(0.0, float(np.dot(rg[t], state_b)))

                delt[t] = (V_a - V_b) / (2.0 * ds_t)
                  # updated delta
                bo = po[t] - delt[t] * S[t, p]
                  # updated bond position
                vt[t] = max(h[t, p], float(np.dot(
                    rg[t],
                    state_a,
                )))
                  # approximate option value at time t
            else:
                po[t] = delt[t - 1] * S[t, p] + bo * df
                vt[t] = h[t, p]
                  # inner value at maturity
                delt[t] = 0.0

            print("%4d|" % t + 7 * " %9.3f|" %
                  (S[t, p], po[t], vt[t],
                   (po[t] - vt[t]), delt[t],
                   bo, delt[t] * S[t, p] + bo))
        else:
            df = math.exp((r[t, p] + r[t - 1, p]) * 0.5 * dt)
            po[t] = delt[t - 1] * S[t, p] + bo * df
            vt[t] = h[t, p]
              # inner value at early exercise
            t_ex = t
            break

    errs = po[: t_ex + 1] - vt[: t_ex + 1]
      # hedge errors
    print(f"MSE             {float(np.mean(errs * errs)):7.3f}")
    print(f"Average Error   {float(np.mean(errs)):7.3f}")
    print(f"Total P&L       {float(np.sum(errs)):7.3f}")

    return S[:, p], po[: t_ex + 1], vt[: t_ex + 1], errs, t_ex


if __name__ == "__main__":
    from dawp_pIIIb_ch11_bcc_calibration import load_bcc_parameters

    params = load_bcc_parameters()
    S, po, vt, errs, t_ex = bcc_american_put_hedge_run(
        params,
        K=params["S0"],
        T=1.0,
        n_steps=50,
        n_paths=50_000,
        dis=0.01,
        a=1.0,
        seed=50_000,
        path_index=0,
    )

