#
# Longstaff-Schwartz LSM Primal Valuation for an American Put
# dawp_pII_ch07_lsm_primal.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions
from dataclasses import dataclass  # structured parameter container
from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting
from numpy.random import default_rng  # random number generator

FIG_SAVE = False  # set to True to export figures as PDFs
FIG_DIR = Path("../figures")  # figure output directory
FIG_DPI = 300  # target resolution for exported figures


def maybe_save(fig, filename):
    # Optionally saves a Matplotlib figure as a PDF file.
    if not FIG_SAVE:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{filename}.pdf"
    fig.savefig(path, format="pdf", dpi=FIG_DPI)
    print(f"saved: {path}")


@dataclass(frozen=True)
class PutParams:
    # Parameter set for the American put example.
    S0: float=36.0
    K: float=40.0
    T: float=1.0
    r: float=0.06
    sigma: float=0.20


def simulate_gbm_paths(params, n_steps, n_paths, seed=150000):
    # Simulates GBM stock paths under risk-neutral dynamics.
    dt = params.T / n_steps  # time step length
    drift = (params.r - 0.5 * params.sigma * params.sigma) * dt
      # drift of log-returns per step
    vol = params.sigma * math.sqrt(dt)  # diffusion scale per step

    rng = default_rng(seed)  # random number generator
    Z = rng.standard_normal((n_steps, n_paths))  # standard normal shocks
    log_increments = drift + vol * Z  # log-return increments

    log_paths = np.vstack(
        [np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)]
    )  # cumulative log-returns, includes time 0
    S = params.S0 * np.exp(log_paths)  # convert to levels
    t_grid = np.linspace(0.0, params.T, n_steps + 1)  # time grid
    return t_grid, S


def american_put_payoff(S, K):
    # Intrinsic value matrix for an American put option.
    return np.maximum(K - S, 0.0)


def lsm_american_put_value(S, K, r, T, poly_degree=5, itm_only=True):
    # Values an American put by the Longstaff-Schwartz primal algorithm.
    n_steps = S.shape[0] - 1  # number of time steps
    dt = T / n_steps  # time step length
    df = math.exp(-r * dt)  # discount factor per step

    h = american_put_payoff(S, K)  # intrinsic values
    V = h[-1].copy()  # start with intrinsic value at maturity

    exercise = np.zeros_like(h, dtype=bool)  # exercise decision matrix
    exercise[-1] = h[-1] > 0.0  # exercise at maturity if ITM

    for t in range(n_steps - 1, 0, -1):
        x = S[t]  # state variable at time t
        y = V * df  # discounted continuation cash flows

        if itm_only:
            itm = h[t] > 0.0  # in-the-money mask
            x_reg = x[itm]  # regression x values
            y_reg = y[itm]  # regression y values
        else:
            x_reg = x  # all paths
            y_reg = y  # all paths

        if len(x_reg) < poly_degree + 1:
            cont = np.zeros_like(x)  # fallback if regression is ill-posed
        else:
            coeff = np.polyfit(x_reg, y_reg, poly_degree)
              # polynomial regression coefficients
            cont = np.polyval(coeff, x)  # continuation value estimate

        ex_now = h[t] > cont  # exercise decision
        exercise[t] = ex_now & (h[t] > 0.0)  # only exercise if ITM
        V = np.where(ex_now, h[t], V * df)  # update along paths

    V0 = float(df * np.mean(V))  # discount from time 1 to time 0
    return V0, exercise


def plot_exercise_boundary(t_grid, S, exercise, payoff, filename):
    # Plots the LSM exercise points in the (t, S_t) plane.
    t_mat = np.repeat(t_grid[:, None], S.shape[1], axis=1)  # time matrix
    mask = exercise & (payoff > 0.0)  # exercise points in the ITM region

    fig, ax = plt.subplots()  # create a single plot
    ax.scatter(t_mat[mask], S[mask], s=2, alpha=0.15, label="exercise")
      # exercise scatter plot
    ax.set_xlabel("time")  # x-axis label
    ax.set_ylabel(r"underlying level $S_t$")  # y-axis label
    ax.set_title("LSM early exercise points (American put)")  # plot title
    ax.legend(loc=0)  # place the legend

    maybe_save(fig, filename)  # optional PDF export
    plt.show()  # display the plot


def main():
    # Runs a baseline LSM valuation and exports the exercise boundary figure.
    params = PutParams()  # parameter set
    n_steps = 50  # exercise dates excluding time 0
    n_paths = 25_000  # Monte Carlo paths

    t_grid, S = simulate_gbm_paths(params, n_steps, n_paths)  # simulated paths
    payoff = american_put_payoff(S, params.K)  # intrinsic values

    V0, exercise = lsm_american_put_value(S, params.K, params.r, params.T)
      # primal LSM valuation
    print(f"LSM American put value: {V0:.6f}")  # valuation output

    plot_exercise_boundary(
        t_grid,
        S,
        exercise,
        payoff,
        "dawp_pII_fig06_lsm_exercise_boundary",
    )  # boundary figure


if __name__ == "__main__":
    main()  # run the main function
