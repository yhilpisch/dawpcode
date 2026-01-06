#
# Cox-Ross-Rubinstein Binomial Model for European Option Valuation
# dawp_pII_ch05_crr_option_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions
from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pII_ch05_bsm_option_valuation import bsm_call_value

FIG_SAVE = False  # set to True to export the figure as a PDF
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


def crr_parameters(T, r, sigma, M):
    # Returns (dt, disc, u, d, q) for a CRR tree with M steps.
    dt = T / M  # time step length
    disc = math.exp(-r * dt)  # discount factor per step
    u = math.exp(sigma * math.sqrt(dt))  # up factor
    d = 1.0 / u  # down factor
    q = (math.exp(r * dt) - d) / (u - d)  # risk-neutral probability
    return dt, disc, u, d, q


def crr_option_value(S0, K, T, r, sigma, option_type="call", M=100):
    # Prices a European option in the CRR model by backward induction.
    _, disc, u, d, q = crr_parameters(T, r, sigma, M)  # CRR parameters

    j = np.arange(M + 1)  # node index at maturity
    S_T = S0 * (u**j) * (d ** (M - j))  # terminal stock levels

    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)  # call payoff at maturity
    elif option_type == "put":
        V = np.maximum(K - S_T, 0.0)  # put payoff at maturity
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    for _ in range(M, 0, -1):
        V = disc * (q * V[1:] + (1.0 - q) * V[:-1])  # backward step
    return float(V[0])


def plot_convergence(S0, K, T, r, sigma, m_min=5, m_max=401, step=5):
    # Plots CRR call values vs the BSM benchmark as M increases.
    m_grid = np.arange(m_min, m_max, step)  # number of time steps grid
    crr_vals = np.array([crr_option_value(S0, K, T, r, sigma, "call", int(m))
                         for m in m_grid])
      # CRR call values on the grid
    bsm_benchmark = bsm_call_value(S0, K, T, r, sigma)  # BSM benchmark

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(m_grid, crr_vals, lw=2.0, label="CRR values")  # CRR curve
    ax.axhline(
        bsm_benchmark,
        ls="--",
        lw=2.0,
        color="C3",
        label="BSM benchmark",
    )  # BSM benchmark line

    ax.set_xlabel("number of binomial steps")  # x-axis label
    ax.set_ylabel("European call value")  # y-axis label
    ax.set_title("CRR convergence to the BSM benchmark")  # plot title
    ax.legend(loc=0)  # place the legend

    maybe_save(fig, "dawp_pII_fig03_crr_convergence")  # optional PDF export
    plt.show()  # display the plot


def main():
    # Runs a convergence plot for a default parameter set.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility

    plot_convergence(S0, K, T, r, sigma)  # run the plot


if __name__ == "__main__":
    main()  # run the main function
