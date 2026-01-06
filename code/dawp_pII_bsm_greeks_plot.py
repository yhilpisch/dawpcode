#
# Black-Scholes-Merton Call Greeks (Delta, Gamma, Vega)
# dawp_pII_bsm_greeks_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions
from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

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


def norm_pdf(x):
    # Standard normal probability density function.
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    # Standard normal cumulative distribution function via the error function.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bsm_d1(S0, K, T, r, sigma):
    # Black-Scholes-Merton d1 term.
    vol_sqrt_T = sigma * math.sqrt(T)  # volatility scaling term
    num = math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T  # d1 numerator
    return num / vol_sqrt_T


def bsm_call_delta(S0, K, T, r, sigma):
    # Delta of a European call option in the BSM model.
    d1 = bsm_d1(S0, K, T, r, sigma)
    return norm_cdf(d1)


def bsm_call_gamma(S0, K, T, r, sigma):
    # Gamma of a European call option in the BSM model.
    d1 = bsm_d1(S0, K, T, r, sigma)
    return norm_pdf(d1) / (S0 * sigma * math.sqrt(T))


def bsm_call_vega(S0, K, T, r, sigma):
    # Vega of a European call option in the BSM model.
    d1 = bsm_d1(S0, K, T, r, sigma)
    return S0 * norm_pdf(d1) * math.sqrt(T)


def main():
    # Plots delta, gamma, and vega as functions of the underlying level.
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility

    S0_grid = np.linspace(50.0, 150.0, 201)  # underlying grid for plotting
    delta = np.array([bsm_call_delta(x, K, T, r, sigma) for x in S0_grid])
      # delta values
    gamma = np.array([bsm_call_gamma(x, K, T, r, sigma) for x in S0_grid])
      # gamma values
    vega = np.array([bsm_call_vega(x, K, T, r, sigma) for x in S0_grid])
      # vega values

    fig, axes = plt.subplots(3, 1, sharex=True)  # three-row figure

    axes[0].plot(S0_grid, delta, lw=2.0)  # plot delta
    axes[0].set_ylabel("delta")  # y-axis label

    axes[1].plot(S0_grid, gamma, lw=2.0)  # plot gamma
    axes[1].set_ylabel("gamma")  # y-axis label

    axes[2].plot(S0_grid, vega, lw=2.0)  # plot vega
    axes[2].set_ylabel("vega")  # y-axis label
    axes[2].set_xlabel(r"underlying level $S_0$")  # x-axis label

    fig.suptitle("BSM call Greeks (fixed K, T, r, sigma)")  # overall title
    fig.tight_layout()  # improve spacing

    maybe_save(fig, "dawp_pII_fig02_bsm_greeks")  # optional PDF export
    plt.show()  # display the figure


if __name__ == "__main__":
    main()  # run the main function

