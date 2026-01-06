#
# Black-Scholes-Merton Call and Put Value Curves
# dawp_pII_bsm_values_plot.py
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


def norm_cdf(x):
    # Standard normal cumulative distribution function via the error function.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bsm_call_value(S0, K, T, r, sigma):
    # Black-Scholes-Merton European call value (no dividends).
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if sigma <= 0.0:
        return math.exp(-r * T) * max(S0 - K, 0.0)

    vol_sqrt_T = sigma * math.sqrt(T)  # volatility scaling term
    num = math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T  # d1 numerator
    d1 = num / vol_sqrt_T  # d1 term
    d2 = d1 - vol_sqrt_T  # d2 term

    pvK = K * math.exp(-r * T)  # discounted strike
    return S0 * norm_cdf(d1) - pvK * norm_cdf(d2)


def bsm_put_value(S0, K, T, r, sigma):
    # Black-Scholes-Merton European put value via put-call parity.
    call = bsm_call_value(S0, K, T, r, sigma)  # call value
    return call - S0 + K * math.exp(-r * T)  # parity relationship


def main():
    # Plots call and put values as functions of the underlying level.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility

    S0_grid = np.linspace(50.0, 150.0, 201)  # underlying grid for plotting
    call_vals = np.array([bsm_call_value(x, K, T, r, sigma) for x in S0_grid])
      # call values on the grid
    put_vals = np.array([bsm_put_value(x, K, T, r, sigma) for x in S0_grid])
      # put values on the grid

    call_inner = np.maximum(S0_grid - K, 0.0)  # intrinsic call values
    put_inner = np.maximum(K - S0_grid, 0.0)  # intrinsic put values

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(S0_grid, call_vals, lw=2.2, label="call value")  # call value curve
    ax.plot(S0_grid, put_vals, lw=2.2, label="put value")  # put value curve
    ax.plot(S0_grid, call_inner, ls="--", lw=1.8, label="call intrinsic")
      # intrinsic call value
    ax.plot(S0_grid, put_inner, ls="--", lw=1.8, label="put intrinsic")
      # intrinsic put value

    ax.set_xlabel(r"underlying level $S_0$")  # x-axis label
    ax.set_ylabel(r"value at $t=0$")  # y-axis label
    ax.set_title("BSM call and put values")  # plot title
    ax.legend(loc=0)  # place the legend

    maybe_save(fig, "dawp_pII_fig01_bsm_values")  # optional PDF export
    plt.show()  # display the plot


if __name__ == "__main__":
    main()  # run the main function

