#
# European Call Option Present Value vs Inner Value Plot
# dawp_pI_bsm_vs_inner_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # math functions for the BSM formula
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
    # Black--Scholes--Merton call option value for non-dividend paying underlyings.
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if sigma <= 0.0:
        return math.exp(-r * T) * max(S0 - K, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def main():
    # Compares the inner value curve to a benchmark present value curve.
    K = 8000.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.025  # constant short rate
    sigma = 0.20  # annual volatility

    S0 = np.linspace(4000.0, 12000.0, 200)  # initial levels for plotting
    inner = np.maximum(S0 - K, 0.0)  # inner value proxy on the grid
    C0 = np.array([bsm_call_value(x, K, T, r, sigma) for x in S0])
      # BSM present values on the grid

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(S0, inner, ls="-.", lw=2.5, label="inner value")  # inner value
    ax.plot(S0, C0, lw=2.5, label="present value")  # BSM value
    ax.set_xlabel(r"initial level $S_0$")  # x-axis label
    ax.set_ylabel(r"value at $t=0$")  # y-axis label
    ax.legend(loc=0)  # place the legend
    ax.set_title("Present value vs. inner value")  # plot title

    maybe_save(fig, "dawp_pI_fig02_bsm_vs_inner")  # optional PDF export
    plt.show()  # display the plot


if __name__ == "__main__":
    main()  # run the main function
