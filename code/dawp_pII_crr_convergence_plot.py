#
# Cox-Ross-Rubinstein Convergence to the BSM Benchmark
# dawp_pII_crr_convergence_plot.py
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


def crr_call_value(S0, K, T, r, sigma, M=100):
    # Cox-Ross-Rubinstein European call value via backward induction.
    dt = T / M  # time step
    u = math.exp(sigma * math.sqrt(dt))  # up factor
    d = 1.0 / u  # down factor
    disc = math.exp(-r * dt)  # discount factor per step
    q = (math.exp(r * dt) - d) / (u - d)  # risk-neutral probability

    j = np.arange(M + 1)  # node index at maturity
    S_T = S0 * (u**j) * (d ** (M - j))  # terminal stock levels
    V = np.maximum(S_T - K, 0.0)  # terminal call payoffs

    for _ in range(M, 0, -1):
        V = disc * (q * V[1:] + (1.0 - q) * V[:-1])  # backward step
    return float(V[0])


def main():
    # Visualizes convergence of CRR call values to the BSM benchmark.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility

    M_grid = np.arange(5, 401, 5)  # number of time steps for convergence plot
    crr_vals = np.array([crr_call_value(S0, K, T, r, sigma, int(m)) for m in M_grid])
      # CRR call values as a function of M

    bsm_benchmark = bsm_call_value(S0, K, T, r, sigma)  # BSM benchmark call value

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(M_grid, crr_vals, lw=2.0, label="CRR values")  # CRR value curve
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


if __name__ == "__main__":
    main()  # run the main function

