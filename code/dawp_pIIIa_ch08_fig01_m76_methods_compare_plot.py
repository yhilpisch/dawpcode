#
# Merton (1976) Pricing Method Comparison Figure
# dawp_pIIIa_ch08_fig01_m76_methods_compare_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pIIIa_ch08_m76_pricing import (
    m76_call_value_fft,
    m76_call_value_int,
    m76_call_value_mcs,
)

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


def main():
    # Compares integration, FFT, and Monte Carlo call values across strikes.
    S0 = 3225.93  # EURO STOXX 50 level (30.09.2014)
    T = 0.22  # short maturity (approx.)
    r = 0.005  # short rate assumption

    sigma = 0.124  # diffusion volatility
    lamb = 1.393  # jump intensity
    mu = -0.137  # mean log jump size
    delta = 0.000  # log jump volatility

    strikes = np.arange(3000.0, 3601.0, 50.0)  # strike grid

    v_int = np.array([m76_call_value_int(S0, k, T, r, sigma, lamb, mu, delta)
                      for k in strikes])
      # integration values
    v_fft = np.array([m76_call_value_fft(S0, k, T, r, sigma, lamb, mu, delta)
                      for k in strikes])
      # FFT values
    v_mcs = np.array(
        [
            m76_call_value_mcs(
                S0,
                k,
                T,
                r,
                sigma,
                lamb,
                mu,
                delta,
                n_steps=50,
                n_paths=30_000,
                seed=10,
            )
            for k in strikes
        ]
    )  # Monte Carlo values

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(strikes, v_int, lw=2.2, label="integration")  # integration curve
    ax.plot(strikes, v_fft, lw=2.2, ls="--", label="FFT")  # FFT curve
    ax.plot(strikes, v_mcs, lw=0.0, marker="o", ms=4, label="Monte Carlo")
      # Monte Carlo markers

    ax.set_xlabel("strike K")  # x-axis label
    ax.set_ylabel("call value")  # y-axis label
    ax.set_title("Merton (1976) call values by different methods")  # title
    ax.legend(loc=0)  # place the legend

    maybe_save(fig, "dawp_pIIIa_fig01_m76_methods_compare")  # optional PDF export
    plt.show()  # display the plot


if __name__ == "__main__":
    main()  # run the main function

