#
# Fourier Pricing in BSM: Analytical vs Lewis Integral vs Carr-Madan FFT
# dawp_pII_ch06_fourier_compare_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pII_ch06_fourier_pricing_bsm import (
    bsm_call_value,
    bsm_call_value_fft,
    bsm_call_value_lewis,
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
    # Compares analytical, Lewis integral, and FFT call values on a strike grid.
    S0 = 100.0  # initial underlying level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility

    K_grid = np.linspace(60.0, 140.0, 41)  # strike grid for comparison

    ana = np.array([bsm_call_value(S0, k, T, r, sigma) for k in K_grid])
      # analytical BSM call values
    lewis = np.array([bsm_call_value_lewis(S0, k, T, r, sigma) for k in K_grid])
      # Lewis integral call values
    fft_vals = np.array([bsm_call_value_fft(S0, k, T, r, sigma) for k in K_grid])
      # Carr-Madan FFT call values

    rel_err_lewis = (lewis - ana) / np.maximum(ana, 1e-12)  # relative error
    rel_err_fft = (fft_vals - ana) / np.maximum(ana, 1e-12)  # relative error

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # two-row figure

    ax1.plot(K_grid, ana, lw=2.0, label="analytical")  # analytical values
    ax1.plot(K_grid, lewis, lw=2.0, ls="--", label="Lewis integral")
      # integral values
    ax1.plot(K_grid, fft_vals, lw=2.0, ls=":", label="Carr-Madan FFT")
      # FFT values
    ax1.set_ylabel("call value")  # y-axis label
    ax1.set_title("BSM call values: analytical vs Fourier methods")  # title
    ax1.legend(loc=0)  # place the legend

    ax2.plot(K_grid, rel_err_lewis, lw=1.8, label="Lewis rel. error")
      # relative error curve
    ax2.plot(K_grid, rel_err_fft, lw=1.8, label="Carr-Madan rel. error")
      # Carr-Madan relative error curve
    ax2.axhline(0.0, lw=1.0, color="black")  # zero line
    ax2.set_xlabel("strike K")  # x-axis label
    ax2.set_ylabel("relative error")  # y-axis label
    ax2.legend(loc=0)  # place the legend

    fig.tight_layout()  # improve spacing
    maybe_save(fig, "dawp_pII_fig04_fou_compare")  # optional PDF export
    plt.show()  # display the figure


if __name__ == "__main__":
    main()  # run the main function
