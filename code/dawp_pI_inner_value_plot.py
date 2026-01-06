#
# European Call Option Inner Value Plot
# dawp_pI_inner_value_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

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


def main():
    # Generates the inner value curve of a European call option.
    K = 8000.0  # strike level

    S_T = np.linspace(7000.0, 9000.0, 200)  # terminal levels for plotting
    h_T = np.maximum(S_T - K, 0.0)  # inner values on the grid

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(S_T, h_T, lw=2.5)  # plot call inner value
    ax.set_xlabel(r"terminal level $S_T$")  # x-axis label
    ax.set_ylabel("inner value")  # y-axis label
    ax.set_title("European call inner value")  # plot title

    maybe_save(fig, "dawp_pI_fig01_inner_value")  # optional PDF export
    plt.show()  # display the plot


if __name__ == "__main__":
    main()  # run the main function
