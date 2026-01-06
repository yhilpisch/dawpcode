#
# Short Condor Spread Payoff Figure
# dawp_pII_ch07_short_condor_payoff_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pII_ch07_crr_american_options import short_condor_payoff

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
    # Generates the payoff profile for a short condor spread.
    S_grid = np.linspace(60.0, 150.0, 400)  # underlying grid
    payoff = short_condor_payoff(S_grid)  # payoff values

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(S_grid, payoff, lw=2.2)  # payoff curve
    ax.set_xlabel("underlying level")  # x-axis label
    ax.set_ylabel("payoff")  # y-axis label
    ax.set_title("Short condor spread payoff")  # plot title

    maybe_save(fig, "dawp_pII_fig08_short_condor_payoff")  # optional PDF export
    plt.show()  # display the plot


if __name__ == "__main__":
    main()  # run the main function

