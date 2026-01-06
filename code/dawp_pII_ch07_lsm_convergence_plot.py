#
# LSM Convergence Diagnostic (American Put)
# dawp_pII_ch07_lsm_convergence_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions
from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pII_ch07_crr_american_options import american_put_value
from dawp_pII_ch07_lsm_primal import (
    PutParams,
    lsm_american_put_value,
    simulate_gbm_paths,
)

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


def main():
    # Generates the LSM convergence figure and exports it as a PDF.
    params = PutParams()  # parameter set
    n_steps = 50  # exercise dates excluding time 0

    V0_crr = american_put_value(params.S0, params.K, params.T, params.r,
                                params.sigma, M=500)
      # CRR benchmark value

    path_grid = np.array([2_000, 5_000, 10_000, 25_000, 50_000])  # path counts
    values = []
    for n in path_grid:
        _, S = simulate_gbm_paths(params, n_steps, int(n), seed=150000)
        V0, _ = lsm_american_put_value(S, params.K, params.r, params.T)
        values.append(V0)

    values = np.array(values)  # convert to an array

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(path_grid, values, lw=2.0, marker="o", label="LSM estimate")
      # LSM values
    ax.axhline(V0_crr, ls="--", lw=2.0, color="C3", label="CRR benchmark")
      # CRR benchmark

    ax.set_xscale("log")  # log scale for paths
    ax.set_xlabel("number of Monte Carlo paths")  # x-axis label
    ax.set_ylabel("American put value")  # y-axis label
    ax.set_title("LSM convergence diagnostic")  # plot title
    ax.legend(loc=0)  # place the legend

    maybe_save(fig, "dawp_pII_fig07_lsm_convergence")  # optional PDF export
    plt.show()  # display the plot


if __name__ == "__main__":
    main()  # run the main function
