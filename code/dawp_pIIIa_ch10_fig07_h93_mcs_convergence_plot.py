#
# Heston (1993) European Call: Monte Carlo Convergence Figure
# dawp_pIIIa_ch10_fig07_h93_mcs_convergence_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pIIIa_ch09_bcc_option_valuation import h93_call_value_int  # benchmark
from dawp_pIIIa_ch10_h93_mcs import h93_call_value_mcs  # MCS estimator

FIG_SAVE = False  # set to True to export the figure as a PDF
FIG_DIR = Path("../figures")  # figure output directory
FIG_DPI = 300  # target resolution for exported figures


def maybe_save(fig, filename: str) -> None:
    # Optionally saves a Matplotlib figure as a PDF file.
    if not FIG_SAVE:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{filename}.pdf"
    fig.savefig(path, format="pdf", dpi=FIG_DPI)
    print(f"saved: {path}")


def main() -> None:
    # Plots convergence of the Heston Monte Carlo call estimator.
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    kappa_v, theta_v, sigma_v, rho, v0 = 1.5, 0.02, 0.30, -0.7, 0.04
    n_steps = 250

    ref = h93_call_value_int(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)

    paths_grid = np.array([2_500, 5_000, 10_000, 25_000, 50_000, 100_000])
    vals = np.array(
        [
            h93_call_value_mcs(
                S0,
                K,
                T,
                r,
                kappa_v,
                theta_v,
                sigma_v,
                rho,
                v0,
                n_steps=n_steps,
                n_paths=int(n_paths),
                seed=10,
            )
            for n_paths in paths_grid
        ]
    )
    rel_err = (vals - ref) / ref

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.2, 6.2))
    ax[0].plot(paths_grid, 0.0 * paths_grid + ref, lw=2.0, label="Lewis (benchmark)")
    ax[0].plot(paths_grid, vals, lw=0.0, marker="o", ms=5, label="Monte Carlo")
    ax[0].set_ylabel("call value")
    ax[0].set_title("Heston (1993) call values: benchmark vs Monte Carlo")
    ax[0].grid(True)
    ax[0].legend(loc=0)

    ax[1].plot(paths_grid, rel_err, lw=2.0, color="tab:green",
               label="relative error")
    ax[1].axhline(0.0, lw=1.2, color="k")
    ax[1].set_xlabel("number of paths")
    ax[1].set_ylabel("relative error")
    ax[1].grid(True)
    ax[1].legend(loc=0)

    ax[1].set_xscale("log")
    fig.tight_layout()

    maybe_save(fig, "dawp_pIIIa_fig07_h93_mcs_convergence")
    plt.show()


if __name__ == "__main__":
    main()

