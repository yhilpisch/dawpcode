#
# CIR (1985) ZCB Valuation: MCS vs Analytical Figure
# dawp_pIIIa_ch10_fig06_zcb_mcs_vs_analytical_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pIIIa_ch09_cir_zcb import cir_zcb_value  # analytical ZCB values
from dawp_pIIIa_ch10_cir_mcs import (
    cir_generate_paths_exact,
    cir_generate_paths_full_truncation,
    zcb_value_mcs,
)

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
    # Compares analytical and Monte Carlo ZCB values for the CIR model.
    r0, kappa_r, theta_r, sigma_r = 0.01, 0.10, 0.03, 0.20
    T, n_steps, n_paths = 2.0, 50, 50_000

    r_exact = cir_generate_paths_exact(
        r0,
        kappa_r,
        theta_r,
        sigma_r,
        T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=10,
    )
    r_euler = cir_generate_paths_full_truncation(
        r0,
        kappa_r,
        theta_r,
        sigma_r,
        T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=10,
    )

    mcs_exact = zcb_value_mcs(r_exact, T=T)
    mcs_euler = zcb_value_mcs(r_euler, T=T)

    t_grid = np.linspace(0.0, T, n_steps + 1)
    ana = np.empty_like(t_grid)
    for i, t in enumerate(t_grid):
        if t >= T:
            ana[i] = 1.0
        else:
            ana[i] = cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T=T, t=float(t))

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.2, 6.2))

    ax[0].plot(t_grid, ana, lw=2.0, label="analytical")
    ax[0].plot(t_grid, mcs_exact, lw=0.0, marker="o", ms=3.5, label="MCS (exact)")
    ax[0].plot(t_grid, mcs_euler, lw=0.0, marker="x", ms=3.8, label="MCS (Euler)")
    ax[0].set_ylabel("ZCB value $B_t(T)$")
    ax[0].set_title("CIR (1985) ZCB valuation for maturity $T=2$")
    ax[0].grid(True)
    ax[0].legend(loc=0)

    ax[1].bar(t_grid - 0.01, mcs_exact - ana, width=0.02, label="exact - analytical")
    ax[1].bar(t_grid + 0.01, mcs_euler - ana, width=0.02, label="Euler - analytical")
    ax[1].set_xlabel("valuation time $t$")
    ax[1].set_ylabel("difference")
    ax[1].grid(True)
    ax[1].legend(loc=0)

    fig.tight_layout()
    maybe_save(fig, "dawp_pIIIa_fig06_zcb_mcs_vs_analytical")
    plt.show()


if __name__ == "__main__":
    main()
