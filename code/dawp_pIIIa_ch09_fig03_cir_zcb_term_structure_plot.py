#
# CIR (1985) Bond Term Structure Figure
# dawp_pIIIa_ch09_fig03_cir_zcb_term_structure_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pIIIa_ch09_cir_zcb import cir_zcb_value, cir_zcb_yield  # analytics

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
    # Plots ZCB prices and yields implied by a CIR short-rate model.
    r0, kappa_r, theta_r, sigma_r = 0.04, 0.30, 0.04, 0.10
    maturities = np.linspace(0.25, 10.0, 40)

    prices = np.array(
        [cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T=float(T)) for T in maturities]
    )
    yields = np.array(
        [cir_zcb_yield(r0, kappa_r, theta_r, sigma_r, T=float(T)) for T in maturities]
    )

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.2, 5.8))
    ax[0].plot(maturities, prices, lw=2.2)
    ax[0].set_ylabel("ZCB value $B_0(T)$")
    ax[0].grid(True)

    ax[1].plot(maturities, 100.0 * yields, lw=2.2, color="tab:green")
    ax[1].set_xlabel("maturity $T$ (years)")
    ax[1].set_ylabel("yield $y_0(T)$ in \\%")
    ax[1].grid(True)

    fig.suptitle("CIR (1985) term structure: discount factors and yields", y=0.98)
    fig.tight_layout()

    maybe_save(fig, "dawp_pIIIa_fig03_cir_term_structure")
    plt.show()


if __name__ == "__main__":
    main()

