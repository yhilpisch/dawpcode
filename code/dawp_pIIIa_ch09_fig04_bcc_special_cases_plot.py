#
# BCC (1997) Special Cases: Call Values Figure
# dawp_pIIIa_ch09_fig04_bcc_special_cases_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import math  # elementary math functions

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pIIIa_ch08_m76_pricing import m76_call_value_int  # Merton reference
from dawp_pIIIa_ch09_bcc_option_valuation import (
    bcc_call_value_int,
    bsm_call_value,
    cir_implied_short_rate,
    h93_call_value_int,
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
    # Compares European call values for special cases within the BCC framework.
    S0, T = 100.0, 1.0
    strikes = np.arange(70.0, 131.0, 5.0)

    r0, kappa_r, theta_r, sigma_r = 0.04, 0.30, 0.04, 0.10
    r = cir_implied_short_rate(r0, kappa_r, theta_r, sigma_r, T=T)

    kappa_v, theta_v, sigma_v, rho, v0 = 1.5, 0.02, 0.15, -0.7, 0.02

    lamb, mu, delta = 0.25, -0.20, 0.10
    sigma = math.sqrt(v0)

    bsm = np.array([bsm_call_value(S0, K, T, r, sigma) for K in strikes])
    m76 = np.array([m76_call_value_int(S0, K, T, r, sigma, lamb, mu, delta)
                    for K in strikes])
    h93 = np.array([h93_call_value_int(S0, K, T, r, kappa_v, theta_v,
                                       sigma_v, rho, v0) for K in strikes])
    bcc = np.array([bcc_call_value_int(S0, K, T, r, kappa_v, theta_v,
                                       sigma_v, rho, v0, lamb, mu, delta)
                    for K in strikes])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(strikes, bsm, lw=2.0, label="BSM (diffusion)")
    ax.plot(strikes, m76, lw=2.0, ls="--", label="Merton (jumps)")
    ax.plot(strikes, h93, lw=2.0, ls="-.", label="Heston (stoch. vol)")
    ax.plot(strikes, bcc, lw=2.0, color="tab:red", label="Bates (stoch. vol + jumps)")

    ax.set_xlabel("strike $K$")
    ax.set_ylabel("call value")
    ax.set_title("European call values: special cases within the BCC framework")
    ax.grid(True)
    ax.legend(loc=0)

    maybe_save(fig, "dawp_pIIIa_fig04_bcc_special_cases")
    plt.show()


if __name__ == "__main__":
    main()
