#
# CIR (1985) Short-Rate Paths Figure
# dawp_pIIIa_ch10_fig05_cir_paths_plot.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import matplotlib.pyplot as plt  # plotting

from dawp_pIIIa_ch10_cir_mcs import cir_generate_paths_exact  # CIR simulation

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
    # Generates a sample of simulated CIR short-rate paths.
    r0, kappa_r, theta_r, sigma_r = 0.01, 0.10, 0.03, 0.20
    T, n_steps, n_paths = 2.0, 250, 20

    r = cir_generate_paths_exact(
        r0,
        kappa_r,
        theta_r,
        sigma_r,
        T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=10,
    )

    t = np.linspace(0.0, T, n_steps + 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(t, r, lw=1.2)
    ax.set_xlabel("time $t$ (years)")
    ax.set_ylabel("short rate $r_t$")
    ax.set_title("CIR (1985): twenty simulated short-rate paths (exact scheme)")
    ax.grid(True)

    maybe_save(fig, "dawp_pIIIa_fig05_cir_paths")
    plt.show()


if __name__ == "__main__":
    main()

