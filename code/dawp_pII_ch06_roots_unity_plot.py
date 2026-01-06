#
# Roots of Unity for Discrete Fourier Transforms
# dawp_pII_ch06_roots_unity_plot.py
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
    # Visualizes N roots of unity on the complex unit circle.
    N = 16  # number of roots of unity
    m = np.arange(N)  # root indices
    z = np.exp(2.0j * np.pi * m / N)  # roots of unity on the unit circle

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(np.real(z), np.imag(z), "o", ms=7)  # plot the roots
    ax.plot([0.0, 1.0], [0.0, 0.0], lw=1.2, color="black")  # reference axis
    ax.plot([0.0, 0.0], [0.0, 1.0], lw=1.2, color="black")  # reference axis

    theta = np.linspace(0.0, 2.0 * np.pi, 400)  # unit circle parameter grid
    ax.plot(np.cos(theta), np.sin(theta), lw=1.2)  # draw the unit circle

    ax.set_aspect("equal", "box")  # equal scaling in x and y
    ax.set_xlabel("real part")  # x-axis label
    ax.set_ylabel("imaginary part")  # y-axis label
    ax.set_title(f"Roots of unity (N={N})")  # plot title

    maybe_save(fig, "dawp_pII_fig05_roots_unity")  # optional PDF export
    plt.show()  # display the plot


if __name__ == "__main__":
    main()  # run the main function

