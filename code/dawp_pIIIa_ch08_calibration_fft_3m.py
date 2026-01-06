#
# Merton (1976) Calibration to Option Market Data (Three Maturities)
# dawp_pIIIa_ch08_calibration_fft_3m.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions
from pathlib import Path  # robust path handling

import numpy as np  # numerical arrays
import pandas as pd  # tabular data handling
import matplotlib.pyplot as plt  # plotting
import scipy.optimize as sop  # numerical optimization

from dawp_pIIIa_ch08_m76_pricing import m76_call_value_fft

FIG_SAVE = False  # set to True to export figures as PDFs
FIG_DIR = Path("../figures")  # figure output directory
FIG_DPI = 300  # target resolution for exported figures

DATA_PATH = Path(__file__).resolve().with_name("dawp_pIIIa_ch08_option_data.h5")


def maybe_save(fig, filename):
    # Optionally saves a Matplotlib figure as a PDF file.
    if not FIG_SAVE:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{filename}.pdf"
    fig.savefig(path, format="pdf", dpi=FIG_DPI)
    print(f"saved: {path}")


def load_option_data():
    # Loads the option data table from the local HDF5 file.
    data = pd.read_hdf(DATA_PATH, key="data")  # options table
    data = data.copy()  # explicit copy to avoid view warnings

    data["Date"] = pd.to_datetime(data["Date"])  # valuation date
    data["Maturity"] = pd.to_datetime(data["Maturity"])  # maturity date
    return data


def rmse(model_vals, market_vals):
    # Root mean squared error for model vs market prices.
    diff = np.asarray(model_vals) - np.asarray(market_vals)
    return float(np.sqrt(np.mean(diff * diff)))


def calibrate_m76_fft(options, S0, r):
    # Calibrates M76 parameters by minimizing RMSE across the option set.
    strikes = options["Strike"].to_numpy()  # strike vector
    calls = options["Call"].to_numpy()  # market call prices
    T = options["T"].to_numpy()  # maturity times in years

    def objective(p):
        # Objective function returning RMSE for a parameter vector.
        sigma, lamb, mu, delta = p
        if sigma <= 0.0 or delta < 0.0 or lamb < 0.0:
            return 1e6

        model = [
            m76_call_value_fft(S0, K, t, r, sigma, lamb, mu, delta)
            for K, t in zip(strikes, T)
        ]
        return rmse(model, calls)

    x0 = np.array([0.12, 1.0, -0.12, 0.10])  # initial guess
    bounds = [(0.02, 0.50), (0.0, 5.0), (-0.50, 0.05), (0.0, 0.50)]
      # parameter bounds

    res = sop.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 250},
    )
    return res


def plot_fit(options, S0, r, params):
    # Plots market call prices and calibrated model values across maturities.
    sigma, lamb, mu, delta = params  # calibrated parameters
    options = options.copy()  # work on a copy

    options["Model"] = [
        m76_call_value_fft(S0, K, T, r, sigma, lamb, mu, delta)
        for K, T in zip(options["Strike"], options["T"])
    ]  # model call prices

    mats = sorted(options["Maturity"].unique())  # maturities in the data
    fig, axes = plt.subplots(len(mats), 1, sharex=True)  # one panel per maturity
    if len(mats) == 1:
        axes = [axes]

    for ax, mat in zip(axes, mats):
        sub = options[options["Maturity"] == mat].sort_values("Strike")
        ax.plot(sub["Strike"], sub["Call"], lw=2.0, label="market")  # market
        ax.plot(sub["Strike"], sub["Model"], lw=0.0, marker="o", ms=4,
                label="model")  # model markers
        ax.set_ylabel("call value")  # y-axis label
        ax.set_title(str(mat)[:10])  # panel title
        ax.legend(loc=0)  # place the legend

    axes[-1].set_xlabel("strike K")  # x-axis label
    fig.suptitle("Merton (1976) calibration fit across maturities")  # overall title
    fig.tight_layout()  # improve spacing

    maybe_save(fig, "dawp_pIIIa_fig02_m76_calibration_fit")  # optional PDF export
    plt.show()  # display the plot


def main():
    # Loads market data, calibrates the model, and exports the fit figure.
    data = load_option_data()  # market option data

    S0 = 3225.93  # EURO STOXX 50 level (30.09.2014)
    r = 0.0005  # short rate proxy used in the book example

    tol = 0.02  # relative strike window around S0
    options = data[(np.abs(data["Strike"] - S0) / S0) < tol].copy()
      # near-the-money option selection
    options["T"] = (options["Maturity"] - options["Date"]).dt.days / 365.0
      # time to maturity in years

    res = calibrate_m76_fft(options, S0, r)  # model calibration
    sigma, lamb, mu, delta = res.x  # calibrated parameters

    print("Calibration success:", bool(res.success))  # status
    print("RMSE:", float(res.fun))  # RMSE value
    print("sigma, lambda, mu, delta:", res.x)  # parameter vector

    plot_fit(options, S0, r, (sigma, lamb, mu, delta))  # fit plot


if __name__ == "__main__":
    main()  # run the main function

