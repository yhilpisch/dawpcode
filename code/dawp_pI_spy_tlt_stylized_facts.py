#
# Stylized Facts from SPY and TLT (EOD Data)
# dawp_pI_spy_tlt_stylized_facts.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # math functions
from pathlib import Path  # path handling for optional figure export

import numpy as np  # numerical arrays
import pandas as pd  # tabular data handling
import matplotlib.pyplot as plt  # plotting

FIG_SAVE = False  # set to True to export figures as PDFs
FIG_DIR = Path("../figures")  # figure output directory
FIG_DPI = 300  # target resolution for exported figures

DATA_URL = "https://hilpisch.com/nov25eod.csv"  # EOD data source


def maybe_save(fig, filename):
    # Optionally saves a Matplotlib figure as a PDF file.
    if not FIG_SAVE:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{filename}.pdf"
    fig.savefig(path, format="pdf", dpi=FIG_DPI)
    print(f"saved: {path}")


def load_prices():
    # Loads EOD data and returns a wide price DataFrame indexed by date.
    raw = pd.read_csv(DATA_URL)  # load data from the URL

    possible_dates = {"date", "datetime", "timestamp"}  # date column candidates
    date_col = None  # date column name
    for col in raw.columns:
        if col.lower() in possible_dates:
            date_col = col  # store the date column name
            break
    if date_col is None:
        raise ValueError("No date column found in the data.")  # fail early

    raw[date_col] = pd.to_datetime(raw[date_col])  # parse dates
    raw = raw.sort_values(date_col)  # sort by time

    cols_lower = {c.lower(): c for c in raw.columns}  # lower-case lookup
    sym_col = cols_lower.get("symbol") or cols_lower.get("ticker")  # symbol column

    price_candidates = [
        "adj_close",
        "adjclose",
        "adj close",
        "close",
        "price",
    ]  # price column candidates

    price_col = None  # price column name
    for key in price_candidates:
        if key in cols_lower:
            price_col = cols_lower[key]  # store the price column name
            break

    if sym_col and price_col:
        prices = raw.pivot(index=date_col, columns=sym_col, values=price_col)
          # pivot to wide format: dates x tickers
    else:
        prices = raw.set_index(date_col)  # assume already wide

    return prices.sort_index()  # enforce chronological order


def main():
    # Produces stylized fact plots used in Part I slides.
    prices = load_prices()  # wide price panel

    spy = prices["SPY"].dropna()  # SPY close series
    r_spy = np.log(spy / spy.shift(1)).dropna()  # SPY log-returns

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # two-row figure
    ax1.plot(spy, lw=1.6)  # plot the SPY level
    ax1.set_ylabel(r"level $S_t$")  # y-axis label
    ax1.set_title("SPY level and log-returns")  # title

    ax2.plot(r_spy, lw=1.1)  # plot daily log-returns
    ax2.set_ylabel(r"log-return $r_t$")  # y-axis label
    ax2.set_xlabel("trading day")  # x-axis label

    fig.tight_layout()  # improve spacing
    maybe_save(fig, "dawp_pI_fig03_spy_path_returns")  # optional PDF export
    plt.show()  # display the figure

    window = 21  # 1-month rolling window (trading days)
    roll_vol = r_spy.rolling(window).std(ddof=1) * math.sqrt(252.0)
      # annualized rolling volatility

    fig, ax = plt.subplots()  # create a single plot
    ax.plot(roll_vol, lw=1.4)  # plot rolling volatility
    ax.set_xlabel("trading day")  # x-axis label
    ax.set_ylabel("rolling volatility (annualized)")  # y-axis label
    ax.set_title("Volatility clustering: rolling estimate (SPY)")  # plot title

    maybe_save(fig, "dawp_pI_fig05_spy_rolling_vol")  # optional PDF export
    plt.show()  # display the plot

    tlt = prices["TLT"].dropna()  # TLT close series
    D_TLT = 17.0  # duration proxy for TLT (rough, but serviceable)

    dy = -(tlt.pct_change().dropna()) / D_TLT
      # yield change proxy from bond price changes

    y0 = 0.04  # baseline yield level for the proxy
    y_proxy = y0 + dy.cumsum()  # integrate yield changes into a yield series

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # two-row figure
    ax1.plot(100.0 * y_proxy, lw=1.6)  # plot the yield proxy
    ax1.set_ylabel("yield proxy (%)")  # y-axis label
    ax1.set_title("Rates proxy from TLT via duration approximation")  # title

    ax2.plot(1e4 * dy, lw=1.1)  # plot daily yield changes
    ax2.set_ylabel("daily change (bp)")  # y-axis label
    ax2.set_xlabel("trading day")  # x-axis label

    fig.tight_layout()  # improve spacing
    maybe_save(fig, "dawp_pI_fig07_tlt_yield_proxy")  # optional PDF export
    plt.show()  # display the figure


if __name__ == "__main__":
    main()  # run the main function
