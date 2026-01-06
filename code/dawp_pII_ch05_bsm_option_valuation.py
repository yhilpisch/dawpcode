#
# Black-Scholes-Merton European Call and Put Valuation
# dawp_pII_ch05_bsm_option_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions

import numpy as np  # numerical arrays
from scipy import stats  # normal distribution functions


def bsm_d1(S0, K, T, r, sigma):
    # Returns the d1 term for the BSM model (no dividends).
    num = math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T  # d1 numerator
    den = sigma * math.sqrt(T)  # d1 denominator
    return num / den


def bsm_d2(S0, K, T, r, sigma):
    # Returns the d2 term for the BSM model (no dividends).
    return bsm_d1(S0, K, T, r, sigma) - sigma * math.sqrt(T)


def bsm_call_value(S0, K, T, r, sigma):
    # Calculates the BSM European call option value (no dividends).
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if sigma <= 0.0:
        return math.exp(-r * T) * max(S0 - K, 0.0)

    d1 = bsm_d1(S0, K, T, r, sigma)  # d1 term
    d2 = d1 - sigma * math.sqrt(T)  # d2 term
    pvK = K * math.exp(-r * T)  # discounted strike
    return S0 * stats.norm.cdf(d1) - pvK * stats.norm.cdf(d2)


def bsm_put_value(S0, K, T, r, sigma):
    # Calculates the BSM European put option value via put-call parity.
    call = bsm_call_value(S0, K, T, r, sigma)  # call value
    return call - S0 + K * math.exp(-r * T)


def bsm_values_on_grid(S0_grid, K, T, r, sigma):
    # Vectorized helper returning call and put values on a grid of underlying levels.
    call_vals = np.array([bsm_call_value(x, K, T, r, sigma) for x in S0_grid])
      # call values on the grid
    put_vals = np.array([bsm_put_value(x, K, T, r, sigma) for x in S0_grid])
      # put values on the grid
    return call_vals, put_vals


def main():
    # Runs a minimal sanity check on call and put values.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility

    call = bsm_call_value(S0, K, T, r, sigma)  # call value
    put = bsm_put_value(S0, K, T, r, sigma)  # put value

    print(f"BSM call value: {call:.6f}")  # call value output
    print(f"BSM put value:  {put:.6f}")  # put value output


if __name__ == "__main__":
    main()  # run the main function

