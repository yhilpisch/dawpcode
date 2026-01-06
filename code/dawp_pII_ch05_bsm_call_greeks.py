#
# Black-Scholes-Merton European Call Greeks
# dawp_pII_ch05_bsm_call_greeks.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions

from scipy import stats  # normal distribution functions

from dawp_pII_ch05_bsm_option_valuation import bsm_d1, bsm_d2


def bsm_call_delta(S0, K, T, r, sigma):
    # Delta of a European call option in the BSM model (no dividends).
    d1 = bsm_d1(S0, K, T, r, sigma)  # d1 term
    return stats.norm.cdf(d1)


def bsm_call_gamma(S0, K, T, r, sigma):
    # Gamma of a European call option in the BSM model (no dividends).
    d1 = bsm_d1(S0, K, T, r, sigma)  # d1 term
    den = S0 * sigma * math.sqrt(T)  # scaling denominator
    return stats.norm.pdf(d1) / den


def bsm_call_theta(S0, K, T, r, sigma):
    # Theta of a European call option in the BSM model (no dividends).
    d1 = bsm_d1(S0, K, T, r, sigma)  # d1 term
    d2 = bsm_d2(S0, K, T, r, sigma)  # d2 term

    term1 = -(S0 * stats.norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
      # diffusion contribution
    term2 = -r * K * math.exp(-r * T) * stats.norm.cdf(d2)
      # discounting contribution
    return term1 + term2


def bsm_call_rho(S0, K, T, r, sigma):
    # Rho of a European call option in the BSM model (no dividends).
    d2 = bsm_d2(S0, K, T, r, sigma)  # d2 term
    return K * T * math.exp(-r * T) * stats.norm.cdf(d2)


def bsm_call_vega(S0, K, T, r, sigma):
    # Vega of a European call option in the BSM model (no dividends).
    d1 = bsm_d1(S0, K, T, r, sigma)  # d1 term
    return S0 * stats.norm.pdf(d1) * math.sqrt(T)


def main():
    # Prints a small set of Greeks for a single parameter set.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility

    delta = bsm_call_delta(S0, K, T, r, sigma)  # delta
    gamma = bsm_call_gamma(S0, K, T, r, sigma)  # gamma
    vega = bsm_call_vega(S0, K, T, r, sigma)  # vega

    print(f"delta: {delta:.6f}")  # delta output
    print(f"gamma: {gamma:.6f}")  # gamma output
    print(f"vega:  {vega:.6f}")  # vega output


if __name__ == "__main__":
    main()  # run the main function
