"""Derivatives Analytics with Python.

Merton (1976) Jump Diffusion: Call Valuation by Numerical Integration.
(c) Dr. Yves J. Hilpisch
AI-Powered by different LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""
from dawp_pIIIa_ch08_m76_pricing import m76_call_value_int


def main():
    # Computes a single call value via Lewis-style Fourier integration.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate

    sigma = 0.40  # diffusion volatility
    lamb = 1.0  # jump intensity
    mu = -0.2  # mean log jump size
    delta = 0.10  # log jump volatility

    value = m76_call_value_int(S0, K, T, r, sigma, lamb, mu, delta)
    print(f"M76 call value (integration): {value:.6f}")


if __name__ == "__main__":
    main()  # run the main function
