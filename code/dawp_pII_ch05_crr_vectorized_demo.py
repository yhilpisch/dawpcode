#
# Cox-Ross-Rubinstein European Call Valuation (Vectorized Version)
# dawp_pII_ch05_crr_vectorized_demo.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions

import numpy as np  # numerical arrays


def crr_call_value_vectorized(S0, K, T, r, sigma, M=200):
    # Values a European call option via a CRR binomial tree using vectorization.
    dt = T / M  # time step length
    disc = math.exp(-r * dt)  # discount factor per step
    u = math.exp(sigma * math.sqrt(dt))  # up factor
    d = 1.0 / u  # down factor
    q = (math.exp(r * dt) - d) / (u - d)  # risk-neutral probability

    j = np.arange(M + 1)  # node index at maturity
    S_T = S0 * (u**j) * (d ** (M - j))  # terminal stock levels
    V = np.maximum(S_T - K, 0.0)  # terminal call payoffs

    for _ in range(M, 0, -1):
        V = disc * (q * V[1:] + (1.0 - q) * V[:-1])  # backward step
    return float(V[0])


def main():
    # Prices a call option for a single parameter set and prints the result.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility
    M = 200  # number of time steps

    value = crr_call_value_vectorized(S0, K, T, r, sigma, M=M)  # CRR call value
    print(f"CRR call value (vectorized, M={M}): {value:.6f}")


if __name__ == "__main__":
    main()  # run the main function

