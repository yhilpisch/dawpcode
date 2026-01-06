#
# Cox-Ross-Rubinstein European Call Valuation (Loop Version)
# dawp_pII_ch05_crr_loop_demo.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions

import numpy as np  # numerical arrays


def crr_call_value_loop(S0, K, T, r, sigma, M=200):
    # Values a European call option via a CRR binomial tree using explicit loops.
    dt = T / M  # time step length
    disc = math.exp(-r * dt)  # discount factor per step
    u = math.exp(sigma * math.sqrt(dt))  # up factor
    d = 1.0 / u  # down factor
    q = (math.exp(r * dt) - d) / (u - d)  # risk-neutral probability

    S = np.zeros((M + 1, M + 1), dtype=float)  # stock level array
    S[0, 0] = S0  # set initial node

    for t in range(1, M + 1):
        S[0, t] = S[0, t - 1] * u  # top node is always an up move
        for i in range(1, t + 1):
            S[i, t] = S[i - 1, t - 1] * d  # down move from the upper-left node

    V = np.maximum(S[:, M] - K, 0.0)  # call payoffs at maturity

    for t in range(M - 1, -1, -1):
        V[: t + 1] = disc * (q * V[: t + 1] + (1.0 - q) * V[1 : t + 2])
          # backward induction step
    return float(V[0])


def main():
    # Prices a call option for a single parameter set and prints the result.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate
    sigma = 0.20  # volatility
    M = 200  # number of time steps

    value = crr_call_value_loop(S0, K, T, r, sigma, M=M)  # CRR call value
    print(f"CRR call value (loop, M={M}): {value:.6f}")


if __name__ == "__main__":
    main()  # run the main function

