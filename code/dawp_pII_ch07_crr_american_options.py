#
# American Options in the Cox-Ross-Rubinstein Model
# dawp_pII_ch07_crr_american_options.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions

import numpy as np  # numerical arrays


def crr_parameters(T, r, sigma, M):
    # Returns (disc, u, d, q) for a CRR tree with M steps.
    dt = T / M  # time step length
    disc = math.exp(-r * dt)  # discount factor per step
    u = math.exp(sigma * math.sqrt(dt))  # up factor
    d = 1.0 / u  # down factor
    q = (math.exp(r * dt) - d) / (u - d)  # risk-neutral probability
    return disc, u, d, q


def american_put_value(S0, K, T, r, sigma, M=500):
    # Values an American put option via a CRR binomial tree.
    disc, u, d, q = crr_parameters(T, r, sigma, M)  # CRR parameters

    j = np.arange(M + 1)  # node index at maturity
    S_T = S0 * (u**j) * (d ** (M - j))  # terminal stock levels
    V = np.maximum(K - S_T, 0.0)  # put payoff at maturity

    for m in range(M - 1, -1, -1):
        V = disc * (q * V[1:] + (1.0 - q) * V[:-1])  # continuation values
        S_m = S0 * (u ** j[: m + 1]) * (d ** (m - j[: m + 1]))
          # stock levels at time step m
        V = np.maximum(V, np.maximum(K - S_m, 0.0))  # early exercise value
    return float(V[0])


def short_condor_payoff(S, K1=90.0, K2=100.0, K3=110.0, K4=120.0):
    # Short condor spread payoff from four options with the same maturity.
    p1 = np.maximum(K1 - S, 0.0)  # long put at K1
    p2 = -np.maximum(K2 - S, 0.0)  # short put at K2
    c3 = -np.maximum(S - K3, 0.0)  # short call at K3
    c4 = np.maximum(S - K4, 0.0)  # long call at K4
    return p1 + p2 + c3 + c4


def main():
    # Computes a binomial benchmark value for the American put example.
    S0 = 36.0  # initial stock level
    K = 40.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.06  # constant short rate
    sigma = 0.20  # volatility

    value = american_put_value(S0, K, T, r, sigma, M=500)  # benchmark value
    print(f"CRR American put value (M=500): {value:.6f}")  # output


if __name__ == "__main__":
    main()  # run the main function

