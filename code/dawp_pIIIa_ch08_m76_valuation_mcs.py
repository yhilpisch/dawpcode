#
# Merton (1976) Jump Diffusion: Call Valuation by Monte Carlo Simulation
# dawp_pIIIa_ch08_m76_valuation_mcs.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from dawp_pIIIa_ch08_m76_pricing import m76_call_value_mcs


def main():
    # Computes a call value via Monte Carlo simulation and prints the result.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate

    sigma = 0.40  # diffusion volatility
    lamb = 1.0  # jump intensity
    mu = -0.2  # mean log jump size
    delta = 0.10  # log jump volatility

    n_steps = 50  # time steps for simulation
    n_paths = 50_000  # Monte Carlo paths

    value = m76_call_value_mcs(
        S0,
        K,
        T,
        r,
        sigma,
        lamb,
        mu,
        delta,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=10,
    )
    print(f"M76 call value (MCS): {value:.6f}")


if __name__ == "__main__":
    main()  # run the main function

