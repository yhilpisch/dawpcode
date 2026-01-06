#
# Merton (1976) Jump Diffusion: Call Valuation by FFT
# dawp_pIIIa_ch08_m76_valuation_fft.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from dawp_pIIIa_ch08_m76_pricing import m76_call_value_fft


def main():
    # Computes a single call value via Carr-Madan FFT pricing.
    S0 = 100.0  # initial underlying level
    K = 100.0  # strike level
    T = 1.0  # time to maturity in years
    r = 0.05  # constant short rate

    sigma = 0.40  # diffusion volatility
    lamb = 1.0  # jump intensity
    mu = -0.2  # mean log jump size
    delta = 0.10  # log jump volatility

    value = m76_call_value_fft(S0, K, T, r, sigma, lamb, mu, delta)
    print(f"M76 call value (FFT): {value:.6f}")


if __name__ == "__main__":
    main()  # run the main function

