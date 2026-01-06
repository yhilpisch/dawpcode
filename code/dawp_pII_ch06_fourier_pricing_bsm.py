#
# Fourier-Based Pricing in the Black-Scholes-Merton Model
# dawp_pII_ch06_fourier_pricing_bsm.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions

import numpy as np  # numerical arrays
from numpy.fft import fft  # fast Fourier transform
from scipy import stats  # normal distribution functions
from scipy.integrate import quad  # numerical integration


def bsm_call_value(S0, K, T, r, sigma):
    # Analytical BSM European call value (no dividends).
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if sigma <= 0.0:
        return math.exp(-r * T) * max(S0 - K, 0.0)

    vol_sqrt_T = sigma * math.sqrt(T)  # volatility scaling term
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T  # d2 term
    pvK = K * math.exp(-r * T)  # discounted strike
    return S0 * stats.norm.cdf(d1) - pvK * stats.norm.cdf(d2)


def bsm_characteristic_function(u, T, r, sigma):
    # Characteristic function of X_T = log(S_T / S_0) under the risk-neutral measure.
    drift = (r - 0.5 * sigma * sigma) * T  # risk-neutral drift term
    var = sigma * sigma * T  # variance of the log-return
    return np.exp(1j * u * drift - 0.5 * var * u * u)


def bsm_characteristic_function_ln_st(u, S0, T, r, sigma):
    # Characteristic function of ln(S_T) under the risk-neutral measure.
    x0 = math.log(S0)  # initial log-level
    return np.exp(1j * u * x0) * bsm_characteristic_function(u, T, r, sigma)


def lewis_integrand(u, S0, K, T, r, sigma):
    # Integrand for Lewis (2001) call pricing formula in the BSM model.
    k = math.log(S0 / K)  # log-moneyness in the exponential term
    cf = bsm_characteristic_function(u - 0.5j, T, r, sigma)  # shifted CF
    num = np.exp(1j * u * k) * cf  # complex numerator
    den = u * u + 0.25  # denominator to ensure integrability
    return (num / den).real


def bsm_call_value_lewis(S0, K, T, r, sigma, u_max=100.0):
    # BSM call value via the Lewis (2001) Fourier integral representation.
    int_value = quad(lambda u: lewis_integrand(u, S0, K, T, r, sigma), 0.0, u_max)[0]
      # numerical integral approximation
    factor = math.exp(-r * T) * math.sqrt(S0 * K) / math.pi  # integral prefactor
    value = S0 - factor * int_value  # Lewis call value
    return max(0.0, float(value))


def carr_madan_fft_call_grid(S0, T, r, sigma, alpha=1.5, N=4096, eta=0.25):
    # Computes a grid of call values via the Carr-Madan FFT method.
    lam = 2.0 * math.pi / (N * eta)  # log-strike spacing
    b = 0.5 * N * lam  # upper bound for log-strikes

    v = eta * np.arange(N)  # Fourier grid
    k = -b + lam * np.arange(N)  # log-strike grid k = log(K)

    u = v - 1j * (alpha + 1.0)  # complex argument for the CF
    cf = bsm_characteristic_function_ln_st(u, S0, T, r, sigma)
      # characteristic function values

    denom = (alpha * alpha + alpha - v * v) + 1j * (2.0 * alpha + 1.0) * v
      # Carr-Madan denominator
    psi = math.exp(-r * T) * cf / denom  # transform of damped call price

    weights = np.ones(N)  # trapezoidal weights
    weights[0] = 0.5  # half weight at the first point
    integrand = np.exp(1j * b * v) * psi * eta * weights
      # integrand for the FFT

    fft_values = fft(integrand).real  # FFT output, real part is used
    C = np.exp(-alpha * k) / math.pi * fft_values  # undamp the call prices

    K_grid = np.exp(k)  # strike grid
    return K_grid, C


def bsm_call_value_fft(S0, K, T, r, sigma, alpha=1.5, N=4096, eta=0.25):
    # Approximates a single call value via Carr-Madan FFT and interpolation.
    K_grid, C_grid = carr_madan_fft_call_grid(S0, T, r, sigma, alpha, N, eta)
    return float(np.interp(K, K_grid, C_grid))
