#
# Merton (1976) Jump Diffusion: Pricing Building Blocks
# dawp_pIIIa_ch08_m76_pricing.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import math  # elementary math functions

import numpy as np  # numerical arrays
from numpy.fft import fft  # fast Fourier transform
from scipy.integrate import quad  # numerical integration


def m76_char_func(u, T, r, sigma, lamb, mu, delta):
    # Characteristic function of X_T = log(S_T / S_0) under risk-neutral dynamics.
    kappa = math.exp(mu + 0.5 * delta * delta) - 1.0  # mean jump size minus one
    omega = r - 0.5 * sigma * sigma - lamb * kappa
      # risk-neutral drift of log-return

    diff = 1j * u * omega - 0.5 * sigma * sigma * u * u  # diffusion term
    jump = lamb * (np.exp(1j * u * mu - 0.5 * delta * delta * u * u) - 1.0)
      # compound Poisson jump term
    return np.exp((diff + jump) * T)


def m76_call_value_int(S0, K, T, r, sigma, lamb, mu, delta, u_max=50.0):
    # European call value via the Lewis (2001) Fourier integral representation.
    def integrand(u):
        # Integrand for the M76 Lewis integral pricing formula.
        cf = m76_char_func(u - 0.5j, T, r, sigma, lamb, mu, delta)
        num = np.exp(1j * u * math.log(S0 / K)) * cf  # complex numerator
        den = u * u + 0.25  # integrability denominator
        return (num / den).real

    int_value = quad(lambda x: integrand(x), 0.0, u_max, limit=250)[0]
      # numerical integral approximation
    factor = math.exp(-r * T) * math.sqrt(S0 * K) / math.pi  # prefactor
    return float(S0 - factor * int_value)


def m76_call_value_fft(S0, K, T, r, sigma, lamb, mu, delta, g=2):
    # European call value via Carr-Madan FFT method (single strike via grid lookup).
    k = math.log(K / S0)  # log-strike in terms of K/S0
    N = g * 4096  # FFT grid size
    eps = (g * 150.0) ** -1  # log-strike grid spacing
    eta = 2.0 * math.pi / (N * eps)  # Fourier grid spacing
    b = 0.5 * N * eps - k  # shift for strike alignment

    u = np.arange(1, N + 1)  # grid indices
    v0 = eta * (u - 1)  # Fourier grid

    if S0 >= 0.95 * K:
        alpha = 1.5  # damping parameter for ITM
        v = v0 - (alpha + 1.0) * 1j  # complex argument for CF
        mod_cf = math.exp(-r * T) * m76_char_func(v, T, r, sigma, lamb, mu, delta)
          # discounted characteristic function
        den = alpha * alpha + alpha - v0 * v0 + 1j * (2.0 * alpha + 1.0) * v0
          # Carr-Madan denominator
        psi = mod_cf / den  # modified transform
    else:
        alpha = 1.1  # damping parameter for OTM
        v = (v0 - 1j * alpha) - 1j  # complex argument for CF
        term = 1.0 / (1.0 + 1j * (v0 - 1j * alpha))
        term -= math.exp(r * T) / (1j * (v0 - 1j * alpha))
        term -= m76_char_func(v, T, r, sigma, lamb, mu, delta) / (
            (v0 - 1j * alpha) ** 2 - 1j * (v0 - 1j * alpha)
        )
        psi_1 = math.exp(-r * T) * term  # first modified transform

        v = (v0 + 1j * alpha) - 1j  # complex argument for CF
        term = 1.0 / (1.0 + 1j * (v0 + 1j * alpha))
        term -= math.exp(r * T) / (1j * (v0 + 1j * alpha))
        term -= m76_char_func(v, T, r, sigma, lamb, mu, delta) / (
            (v0 + 1j * alpha) ** 2 - 1j * (v0 + 1j * alpha)
        )
        psi_2 = math.exp(-r * T) * term  # second modified transform

    delt = np.zeros(N, dtype=float)  # Simpson delta vector
    delt[0] = 1.0  # first element
    j = np.arange(1, N + 1)  # index vector for Simpson weights
    simpson_w = (3.0 + (-1.0) ** j - delt) / 3.0  # Simpson weights

    if S0 >= 0.95 * K:
        fft_func = np.exp(1j * b * v0) * psi * eta * simpson_w
          # FFT input function
        payoff = fft(fft_func).real  # FFT output
        c_m = np.exp(-alpha * k) / math.pi * payoff  # call values on the grid
    else:
        fft_func = np.exp(1j * b * v0) * (psi_1 - psi_2) * 0.5 * eta * simpson_w
          # FFT input function
        payoff = fft(fft_func).real  # FFT output
        c_m = payoff / (np.sinh(alpha * k) * math.pi)  # call values on the grid

    pos = int((k + b) / eps)  # strike position on the grid
    return float(c_m[pos] * S0)


def m76_call_value_mcs(
    S0,
    K,
    T,
    r,
    sigma,
    lamb,
    mu,
    delta,
    n_steps,
    n_paths,
    seed=10,
):
    # European call value via Monte Carlo simulation in the Merton model.
    dt = T / n_steps  # time step length
    disc = math.exp(-r * T)  # discount factor to time 0

    kappa = math.exp(mu + 0.5 * delta * delta) - 1.0  # mean jump size minus one
    drift = (r - lamb * kappa - 0.5 * sigma * sigma) * dt
      # risk-neutral drift for log Euler
    vol = sigma * math.sqrt(dt)  # diffusion scale per step

    rng = np.random.default_rng(seed)  # random number generator
    Z = rng.standard_normal((n_steps, n_paths))  # diffusion shocks
    J = rng.standard_normal((n_steps, n_paths))  # jump size shocks
    N = rng.poisson(lamb * dt, (n_steps, n_paths))  # jump counts

    log_jump = mu * N + delta * np.sqrt(N) * J  # aggregated log jump sizes per step
    log_increments = drift + vol * Z + log_jump  # log increments

    S_T = S0 * np.exp(np.sum(log_increments, axis=0))  # terminal stock levels
    payoff = np.maximum(S_T - K, 0.0)  # terminal call payoff
    return float(disc * np.mean(payoff))
