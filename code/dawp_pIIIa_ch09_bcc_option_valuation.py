#
# Bakshi-Cao-Chen (1997): European Call Valuation Building Blocks
# dawp_pIIIa_ch09_bcc_option_valuation.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from __future__ import annotations

import math  # elementary math functions

import numpy as np  # numerical arrays
from scipy.integrate import quad  # numerical integration

from dawp_pIIIa_ch09_cir_zcb import cir_zcb_value  # CIR discounting helper


def bsm_call_value(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    # European BSM call value with constant short rate and volatility.
    if T <= 0.0:
        return float(max(S0 - K, 0.0))

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    from math import erf  # local import to keep module dependencies minimal

    def cdf(x: float) -> float:
        # Standard normal cumulative distribution function.
        return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

    return float(S0 * cdf(d1) - K * math.exp(-r * T) * cdf(d2))


def m76_char_func(u: complex | np.ndarray, T: float, lamb: float, mu: float,
                 delta: float) -> complex | np.ndarray:
    # Characteristic function of the pure jump component in the Merton model.
    # Compensator for the jump component only.
    omega = -lamb * (math.exp(mu + 0.5 * delta * delta) - 1.0)
    jump_cf = np.exp(1j * u * omega * T)
    jump_cf *= np.exp(lamb * T * (np.exp(1j * u * mu - 0.5 * delta * delta * u * u) - 1.0))
    return jump_cf


def h93_char_func(
    u: complex | np.ndarray,
    T: float,
    r: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    v0: float,
) -> complex | np.ndarray:
    # Characteristic function of log-return X_T = log(S_T / S_0) in the Heston model.
    iu = 1j * u  # i * u
    d = np.sqrt((rho * sigma_v * iu - kappa_v) ** 2 + sigma_v * sigma_v * (iu + u * u))
    g = (kappa_v - rho * sigma_v * iu - d) / (kappa_v - rho * sigma_v * iu + d)

    exp_dt = np.exp(-d * T)  # exponential term
    C = r * iu * T
    C += (kappa_v * theta_v / (sigma_v * sigma_v)) * (
        (kappa_v - rho * sigma_v * iu - d) * T
        - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((kappa_v - rho * sigma_v * iu - d) / (sigma_v * sigma_v)) * (
        (1.0 - exp_dt) / (1.0 - g * exp_dt)
    )
    return np.exp(C + D * v0)


def bcc_char_func(
    u: complex | np.ndarray,
    T: float,
    r: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    v0: float,
    lamb: float,
    mu: float,
    delta: float,
) -> complex | np.ndarray:
    # Characteristic function of log-return in the Bates (1996) model.
    return h93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0) * m76_char_func(
        u, T, lamb, mu, delta
    )


def lewis_call_value(
    S0: float,
    K: float,
    T: float,
    r: float,
    cf_func,
    cf_args: tuple,
    u_max: float=80.0,
) -> float:
    # Generic European call value via the Lewis (2001) single-integral formula.
    k = math.log(S0 / K)  # log-moneyness

    def integrand(u: float) -> float:
        # Lewis integrand for a generic characteristic function.
        cf = cf_func(u - 0.5j, *cf_args)
        num = np.exp(1j * u * k) * cf
        den = u * u + 0.25
        return float((num / den).real)

    int_value = quad(lambda x: integrand(x), 0.0, u_max, limit=250)[0]
    factor = math.exp(-r * T) * math.sqrt(S0 * K) / math.pi
    return float(max(0.0, S0 - factor * int_value))


def h93_call_value_int(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    v0: float,
) -> float:
    # European call value in the Heston model via the Lewis (2001) formula.
    args = (T, r, kappa_v, theta_v, sigma_v, rho, v0)
    return lewis_call_value(S0, K, T, r, h93_char_func, args)


def bcc_call_value_int(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa_v: float,
    theta_v: float,
    sigma_v: float,
    rho: float,
    v0: float,
    lamb: float,
    mu: float,
    delta: float,
) -> float:
    # European call value in the Bates (1996) model via the Lewis (2001) formula.
    args = (T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
    return lewis_call_value(S0, K, T, r, bcc_char_func, args)


def cir_implied_short_rate(
    r0: float,
    kappa_r: float,
    theta_r: float,
    sigma_r: float,
    T: float,
) -> float:
    # Derives a flat short-rate proxy from the CIR discount factor B_0(T).
    disc = cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T=T, t=0.0)
    return float(-math.log(disc) / T)
