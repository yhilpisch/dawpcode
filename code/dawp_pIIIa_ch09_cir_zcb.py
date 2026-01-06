#
# Cox-Ingersoll-Ross (1985): Zero-Coupon Bond Valuation
# dawp_pIIIa_ch09_cir_zcb.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

from __future__ import annotations

import math  # elementary math functions


def _gamma(kappa_r: float, sigma_r: float) -> float:
    # Computes the CIR gamma term used in the affine bond coefficients.
    return math.sqrt(kappa_r * kappa_r + 2.0 * sigma_r * sigma_r)


def cir_zcb_value(
    r0: float,
    kappa_r: float,
    theta_r: float,
    sigma_r: float,
    T: float,
    t: float=0.0,
) -> float:
    # Values a unit zero-coupon bond B_t(T) in the CIR short-rate model.
    if T <= t:
        raise ValueError("Require T > t for a non-trivial bond.")

    tau = T - t  # time to maturity
    g = _gamma(kappa_r, sigma_r)  # CIR gamma

    exp_gt = math.exp(g * tau)  # helper exponential
    denom = 2.0 * g + (kappa_r + g) * (exp_gt - 1.0)  # common denominator

    b2 = 2.0 * (exp_gt - 1.0) / denom  # CIR B(tau)
    b1 = ((2.0 * g * math.exp((kappa_r + g) * tau / 2.0)) / denom) ** (
        2.0 * kappa_r * theta_r / (sigma_r * sigma_r)
    )  # CIR A(tau)

    if t == 0.0:
        r_t = r0  # at time 0 the short rate is known
    else:
        # Expected short rate, used as proxy for r_t.
        r_t = theta_r + math.exp(-kappa_r * t) * (r0 - theta_r)

    return float(b1 * math.exp(-b2 * r_t))


def cir_zcb_yield(
    r0: float,
    kappa_r: float,
    theta_r: float,
    sigma_r: float,
    T: float,
    t: float=0.0,
) -> float:
    # Computes the continuously compounded yield y_t(T) from the ZCB value.
    zcb = cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T=T, t=t)
    tau = T - t  # time to maturity
    return float(-math.log(zcb) / tau)
