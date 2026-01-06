# Code Overview

This folder contains the Python scripts and small data files used throughout *Derivatives Analytics with Python*. Many of the scripts are designed to regenerate figures or provide reusable valuation and calibration functions.

## Part I — The Market

- `dawp_pI_inner_value_plot.py` — plots the inner value curve of a European call as a function of terminal index level.
- `dawp_pI_bsm_vs_inner_plot.py` — compares inner value at maturity with a benchmark Black–Scholes–Merton present-value curve.
- `dawp_pI_spy_tlt_stylized_facts.py` — computes and plots stylised facts (returns, volatility, and rate proxy) for SPY and TLT.

## Part II — Arbitrage Pricing

- `dawp_pII_bsm_values_plot.py` — generates BSM call and put value curves across underlying levels.
- `dawp_pII_bsm_greeks_plot.py` — plots selected BSM Greeks (delta, gamma, vega) as functions of the underlying.
- `dawp_pII_ch05_bsm_option_valuation.py` — provides BSM valuation functions and helper routines for European options.
- `dawp_pII_ch05_bsm_call_greeks.py` — implements BSM call Greeks based on the valuation functions.
- `dawp_pII_ch05_crr_option_valuation.py` — prices European options in the Cox–Ross–Rubinstein binomial model.
- `dawp_pII_ch05_crr_loop_demo.py` — demonstrates CRR valuation via explicit loops.
- `dawp_pII_ch05_crr_vectorized_demo.py` — demonstrates CRR valuation using vectorised NumPy operations.
- `dawp_pII_crr_convergence_plot.py` — compares CRR call values to BSM and illustrates convergence as the number of steps increases.
- `dawp_pII_ch06_roots_unity_plot.py` — visualises roots of unity used in discrete Fourier transforms.
- `dawp_pII_ch06_fourier_pricing_bsm.py` — implements Fourier-based BSM call valuation via characteristic functions.
- `dawp_pII_ch06_fourier_compare_plot.py` — compares analytical BSM, Lewis integral, and Carr–Madan FFT prices across strikes.
- `dawp_pII_ch07_crr_american_options.py` — prices American options in a CRR tree.
- `dawp_pII_ch07_lsm_primal.py` — implements a primal Longstaff–Schwartz estimator for American options.
- `dawp_pII_ch07_lsm_convergence_plot.py` — studies convergence of LSM American put values with increasing paths.
- `dawp_pII_ch07_short_condor_payoff_plot.py` — plots the payoff of a short condor spread used in the American-option chapter.

## Part IIIa — Market-Based Valuation (Merton, CIR, Heston)

- `dawp_pIIIa_ch08_m76_valuation_int.py` — values Merton jump-diffusion calls via numerical integration (Lewis formula).
- `dawp_pIIIa_ch08_m76_valuation_fft.py` — values Merton jump-diffusion calls via Carr–Madan FFT.
- `dawp_pIIIa_ch08_m76_valuation_mcs.py` — values Merton jump-diffusion calls by Monte Carlo simulation.
- `dawp_pIIIa_ch08_m76_pricing.py` — convenience wrapper for Merton pricing routines across methods.
- `dawp_pIIIa_ch08_calibration_fft_3m.py` — calibrates the Merton model to option data using FFT-based pricing.
- `dawp_pIIIa_ch08_fig01_m76_methods_compare_plot.py` — compares Merton call values across methods (integration, FFT, Monte Carlo) for a figure.
- `dawp_pIIIa_ch08_option_data.h5` — example Merton calibration data set in HDF5 format.
- `dawp_pIIIa_ch09_cir_zcb.py` — implements CIR zero-coupon bond valuation and yields.
- `dawp_pIIIa_ch09_fig03_cir_zcb_term_structure_plot.py` — plots CIR discount factors and term structures.
- `dawp_pIIIa_ch09_bcc_option_valuation.py` — provides transform-based valuation routines for the BCC-style framework and special cases.
- `dawp_pIIIa_ch09_fig04_bcc_special_cases_plot.py` — compares option values across BSM, Merton, Heston, and Bates-type models.
- `dawp_pIIIa_ch10_cir_mcs.py` — Monte Carlo simulation for the CIR short-rate model and bond valuation.
- `dawp_pIIIa_ch10_fig05_cir_paths_plot.py` — plots CIR rate paths for the Monte Carlo chapter.
- `dawp_pIIIa_ch10_fig06_zcb_mcs_vs_analytical_plot.py` — compares CIR Monte Carlo bond values to analytical term-structure formulas.
- `dawp_pIIIa_ch10_h93_mcs.py` — Monte Carlo valuation for Heston-style stochastic volatility (H93).
- `dawp_pIIIa_ch10_fig07_h93_mcs_convergence_plot.py` — studies convergence of Heston Monte Carlo call values towards transform benchmarks.

## Part IIIb — Market-Based Valuation (Calibration and Hedging)

- `dawp_pIIIb_ch11_bcc_calibration.py` — calibrates the BCC-style framework to an index option surface using transform-based pricing.
- `dawp_pIIIb_ch11_bcc_params.npy` — example calibrated parameter vector saved as a NumPy array.
- `dawp_pIIIb_ch12_bcc_mcs.py` — Monte Carlo simulation engine for the calibrated BCC-style model (rates, volatility, and index paths).
- `dawp_pIIIb_ch13_bsm_hedging.py` — dynamic delta hedging experiments for an American put under BSM dynamics using LSM.
- `dawp_pIIIb_ch13_bcc_hedging.py` — dynamic delta hedging experiments for an American put under calibrated BCC dynamics, including hedging P&L diagnostics.

Each script is designed to be readable and reusable. You can import the functions into your own projects or run the scripts directly to regenerate figures and numerical examples from the book.

