# Derivatives Analytics with Python — Code & Notebooks

This repository contains the Jupyter notebooks and Python scripts that accompany the *Derivatives Analytics with Python* book and class in the CPF Program. The material is organised to mirror the book structure:

- Part I — The Market
- Part II — Arbitrage Pricing
- Part III — Market-Based Valuation

The notebooks reproduce the main valuation, calibration, and hedging examples used in class, while the scripts provide focused, reusable implementations for figures and numerical experiments.

## Structure

- `notebooks/` — chapter notebooks (`dawp_chapter_*.ipynb`) that combine text, formulas, and code.
- `code/` — standalone Python modules and helper scripts used for figures, valuation routines, and calibration.

See the `README.md` files inside `notebooks/` and `code/` for concise per-file overviews.

## Usage

The notebooks are designed to run in a standard Scientific Python environment (or in Google Colab) with the usual stack:

- Python 3.11+
- `numpy`, `pandas`, `matplotlib`
- `scipy`, `numba` (where used)
- `h5py` for reading HDF5 option data

The scripts under `code/` are written so that you can either run them as standalone programs (for example to regenerate figures) or import their functions into your own projects.

## Disclaimer

This repository and its contents are provided for educational and illustrative purposes only and come without any warranty or guarantees of any kind — express or implied. Use at your own risk. The authors and The Python Quants GmbH are not responsible for any direct or indirect damages, losses, or issues arising from the use of this code. Do not use the provided examples for critical decision‑making, financial transactions, medical advice, or production deployments without rigorous review, testing, and validation.

Some examples may reference third‑party libraries, datasets, services, or APIs subject to their own licenses and terms; you are responsible for ensuring compliance.

## Contact

- Email: `team@tpq.io`
- Linktree: `linktr.ee/dyjh`
- CPF Program: `python-for-finance.com`
- The AI Engineer: `theaiengineer.dev`
- The Crypto Engineer: `thecryptoengineer.dev`
