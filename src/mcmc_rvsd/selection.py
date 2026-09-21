"""Information criteria for model selection (how many stellar lines are needed).

Adding a stellar line always improves the fit, so the number of components is
chosen with the Akaike and Bayesian information criteria::

    AIC = 2k - 2 lnL
    BIC = k ln(n) - 2 lnL

with ``k`` the number of free parameters and ``n`` the number of fitted points.
A lower value is preferred; the model with the lower AIC/BIC wins.  In the paper
a difference of ~125 in favour of the Voigt profile was found for a DIB with a
Lorentzian component, i.e. decisively more than the usual 10 threshold.
"""
from __future__ import annotations

import numpy as np


def aic(lnL, k):
    """Akaike information criterion."""
    return 2.0 * k - 2.0 * lnL


def bic(lnL, k, n):
    """Bayesian information criterion (``n`` = number of fitted data points)."""
    return k * np.log(n) - 2.0 * lnL


def delta_ic(ic_new, ic_ref):
    """Difference ``ic_new - ic_ref``; negative means the new model is preferred."""
    return ic_new - ic_ref


def best_sample_lnL(model, samples, wave, flux, flux_err, rvs, max_samples=2000, thin=None):
    """Log-likelihood of the best-fitting sample of a chain.

    For model comparison the information criteria must be evaluated at a *joint*
    estimate of the parameters.  The marginal posterior medians are fine for a
    well behaved posterior, but for many parameters that are partly degenerate
    (e.g. several blended stellar lines) the combination of the individual
    medians is not a good fit at all.  This helper returns the largest
    log-likelihood found in the chain, which is a safe joint estimate.

    Returns
    -------
    (lnL_best, p_best)
    """
    samples = np.asarray(samples, dtype=float)
    step = thin or max(1, len(samples) // max_samples)
    best_lnL, best_p = -np.inf, None
    for p in samples[::step]:
        value = chi2_lnL(model, p, wave, flux, flux_err, rvs)
        if value > best_lnL:
            best_lnL, best_p = value, p
    return best_lnL, best_p


def chi2_lnL(model, p, wave, flux, flux_err, rvs):
    """Log-likelihood in the form used for the information criteria in the paper.

    The normalisation term ``log(variance)`` is dropped, exactly as in the
    analysis scripts, so that AIC/BIC values are directly comparable with the
    published numbers.
    """
    flux = np.asarray(flux, dtype=float)
    total = 0.0
    for i, rv in enumerate(rvs):
        mod = model.flux(wave, rv, p)
        total += -0.5 * np.sum((flux[i] - mod) ** 2 / np.asarray(flux_err[i], dtype=float) ** 2)
    return total
