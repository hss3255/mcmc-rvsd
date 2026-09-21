"""Priors: flat boxes per parameter, plus an optional RV-constrained centre.

``SpectralModel`` already implements box priors through the ``*_bounds`` fields
of :class:`~mcmc_rvsd.models.StellarLine` and :class:`~mcmc_rvsd.models.DIB`.
This module provides the extra options used in the paper:

* :func:`rv_constrained_center_prior` - keep the stellar line centre inside the
  observed RV range of every epoch (useful when the RV span is large and the
  line would otherwise drift outside the fitted window);
* :func:`normal_prior` - a Gaussian penalty, e.g. for a line whose centre is
  known from a laboratory wavelength.
"""
from __future__ import annotations

import numpy as np

from .profiles import C_KMS


def uniform_box(value, lo, hi):
    """Flat prior: 0.0 inside ``(lo, hi)``, ``-inf`` outside."""
    return 0.0 if lo < value < hi else -np.inf


def rv_constrained_center_prior(center, rvs, rv_errs, lo=6612.5, hi=6615.0):
    """Require the shifted line centre to stay inside the observed RV range.

    The centre must satisfy ``lo < center + (rv +/- rv_err) * center / c < hi``
    for every epoch, which keeps the line inside the fitted window even for the
    most extreme velocity of the orbit.
    """
    for rv, rv_err in zip(rvs, rv_errs):
        lo_shift = center + (rv - rv_err) * center / C_KMS
        hi_shift = center + (rv + rv_err) * center / C_KMS
        if not (lo < lo_shift < hi and lo < hi_shift < hi):
            return -np.inf
    return 0.0


def normal_prior(value, mean, sigma):
    """Gaussian prior (up to the normalisation constant)."""
    return -0.5 * ((value - mean) / sigma) ** 2
