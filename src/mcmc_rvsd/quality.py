"""MCMC convergence quality labels from the effective sample size ``L / tau``.

The labels are those of the paper (Section: Quality Labels). They are defined on the
number of independent samples of the **DIB** parameters, ``L / tau_DIB``, where ``L`` is
the chain length (the number of production steps) and ``tau_DIB`` collects the
autocorrelation times of the three DIB parameters (centre, depth, width)::

    Success            if  max(L / tau_DIB) >= 50
    Severe Degeneracy  if  min(L / tau_DIB) <= 25
    Mild Degeneracy    otherwise

Both conditions use the three DIB parameters together: ``max`` for Success, because one
well-sampled DIB parameter is enough to accept a source, and ``min`` for Severe, because
one badly sampled DIB parameter is enough to reject it.

For the catalogue chain length ``L = 5000`` this is equivalent to
``min(tau_DIB) <= 100`` -> Success and ``max(tau_DIB) >= 200`` -> Severe, but the
thresholds move with ``L``: always pass the chain length that produced the chains
instead of hardcoding an autocorrelation time.  (A common mistake is to use ``max(tau)``
for Success and the fixed values 100/200 - that is only correct for ``L = 5000`` and,
for Success, only when all three DIB parameters are equally well sampled.)

A large autocorrelation time means that the chain needs many steps per independent
sample, i.e. the parameters are poorly constrained (degenerate).
"""
from __future__ import annotations

import numpy as np

SUCCESS, MILD, SEVERE = 'Success', 'Mild Degeneracy', 'Severe Degeneracy'
SUCCESS_NEFF = 50.0     # max(L / tau_DIB) >= 50  ->  Success
SEVERE_NEFF = 25.0      # min(L / tau_DIB) <= 25  ->  Severe Degeneracy


def n_effective(tau, n_steps):
    """Effective number of independent samples, ``N_eff = n_steps / tau``."""
    return np.asarray(n_steps, dtype=float) / np.asarray(tau, dtype=float)


def dib_autocorr_times(tau):
    """Autocorrelation times of the three DIB parameters.

    The parameter vector is built with the stellar lines first and the DIB last, so the
    DIB parameters are the last three entries (centre, depth, width).
    """
    tau = np.asarray(tau, dtype=float).ravel()
    if tau.size < 3:
        raise ValueError('need at least the three DIB autocorrelation times')
    return tau[-3:]


def classify(tau, n_steps, success_neff=SUCCESS_NEFF, severe_neff=SEVERE_NEFF):
    """Classify a fit from its DIB autocorrelation times and the chain length.

    Parameters
    ----------
    tau : array_like
        Autocorrelation times of all fitted parameters, in MCMC steps.  The last three
        entries must be the DIB parameters (see :func:`dib_autocorr_times`).
    n_steps : int
        Chain length used for the estimate, i.e. the number of production steps.
    success_neff, severe_neff : float
        Thresholds on ``L / tau_DIB``; the defaults are the values of the paper.

    Returns
    -------
    str
        ``'Success'``, ``'Mild Degeneracy'`` or ``'Severe Degeneracy'``.

    Examples
    --------
    >>> classify([60., 60., 60., 62., 59., 68.], 5000)          # min(tau_DIB) = 59
    'Success'
    >>> classify([170., 160., 150., 156., 160., 156.], 5000)     # no condition met
    'Mild Degeneracy'
    >>> classify([300., 206., 237., 221., 236., 135.], 5000)     # max(tau_DIB) = 236
    'Severe Degeneracy'
    """
    tau_dib = dib_autocorr_times(tau)
    neff = n_effective(tau_dib, n_steps)
    if np.nanmax(neff) >= success_neff:
        return SUCCESS
    if np.nanmin(neff) <= severe_neff:
        return SEVERE
    return MILD


def classify_tau(tau, n_steps=5000.0, **kwargs):
    """Alias of :func:`classify` kept for readability at call sites."""
    return classify(tau, n_steps, **kwargs)
