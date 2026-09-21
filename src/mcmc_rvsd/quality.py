"""MCMC convergence quality labels based on the autocorrelation time.

The labels used in the paper are defined on the autocorrelation time of the DIB
chains (the same rule stored in the published catalog header)::

    tau <= 100   -> Success
    tau <= 200   -> Mild Degeneracy
    tau >  200   -> Severe Degeneracy

A large autocorrelation time means the chain needs many steps to produce an
independent sample, i.e. the parameters are poorly constrained (degenerate).
"""
from __future__ import annotations

import numpy as np

SUCCESS, MILD, SEVERE = 'Success', 'Mild Degeneracy', 'Severe Degeneracy'
SUCCESS_TAU = 100.0
MILD_TAU = 200.0


def classify_tau(tau, success_tau=SUCCESS_TAU, mild_tau=MILD_TAU):
    """Classify one autocorrelation time into a quality label.

    Parameters
    ----------
    tau : float
        Autocorrelation time (in MCMC steps) of the DIB parameters.
    success_tau, mild_tau : float
        Thresholds; the defaults are the values used for the published catalog.
    """
    tau = float(np.nanmax(tau))
    if tau <= success_tau:
        return SUCCESS
    if tau <= mild_tau:
        return MILD
    return SEVERE


def n_effective(tau, n_steps):
    """Number of independent samples, ``N_eff = n_steps / tau``."""
    return np.asarray(n_steps, dtype=float) / np.asarray(tau, dtype=float)
