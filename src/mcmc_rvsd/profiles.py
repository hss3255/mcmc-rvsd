"""Analytic line profiles used by the MCMC-RVSD forward model.

The stellar lines are Gaussian and move with the radial velocity; the DIB is
centred at a fixed (rest) wavelength and can be modelled either as a Gaussian or
as a Voigt profile (which adds an explicit Lorentzian component, needed when the
line wings carry a non-negligible fraction of the equivalent width).
"""
from __future__ import annotations

import numpy as np
from scipy.special import wofz

C_KMS = 299792.458  # speed of light (km/s)


def shift_wavelength(center, rv):
    """Doppler-shift a rest wavelength ``center`` (Angstrom) by ``rv`` (km/s)."""
    return center + rv * center / C_KMS


def gaussian_absorption(x, center, depth, sigma):
    """Gaussian absorption line, normalised to 1 in the continuum.

    Parameters
    ----------
    x : array_like
        Wavelength grid (Angstrom).
    center : float
        Line centre (Angstrom).
    depth : float
        Flux decrement at the line centre.
    sigma : float
        Gaussian width of the line (Angstrom).
    """
    return 1.0 - depth * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def voigt_absorption(x, center, depth, sigma, gamma):
    """Voigt absorption line, peak-normalised to ``depth``.

    Parameters
    ----------
    sigma : float
        Gaussian width (Angstrom) of the Voigt profile.
    gamma : float
        Lorentzian half-width at half maximum (Angstrom) of the Voigt profile.

    Notes
    -----
    The profile is divided by its value at the line centre, so that ``depth`` is
    exactly the flux decrement at the centre (the same convention as the mock
    data used in the paper).
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    v = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    v0 = np.real(wofz(1j * gamma / (sigma * np.sqrt(2)))) / (sigma * np.sqrt(2 * np.pi))
    return 1.0 - depth * v / v0
