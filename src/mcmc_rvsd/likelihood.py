"""Log-likelihood with RV error propagation and optional SNR weighting.

This is the likelihood used throughout the paper: the flux uncertainty is the
quadrature sum of the measurement error and the uncertainty induced by the RV
error (through the local gradient of the model), and the epochs can be combined
with different SNR weighting schemes.
"""
from __future__ import annotations

import numpy as np

from .profiles import C_KMS

WEIGHT_METHODS = ('variance_scaling', 'likelihood_weight', 'none')


def _model_array(model, wave, p, rvs):
    """Model flux for every epoch: shape (n_epochs, n_pixels)."""
    return np.array([model.flux(wave, rv, p) for rv in rvs])


def log_likelihood(model, p, wave, flux, flux_err, rvs, rv_errs, snrs=None,
                   weight_method='variance_scaling', rv_error_propagation=True):
    """Gaussian log-likelihood with RV-error propagation and SNR weighting.

    Parameters
    ----------
    model : SpectralModel
        Forward model (any object exposing ``flux(wave, rv, p)``).
    p : array_like
        Parameter vector.
    wave : array_like
        Wavelength grid (Angstrom), shared by all epochs.
    flux : array_like, shape (n_epochs, n_pixels)
        Normalised flux for each epoch.
    flux_err : array_like, shape (n_epochs, n_pixels)
        Flux uncertainty for each epoch.
    rvs, rv_errs : array_like, shape (n_epochs,)
        Radial velocity and its uncertainty (km/s).
    snrs : array_like, shape (n_epochs,), optional
        Signal-to-noise ratio of each epoch (needed unless weight_method='none').
    weight_method : {'variance_scaling', 'likelihood_weight', 'none'}
        'variance_scaling'  - divide the effective variance by the SNR weight
                              (higher SNR -> smaller variance), weights are
                              normalised to mean 1 so the average variance is
                              preserved;
        'likelihood_weight' - multiply the per-epoch log-likelihood by the
                              normalised SNR weight;
        'none'              - plain, unweighted combination of the epochs.
    rv_error_propagation : bool
        Add the RV-induced flux uncertainty, approximated as
        ``|d(model)/d(lambda) * lambda * sigma_RV / c|`` (default True).

    Returns
    -------
    float
        The log-likelihood (``-inf`` for invalid input).
    """
    if weight_method not in WEIGHT_METHODS:
        raise ValueError(f'weight_method must be one of {WEIGHT_METHODS}, got {weight_method!r}')

    epsilon = 1e-9
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    rvs = np.asarray(rvs, dtype=float)
    rv_errs = np.asarray(rv_errs, dtype=float)
    if np.any(flux_err <= 0):
        return -np.inf

    model_spectra = _model_array(model, wave, p, rvs)

    total_variance = flux_err ** 2 + epsilon
    if rv_error_propagation:
        d_model_d_lambda = np.gradient(model_spectra, axis=1) / np.gradient(np.asarray(wave, dtype=float))
        mean_lambda = float(np.mean(wave))
        lambda_err_from_rv = mean_lambda * (rv_errs[:, np.newaxis] / C_KMS)
        flux_err_from_rv = np.abs(d_model_d_lambda * lambda_err_from_rv)
        total_variance = total_variance + flux_err_from_rv ** 2

    if weight_method == 'none' or snrs is None:
        return -0.5 * np.sum((flux - model_spectra) ** 2 / total_variance + np.log(total_variance))

    snrs = np.asarray(snrs, dtype=float)
    if weight_method == 'likelihood_weight':
        weights = snrs / np.sum(snrs)
        return -0.5 * np.sum(weights[:, np.newaxis]
                             * ((flux - model_spectra) ** 2 / total_variance + np.log(total_variance)))

    # variance_scaling (default): weights are normalised to mean 1
    scale = float(np.mean(snrs))
    weights = snrs / scale if scale > 0 else np.ones(len(rvs))
    weights = np.maximum(weights, 1e-6)
    effective_variance = total_variance / weights[:, np.newaxis]
    return -0.5 * np.sum((flux - model_spectra) ** 2 / effective_variance + np.log(effective_variance))
