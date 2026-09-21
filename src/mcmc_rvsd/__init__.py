"""MCMC-RVSD: measuring diffuse interstellar bands in binary-star spectra.

The method separates the (moving) stellar lines from the (stationary) DIB by
fitting all epochs of a multi-epoch spectrum simultaneously with an ensemble
MCMC sampler.  Key features:

* a configurable number of stellar lines (default: one Fe I line at 6613.82 A);
* a Gaussian or Voigt profile for the DIB (the Voigt profile adds an explicit
  Lorentzian component for the line wings);
* flat priors with user-editable bounds for every parameter;
* RV-error propagation and SNR weighting when combining epochs;
* model selection with AIC/BIC and a fit-quality label from the autocorrelation
  time of the chains.

Quick start
-----------
>>> from mcmc_rvsd import SpectralModel, DIB, StellarLine, run_mcmc
>>> model = SpectralModel(stellar_lines=[StellarLine()], dib=DIB(profile='gaussian'))
>>> result = run_mcmc(model, wave, flux, flux_err, rvs, rv_errs, snrs=snrs)
>>> print(result.summary())
"""
from .models import DIB, DIB_PROFILES, SpectralModel, StellarLine
from .likelihood import WEIGHT_METHODS, log_likelihood
from .priors import normal_prior, rv_constrained_center_prior, uniform_box
from .profiles import C_KMS, gaussian_absorption, shift_wavelength, voigt_absorption
from .quality import MILD, SEVERE, SUCCESS, classify_tau, n_effective
from .sampler import MCMCResult, run_mcmc
from .selection import aic, bic, chi2_lnL, delta_ic

__version__ = '0.1.0'

__all__ = [
    'SpectralModel', 'StellarLine', 'DIB', 'DIB_PROFILES',
    'log_likelihood', 'WEIGHT_METHODS',
    'run_mcmc', 'MCMCResult',
    'classify_tau', 'n_effective', 'SUCCESS', 'MILD', 'SEVERE',
    'aic', 'bic', 'delta_ic', 'chi2_lnL',
    'gaussian_absorption', 'voigt_absorption', 'shift_wavelength', 'C_KMS',
    'uniform_box', 'normal_prior', 'rv_constrained_center_prior',
    '__version__',
]
