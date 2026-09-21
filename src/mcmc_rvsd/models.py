"""User-configurable forward model: N stellar lines (Gaussian) times one DIB.

The parameter vector is laid out as::

    p = [c1, A1, s1, ..., cN, AN, sN, DIB_c, DIB_A, DIB_s, (DIB_gamma)]

so the stellar lines come first (in the order given by the user) and the DIB
parameters last.  ``SpectralModel`` owns the priors: every parameter has a box
(uniform) prior whose bounds can be overridden per line, and the stellar line
centres can optionally be constrained to stay within the observed RV range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .likelihood import log_likelihood
from .profiles import C_KMS, gaussian_absorption, shift_wavelength, voigt_absorption

DIB_PROFILES = ('gaussian', 'voigt')


@dataclass
class StellarLine:
    """One stellar absorption line: Gaussian, centre moves with the RV."""

    name: str = 'FeI'
    center: float = 6613.8225
    center_bounds: tuple = (6612.5, 6615.0)
    depth_bounds: tuple = (0.001, 0.4)
    sigma_bounds: tuple = (0.001, 1.0)
    depth0: float = 0.010
    sigma0: float = 0.400

    def initial(self):
        return [self.center, self.depth0, self.sigma0]

    @property
    def bounds(self):
        return [self.center_bounds, self.depth_bounds, self.sigma_bounds]

    def flux(self, x, rv, center, depth, sigma):
        return gaussian_absorption(x, shift_wavelength(center, rv), depth, sigma)


@dataclass
class DIB:
    """The diffuse interstellar band: fixed rest wavelength, Gaussian or Voigt."""

    center: float = 6613.6603
    center_bounds: tuple = (6612.8, 6614.6)
    depth_bounds: tuple = (0.001, 0.2)
    sigma_bounds: tuple = (0.3, 1.0)
    profile: str = 'gaussian'
    gamma_bounds: tuple = (0.001, 0.3)
    depth0: float = 0.050
    sigma0: float = 0.550
    gamma0: float = 0.050

    def __post_init__(self):
        if self.profile not in DIB_PROFILES:
            raise ValueError(f'profile must be one of {DIB_PROFILES}, got {self.profile!r}')

    @property
    def n_params(self):
        return 4 if self.profile == 'voigt' else 3

    @property
    def names(self):
        n = ['lambda_DIB', 'A_DIB', 'sigma_DIB']
        return n + (['gamma_DIB'] if self.profile == 'voigt' else [])

    @property
    def bounds(self):
        b = [self.center_bounds, self.depth_bounds, self.sigma_bounds]
        return b + ([self.gamma_bounds] if self.profile == 'voigt' else [])

    def initial(self):
        p = [self.center, self.depth0, self.sigma0]
        return p + ([self.gamma0] if self.profile == 'voigt' else [])

    def flux(self, x, center, depth, sigma, gamma=None):
        if self.profile == 'voigt':
            return voigt_absorption(x, center, depth, sigma, gamma)
        return gaussian_absorption(x, center, depth, sigma)


class SpectralModel:
    """MCMC-RVSD forward model with a user-defined number of stellar lines.

    Parameters
    ----------
    stellar_lines : sequence of StellarLine, optional
        One entry per stellar line to include (default: a single Fe I line at
        6613.8225 A).  Add lines to model blends, e.g. Fe I + Y II + Th I.
    dib : DIB, optional
        DIB component; set ``profile='voigt'`` to fit a Lorentzian component.
    rv_error_propagation : bool
        Propagate the RV uncertainty into the flux uncertainty (default True).
    weight_method : {'variance_scaling', 'likelihood_weight', 'none'}
        How spectra of different SNR are combined (default 'variance_scaling').

    Examples
    --------
    >>> model = SpectralModel(stellar_lines=[StellarLine(name='FeI')],
    ...                       dib=DIB(profile='voigt'))
    >>> model.n_params
    7
    """

    def __init__(self, stellar_lines=None, dib=None,
                 rv_error_propagation=True, weight_method='variance_scaling'):
        self.stellar_lines = list(stellar_lines) if stellar_lines else [StellarLine()]
        self.dib = dib if dib is not None else DIB()
        self.rv_error_propagation = rv_error_propagation
        self.weight_method = weight_method

    # ------------------------------------------------------------------ shape
    @property
    def n_stars(self):
        return len(self.stellar_lines)

    @property
    def n_params(self):
        return 3 * self.n_stars + self.dib.n_params

    @property
    def param_names(self):
        names = []
        for line in self.stellar_lines:
            names += [f'lambda_{line.name}', f'A_{line.name}', f'sigma_{line.name}']
        return names + self.dib.names

    @property
    def bounds(self):
        b = []
        for line in self.stellar_lines:
            b += line.bounds
        return b + self.dib.bounds

    def initial_guess(self, dib=None, stars=None):
        """Return a starting parameter vector (user overrides applied)."""
        p = []
        for i, line in enumerate(self.stellar_lines):
            p += list(stars[i]) if stars and stars[i] is not None else line.initial()
        p += list(dib) if dib is not None else self.dib.initial()
        return np.asarray(p, dtype=float)

    # ------------------------------------------------------------------ model
    def flux(self, wave, rv, p):
        """Model flux for one epoch: the product of all absorption components."""
        wave = np.asarray(wave, dtype=float)
        m = np.ones_like(wave)
        for i, line in enumerate(self.stellar_lines):
            m = m * line.flux(wave, rv, p[3 * i], p[3 * i + 1], p[3 * i + 2])
        q = p[3 * self.n_stars:]
        m = m * self.dib.flux(wave, q[0], q[1], q[2], q[3] if self.dib.profile == 'voigt' else None)
        return m

    def component_fluxes(self, wave, rv, p):
        """Individual components (for plots): stellar lines, DIB, and the product."""
        wave = np.asarray(wave, dtype=float)
        out = {}
        for i, line in enumerate(self.stellar_lines):
            out[line.name] = line.flux(wave, rv, p[3 * i], p[3 * i + 1], p[3 * i + 2])
        q = p[3 * self.n_stars:]
        out['DIB'] = self.dib.flux(wave, q[0], q[1], q[2],
                                   q[3] if self.dib.profile == 'voigt' else None)
        out['total'] = self.flux(wave, rv, p)
        return out

    # --------------------------------------------------------------- priors
    def lnprior(self, p):
        """Flat (box) prior: 0 inside the bounds of every parameter, -inf outside."""
        p = np.asarray(p, dtype=float)
        if not np.all(np.isfinite(p)):
            return -np.inf
        for value, (lo, hi) in zip(p, self.bounds):
            if not (lo < value < hi):
                return -np.inf
        return 0.0

    # ------------------------------------------------------------ likelihood
    def lnlike(self, p, wave, flux, flux_err, rvs, rv_errs, snrs=None):
        return log_likelihood(self, p, wave, flux, flux_err, rvs, rv_errs, snrs=snrs,
                              weight_method=self.weight_method,
                              rv_error_propagation=self.rv_error_propagation)

    def lnprob(self, p, wave, flux, flux_err, rvs, rv_errs, snrs=None):
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf
        total = lp + self.lnlike(p, wave, flux, flux_err, rvs, rv_errs, snrs=snrs)
        return -np.inf if not np.isfinite(total) else total

    # ------------------------------------------------------------------ info
    def __repr__(self):
        stars = ', '.join(f'{l.name}@{l.center:.4f}' for l in self.stellar_lines)
        return (f'SpectralModel(stellar_lines=[{stars}], dib={self.dib.profile}, '
                f'n_params={self.n_params})')
