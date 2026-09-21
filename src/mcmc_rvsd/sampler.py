"""emcee driver: run the MCMC, summarise the posterior, label the fit quality."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import emcee

from .quality import classify_tau
from .selection import aic, best_sample_lnL, bic, chi2_lnL


@dataclass
class MCMCResult:
    """Posterior summary of one MCMC-RVSD run."""

    param_names: list
    pfit: np.ndarray
    perr: np.ndarray
    p16: np.ndarray
    p84: np.ndarray
    samples: np.ndarray
    tau: np.ndarray
    lnL: float
    aic: float
    bic: float
    label: str
    nwalkers: int
    burnin: int
    production: int
    runtime: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(param_names=list(self.param_names), pfit=self.pfit.tolist(),
                    perr=self.perr.tolist(), p16=self.p16.tolist(), p84=self.p84.tolist(),
                    tau=self.tau.tolist(), lnL=self.lnL, aic=self.aic, bic=self.bic,
                    label=self.label, nwalkers=self.nwalkers, burnin=self.burnin,
                    production=self.production, runtime=self.runtime)

    def summary(self):
        lines = [f'{"parameter":>16} {"value":>14} {"+/-":>12}',
                 '-' * 46]
        for n, v, e in zip(self.param_names, self.pfit, self.perr):
            lines.append(f'{n:>16} {v:>14.5f} {e:>12.5f}')
        lines += ['-' * 46,
                  f'lnL = {self.lnL:.2f} | AIC = {self.aic:.2f} | BIC = {self.bic:.2f}',
                  f'tau = {np.round(self.tau, 1)} | label = {self.label}',
                  f'walkers = {self.nwalkers} | burn-in = {self.burnin} | production = {self.production}']
        return '\n'.join(lines)

    def save(self, path, thin=10):
        """Save the result; ``thin`` keeps every n-th sample to limit the file size."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        payload = dict(samples=np.asarray(self.samples[::thin], dtype=np.float32),
                       pfit=self.pfit, perr=self.perr, p16=self.p16, p84=self.p84,
                       tau=self.tau, lnL=self.lnL, aic=self.aic, bic=self.bic,
                       label=self.label, param_names=np.array(self.param_names),
                       nwalkers=self.nwalkers, burnin=self.burnin,
                       production=self.production, thin=thin)
        for key in ('lnL_best', 'p_best'):     # filled in by run_mcmc
            if key in self.extra:
                payload[key] = self.extra[key]
        np.savez_compressed(path, **payload)
        return path


def initial_positions(model, p0, nwalkers, scale=0.01, rng=None, max_tries=10000):
    """Scatter walkers around ``p0``, rejecting positions outside the priors."""
    rng = rng or np.random.default_rng()
    p0 = np.asarray(p0, dtype=float)
    ndim = p0.size
    pos = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        for _ in range(max_tries):
            candidate = p0 + rng.normal(0, scale, ndim)
            if np.isfinite(model.lnprior(candidate)):
                pos[i] = candidate
                break
        else:  # pragma: no cover - only for a degenerate p0
            raise RuntimeError('could not draw valid initial positions; check p0 and the bounds')
    return pos


def run_mcmc(model, wave, flux, flux_err, rvs, rv_errs, p0=None, snrs=None,
             nwalkers=None, burnin=1500, production=5000, seed=None, progress=False,
             scale=0.01):
    """Run the ensemble sampler and return an :class:`MCMCResult`.

    The default setup is the one used for the paper: 48 walkers (8 x ndim),
    1500 burn-in and 5000 production steps.
    """
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if np.sum(flux_err) == 0.0:
        flux_err = 1e-3 * np.ones_like(flux)
    p0 = model.initial_guess() if p0 is None else np.asarray(p0, dtype=float)
    ndim = p0.size
    nwalkers = nwalkers or max(48, 8 * ndim)
    rng = np.random.default_rng(seed)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.lnprob,
                                    args=(wave, flux, flux_err, rvs, rv_errs, snrs))
    pos = initial_positions(model, p0, nwalkers, scale=scale, rng=rng)

    t0 = time.time()
    state = sampler.run_mcmc(pos, burnin, progress=progress)
    sampler.reset()
    sampler.run_mcmc(state, production, progress=progress)
    runtime = time.time() - t0

    samples = sampler.get_chain(flat=True)
    p16, pfit, p84 = np.percentile(samples, [16, 50, 84], axis=0)
    perr = (p84 - p16) / 2.0
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:  # chain too short for a reliable estimate
        tau = np.full(ndim, np.nan)

    lnL = chi2_lnL(model, pfit, wave, flux, flux_err, rvs)
    # information criteria are evaluated at the best-fitting sample (a joint
    # estimate); the marginal medians can be a poor combination when several
    # parameters are degenerate (see selection.best_sample_lnL)
    lnL_best, p_best = best_sample_lnL(model, samples, wave, flux, flux_err, rvs)
    k, n = ndim, flux.size
    return MCMCResult(param_names=model.param_names, pfit=pfit, perr=perr, p16=p16, p84=p84,
                      samples=samples, tau=np.asarray(tau), lnL=lnL, aic=aic(lnL_best, k),
                      bic=bic(lnL_best, k, n), label=classify_tau(tau), nwalkers=nwalkers,
                      burnin=burnin, production=production, runtime=runtime,
                      extra=dict(lnL_best=float(lnL_best), p_best=np.asarray(p_best)))
