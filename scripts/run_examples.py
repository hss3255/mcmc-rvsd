#!/usr/bin/env python
"""Precompute the MCMC results that the notebooks display by default.

The notebooks read ``examples/results/*.npz`` so that they open instantly; set
``RERUN = True`` inside a notebook to redo the fit yourself.  This script (re)builds
all of them with the paper's settings (48 walkers, 1500 burn-in, 5000 production
steps, SNR weighting ``variance_scaling``).

Run::

    python scripts/run_examples.py [--only mock_6614_gaussian ...]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from mcmc_rvsd import DIB, SpectralModel, StellarLine, run_mcmc
from mcmc_rvsd.io import save_summary_json
from mcmc_rvsd.profiles import C_KMS, gaussian_absorption
from mcmc_rvsd.selection import chi2_lnL

MOCK = 'data/mock'
OUT = 'examples/results'

# ----------------------------------------------------------------- jobs
def _load(name):
    return dict(np.load(os.path.join(MOCK, name), allow_pickle=True))


def job_mock_6614_gaussian():
    d = _load('mock_6614_gaussian.npz')
    model = SpectralModel(stellar_lines=[StellarLine('FeI', 6613.8225)],
                          dib=DIB(profile='gaussian'))
    p0 = model.initial_guess(dib=[6613.66, 0.05, 0.55])
    truth = [d['stellar_truth'].item()['center'], d['stellar_truth'].item()['strength'],
             d['stellar_truth'].item()['sigma'], d['dib_truth'].item()['center'],
             d['dib_truth'].item()['strength'], d['dib_truth'].item()['sigma']]
    return model, d, p0, dict(truth=np.array(truth))


def job_mock_6614_voigt_gauss():
    d = _load('mock_6614_voigt.npz')
    model = SpectralModel(stellar_lines=[StellarLine('FeI', 6613.8225)],
                          dib=DIB(profile='gaussian'))
    p0 = model.initial_guess(dib=[6613.66, 0.05, 0.55])
    t = d['dib_truth'].item()
    return model, d, p0, dict(truth=np.array([d['stellar_truth'].item()['center'],
                                              d['stellar_truth'].item()['strength'],
                                              d['stellar_truth'].item()['sigma'],
                                              t['center'], t['strength'], t['sigma']]))


def job_mock_6614_voigt_voigt():
    d = _load('mock_6614_voigt.npz')
    model = SpectralModel(stellar_lines=[StellarLine('FeI', 6613.8225)],
                          dib=DIB(profile='voigt', gamma0=0.05, gamma_bounds=(0.001, 0.3)))
    p0 = model.initial_guess(dib=[6613.66, 0.05, 0.55, 0.05])
    t = d['dib_truth'].item()
    return model, d, p0, dict(truth=np.array([d['stellar_truth'].item()['center'],
                                              d['stellar_truth'].item()['strength'],
                                              d['stellar_truth'].item()['sigma'],
                                              t['center'], t['strength'], t['sigma'], t['gamma']]))


def _voigt06(kind):
    d = _load('mock_6614_voigt_gamma06.npz')
    t = d['dib_truth'].item()
    truth = [d['stellar_truth'].item()['center'], d['stellar_truth'].item()['strength'],
             d['stellar_truth'].item()['sigma'], t['center'], t['strength'], t['sigma']]
    if kind == 'voigt':
        model = SpectralModel(stellar_lines=[StellarLine('FeI', 6613.8225)],
                              dib=DIB(profile='voigt', gamma0=0.05, gamma_bounds=(0.001, 1.0)))
        p0 = model.initial_guess(dib=[6613.66, 0.05, 0.55, 0.05])
        truth = truth + [t['gamma']]
    else:                                    # gamma = 0.60 needs a wider prior bound
        model = SpectralModel(stellar_lines=[StellarLine('FeI', 6613.8225)],
                              dib=DIB(profile='gaussian', sigma_bounds=(0.3, 1.5)))
        p0 = model.initial_guess(dib=[6613.66, 0.05, 0.6])
    return model, d, p0, dict(truth=np.array(truth))


def job_voigt06_gauss():
    return _voigt06('gaussian')


def job_voigt06_voigt():
    return _voigt06('voigt')


def _multistar_model(n):
    lines = [StellarLine('FeI', 6613.825, center_bounds=(6611.5, 6617.0)),
             StellarLine('ThI', 6613.416, center_bounds=(6611.5, 6617.0)),
             StellarLine('YII', 6613.731, center_bounds=(6611.5, 6617.0))]
    return SpectralModel(stellar_lines=lines[:n], dib=DIB(profile='gaussian'))


def _multistar_job(n):
    d = _load('mock_multistar.npz')
    model = _multistar_model(n)
    stars = [[s['center'], s['strength'], s['sigma']] for s in d['stellar_lines']][:n]
    p0 = model.initial_guess(dib=[6613.709, 0.05, 0.546], stars=stars)
    dib = d['dib_truth'].item()
    truth = {}
    for i, s in enumerate(d['stellar_lines']):
        truth[f'lambda_{s["name"]}'] = s['center']
        truth[f'A_{s["name"]}'] = s['strength']
        truth[f'sigma_{s["name"]}'] = s['sigma']
    truth.update(lambda_DIB=dib['center'], A_DIB=dib['strength'], sigma_DIB=dib['sigma'])
    return model, d, p0, dict(truth=truth)


def _third_target_mock():
    """A lambda 6379 mock: the DIB is narrower and weaker than lambda 6614."""
    d = _load('mock_6614_gaussian.npz')
    wave = np.arange(6350.0, 6410.0, 0.12)
    rv, rv_err = d['rv'], d['rv_err']
    rng = np.random.RandomState(42)
    flux = np.zeros((len(rv), len(wave)))
    for i, v in enumerate(rv):
        star = gaussian_absorption(wave, 6378.68 * (1 + v / C_KMS), 0.0044, 0.54)
        dib = gaussian_absorption(wave, 6379.60, 0.0307, 0.384)
        flux[i] = star * dib + rng.normal(0, 1.0 / 200.0, wave.size)
    return dict(wave_analysis=wave, fluxs=flux, y_errs=np.full_like(flux, 1.0 / 200.0),
                rv=rv, rv_obs=d['rv_obs'], rv_err=rv_err)


def _unblended_job(n):
    d = _load('mock_multistar_unblended.npz')
    lines = [StellarLine('FeI', 6613.825, center_bounds=(6611.5, 6617.0)),
             StellarLine('ThI', 6612.325, center_bounds=(6611.5, 6617.0)),
             StellarLine('YII', 6615.325, center_bounds=(6611.5, 6617.0))]
    model = SpectralModel(stellar_lines=lines[:n], dib=DIB(profile='gaussian'))
    stars = [[s['center'], s['strength'], s['sigma']] for s in d['stellar_lines']][:n]
    p0 = model.initial_guess(dib=[6613.709, 0.05, 0.546], stars=stars)
    truth = {}
    for s in d['stellar_lines']:
        truth.update({f'lambda_{s["name"]}': s['center'], f'A_{s["name"]}': s['strength'],
                      f'sigma_{s["name"]}': s['sigma']})
    dib = d['dib_truth'].item()
    truth.update(lambda_DIB=dib['center'], A_DIB=dib['strength'], sigma_DIB=dib['sigma'])
    return model, d, p0, dict(truth=truth)


def job_unblended_1line():
    return _unblended_job(1)


def job_unblended_3line():
    return _unblended_job(3)


def job_other_dib_6379():
    d = _third_target_mock()
    np.savez(os.path.join(MOCK, 'mock_6379.npz'), **d)
    model = SpectralModel(stellar_lines=[StellarLine('star', 6378.68, center_bounds=(6375.0, 6383.0))],
                          dib=DIB(center=6379.60, center_bounds=(6378.5, 6381.0),
                                  sigma_bounds=(0.2, 1.0), profile='gaussian'))
    p0 = model.initial_guess(stars=[[6378.68, 0.0044, 0.54]], dib=[6379.60, 0.031, 0.384])
    truth = np.array([6378.68, 0.0044, 0.54, 6379.60, 0.0307, 0.384])
    return model, d, p0, dict(truth=truth)


JOBS = {
    'mock_6614_gaussian': job_mock_6614_gaussian,
    'mock_6614_voigt_gauss': job_mock_6614_voigt_gauss,
    'mock_6614_voigt_voigt': job_mock_6614_voigt_voigt,
    'mock_6614_voigt06_gauss': job_voigt06_gauss,
    'mock_6614_voigt06_voigt': job_voigt06_voigt,
    'mock_multistar_1line': lambda: _multistar_job(1),
    'mock_multistar_3line': lambda: _multistar_job(3),
    'mock_unblended_1line': job_unblended_1line,
    'mock_unblended_3line': job_unblended_3line,
    'other_dib_6379': job_other_dib_6379,
}


def run(name, burnin=1500, production=5000, thin=10, seed=42):
    model, data, p0, extra = JOBS[name]()
    t0 = time.time()
    result = run_mcmc(model, data['wave_analysis'], data['fluxs'], data['y_errs'],
                      data['rv_obs'], data['rv_err'], p0=p0, snrs=np.full(len(data['rv_obs']), 200.0),
                      burnin=burnin, production=production, seed=seed)
    path = os.path.join(OUT, f'{name}.npz')
    result.save(path, thin=thin)
    save_summary_json(result, os.path.join(OUT, f'{name}.json'), runtime_s=round(time.time() - t0, 1), **extra)
    print(f'[{name}] {result.label} | tau = {np.round(result.tau, 1)} | {time.time() - t0:.0f} s -> {path}')


def main(only=None, **kw):
    os.makedirs(OUT, exist_ok=True)
    for name in (only or list(JOBS)):
        run(name, **kw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--burnin', type=int, default=1500)
    parser.add_argument('--production', type=int, default=5000)
    parser.add_argument('--thin', type=int, default=10)
    main(**vars(parser.parse_args()))
