#!/usr/bin/env python
"""Run MCMC-RVSD on the three observed targets shipped with the examples.

The three targets illustrate the three quality regimes of the paper:

===========  =================  ===========================
label        uid                autocorrelation time (steps)
===========  =================  ===========================
Success      G13619209590707    ~59
Mild         G16857550743084    ~121
Severe       G16954716324740    ~375
===========  =================  ===========================

For every target the script reads ``data/real/<uid>/preprocessed.fits``, fits the
fiducial six-parameter model (one Fe I line + one Gaussian DIB) with the SNR
weighting used in the paper, and writes ``data/real/<uid>/mcmc_result.npz`` plus a
JSON summary.  Decorrelate the chains with ``--thin`` (default 10) if the files are
too large.

Run::

    python scripts/run_real_targets.py [--uids ...] [--production 5000]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from mcmc_rvsd import DIB, SpectralModel, StellarLine, run_mcmc
from mcmc_rvsd.preprocess import read_preprocessed

TARGETS = {
    'G13619209590707': 'Success',
    'G16857550743084': 'Mild Degeneracy',
    'G16954716324740': 'Severe Degeneracy',
}
P0 = dict(lambda_star_0=6613.6, A_star=0.010, sigma_star=0.5,
          lambda_DIB=6613.7303, A_DIB=0.0455, sigma_DIB=0.5367)


def run_target(uid, datadir='data/real', burnin=1500, production=5000, thin=10, seed=42):
    path = os.path.join(datadir, uid, 'preprocessed.fits')
    data = read_preprocessed(path)
    flux, flux_err = np.atleast_2d(data['flux']), np.atleast_2d(data['flux_err'])

    model = SpectralModel(stellar_lines=[StellarLine(name='FeI', center=6613.8225)],
                          dib=DIB(profile='gaussian'))
    p0 = np.array([P0['lambda_star_0'], P0['A_star'], P0['sigma_star'],
                   P0['lambda_DIB'], P0['A_DIB'], P0['sigma_DIB']])
    t0 = time.time()
    result = run_mcmc(model, data['wave'], flux, flux_err, data['rvs'], data['rv_errs'],
                      p0=p0, snrs=data['snrs'], burnin=burnin, production=production, seed=seed)
    out_npz = os.path.join(datadir, uid, 'mcmc_result.npz')
    result.save(out_npz, thin=thin)
    with open(os.path.join(datadir, uid, 'mcmc_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(uid=uid, expected_label=TARGETS.get(uid), n_epochs=int(len(data['rvs'])),
                       rv_span=float(np.ptp(data['rvs'])), snr_median=float(np.median(data['snrs'])),
                       runtime_s=round(time.time() - t0, 1), **result.as_dict()), fh, indent=2)
    print(f'[{uid}] {result.label} | tau = {np.round(result.tau, 1)} | '
          f'{time.time() - t0:.0f} s -> {out_npz}')
    return result


def main(uids=None, **kw):
    for uid in (uids or list(TARGETS)):
        run_target(uid, **kw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uids', nargs='*', default=None, help='subset of target uids')
    parser.add_argument('--datadir', default='data/real')
    parser.add_argument('--burnin', type=int, default=1500)
    parser.add_argument('--production', type=int, default=5000)
    parser.add_argument('--thin', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    main(**vars(parser.parse_args()))
