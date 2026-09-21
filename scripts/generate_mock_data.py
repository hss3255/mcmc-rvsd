#!/usr/bin/env python
"""Generate the mock datasets used by the notebooks (deterministic, seed = 42).

Three datasets, all on the same 24-epoch radial-velocity sequence (SNR = 200,
0.12 A sampling, the setup of the paper):

* ``mock_6614_gaussian.npz`` - fiducial case: one Fe I line + a Gaussian DIB;
* ``mock_6614_voigt.npz``    - the DIB is a Voigt profile (sigma = 0.545 A,
  gamma = 0.10 A), the stellar line stays Gaussian;
* ``mock_multistar.npz``     - three stellar lines (Fe I, Y II, Th I) + one DIB.

Run::

    python scripts/generate_mock_data.py [--outdir data/mock]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from mcmc_rvsd.profiles import C_KMS, gaussian_absorption, voigt_absorption

# ------------------------------------------------------------------ setup
SNR = 200.0
RES = 0.12                       # Angstrom per pixel
WAVE = np.arange(6611.5, 6617.0, RES)

RV = np.array([26.07, -47.83, 0.06, 55.85, -40.82, -16.70, 50.71, -47.93,
               46.08, -40.64, -55.66, -25.89, 6.57, -22.96, -65.23, -65.05,
               -26.18, -65.63, -35.91, -58.84, 53.53, 58.51, 35.59, -22.80])
RV_ERR = np.array([1.27, 1.27, 1.27, 1.27, 1.26, 0.98, 0.98, 0.98, 0.98,
                   0.98, 0.98, 0.98, 0.87, 1.31, 0.87, 1.30, 1.15, 1.26,
                   1.27, 1.16, 1.27, 1.27, 1.26, 1.27])
RV_OBS = RV + np.random.RandomState(42).normal(0, RV_ERR)

# fiducial stellar line (Fe I) and DIB (lambda 6614)
STAR_FEI = {'center': 6613.825, 'sigma': 0.380, 'strength': 0.0110}
DIB_GAUSS = {'center': 6613.660, 'sigma': 0.545, 'strength': 0.049}
DIB_VOIGT = {'center': 6613.660, 'sigma': 0.545, 'gamma': 0.100, 'strength': 0.049}
# the stress case of the paper: a much stronger Lorentzian component (gamma = 0.60 A)
DIB_VOIGT06 = {'center': 6613.660, 'sigma': 0.545, 'gamma': 0.600, 'strength': 0.049}
# the three-line case (paper section on multiple stellar lines)
STARS_MULTI = [{'name': 'FeI', 'center': 6613.825, 'sigma': 0.380, 'strength': 0.0110},
               {'name': 'ThI', 'center': 6613.416, 'sigma': 0.416, 'strength': 0.0111},
               {'name': 'YII', 'center': 6613.731, 'sigma': 0.378, 'strength': 0.0066}]
DIB_MULTI = {'center': 6613.709, 'sigma': 0.546, 'strength': 0.049}
# configuration (b) of the paper: the same three lines, but spread over the window
# (6612.325 / 6613.825 / 6615.325 A) so that they are no longer blended
STARS_UNBLENDED = [{'name': 'FeI', 'center': 6613.825, 'sigma': 0.380, 'strength': 0.0110},
                   {'name': 'ThI', 'center': 6612.325, 'sigma': 0.416, 'strength': 0.0111},
                   {'name': 'YII', 'center': 6615.325, 'sigma': 0.378, 'strength': 0.0066}]


def _noisy(flux_clean, rng):
    return flux_clean + rng.normal(0, 1.0 / SNR, flux_clean.shape)


def build_gaussian():
    """One Gaussian stellar line + one Gaussian DIB."""
    rng = np.random.RandomState(42)
    flux = np.zeros((len(RV), len(WAVE)))
    for i, rv in enumerate(RV):
        star = gaussian_absorption(WAVE, STAR_FEI['center'] * (1 + rv / C_KMS),
                                   STAR_FEI['strength'], STAR_FEI['sigma'])
        dib = gaussian_absorption(WAVE, DIB_GAUSS['center'], DIB_GAUSS['strength'], DIB_GAUSS['sigma'])
        flux[i] = _noisy(star * dib, rng)
    return dict(wave_analysis=WAVE, fluxs=flux, y_errs=np.full_like(flux, 1.0 / SNR),
                rv=RV, rv_obs=RV_OBS, rv_err=RV_ERR,
                stellar_truth=STAR_FEI, dib_truth=DIB_GAUSS)


def build_voigt():
    """One Gaussian stellar line + one Voigt DIB (intrinsic Lorentzian component)."""
    rng = np.random.RandomState(42)
    flux = np.zeros((len(RV), len(WAVE)))
    for i, rv in enumerate(RV):
        star = gaussian_absorption(WAVE, STAR_FEI['center'] * (1 + rv / C_KMS),
                                   STAR_FEI['strength'], STAR_FEI['sigma'])
        dib = voigt_absorption(WAVE, DIB_VOIGT['center'], DIB_VOIGT['strength'],
                               DIB_VOIGT['sigma'], DIB_VOIGT['gamma'])
        flux[i] = _noisy(star * dib, rng)
    return dict(wave_analysis=WAVE, fluxs=flux, y_errs=np.full_like(flux, 1.0 / SNR),
                rv=RV, rv_obs=RV_OBS, rv_err=RV_ERR,
                stellar_truth=STAR_FEI, dib_truth=DIB_VOIGT, multistar_truth=STARS_MULTI)


def build_voigt06():
    """As build_voigt, but with a strong Lorentzian component (gamma = 0.60 A)."""
    rng = np.random.RandomState(42)
    flux = np.zeros((len(RV), len(WAVE)))
    for i, rv in enumerate(RV):
        star = gaussian_absorption(WAVE, STAR_FEI['center'] * (1 + rv / C_KMS),
                                   STAR_FEI['strength'], STAR_FEI['sigma'])
        dib = voigt_absorption(WAVE, DIB_VOIGT06['center'], DIB_VOIGT06['strength'],
                               DIB_VOIGT06['sigma'], DIB_VOIGT06['gamma'])
        flux[i] = _noisy(star * dib, rng)
    return dict(wave_analysis=WAVE, fluxs=flux, y_errs=np.full_like(flux, 1.0 / SNR),
                rv=RV, rv_obs=RV_OBS, rv_err=RV_ERR,
                stellar_truth=STAR_FEI, dib_truth=DIB_VOIGT06)


def build_multistar():
    """Three stellar lines (Fe I, Th I, Y II) + one Gaussian DIB."""
    rng = np.random.RandomState(42)
    flux = np.zeros((len(RV), len(WAVE)))
    for i, rv in enumerate(RV):
        m = np.ones_like(WAVE)
        for s in STARS_MULTI:
            m = m * gaussian_absorption(WAVE, s['center'] * (1 + rv / C_KMS),
                                        s['strength'], s['sigma'])
        m = m * gaussian_absorption(WAVE, DIB_MULTI['center'], DIB_MULTI['strength'], DIB_MULTI['sigma'])
        flux[i] = _noisy(m, rng)
    return dict(wave_analysis=WAVE, fluxs=flux, y_errs=np.full_like(flux, 1.0 / SNR),
                rv=RV, rv_obs=RV_OBS, rv_err=RV_ERR,
                stellar_lines=np.array(STARS_MULTI, dtype=object), dib_truth=DIB_MULTI)


def build_multistar_unblended():
    """The same three lines as build_multistar, but unblended (configuration b)."""
    rng = np.random.RandomState(42)
    flux = np.zeros((len(RV), len(WAVE)))
    for i, rv in enumerate(RV):
        m = np.ones_like(WAVE)
        for s in STARS_UNBLENDED:
            m = m * gaussian_absorption(WAVE, s['center'] * (1 + rv / C_KMS),
                                        s['strength'], s['sigma'])
        m = m * gaussian_absorption(WAVE, DIB_MULTI['center'], DIB_MULTI['strength'], DIB_MULTI['sigma'])
        flux[i] = _noisy(m, rng)
    return dict(wave_analysis=WAVE, fluxs=flux, y_errs=np.full_like(flux, 1.0 / SNR),
                rv=RV, rv_obs=RV_OBS, rv_err=RV_ERR,
                stellar_lines=np.array(STARS_UNBLENDED, dtype=object), dib_truth=DIB_MULTI)


def main(outdir='data/mock'):
    os.makedirs(outdir, exist_ok=True)
    for name, builder in (('mock_6614_gaussian', build_gaussian),
                          ('mock_6614_voigt', build_voigt),
                          ('mock_6614_voigt_gamma06', build_voigt06),
                          ('mock_multistar', build_multistar),
                          ('mock_multistar_unblended', build_multistar_unblended)):
        data = builder()
        path = os.path.join(outdir, f'{name}.npz')
        np.savez(path, **data)
        print(f'wrote {path} | fluxs {data["fluxs"].shape} | RV span '
              f'{data["rv_obs"].min():.1f} to {data["rv_obs"].max():.1f} km/s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='data/mock')
    main(**vars(parser.parse_args()))
