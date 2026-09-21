"""Smoke tests: the forward model, the priors, one short MCMC run and the I/O.

Run with ``pytest -q`` from the repository root (after ``pip install -e .``), or::

    PYTHONPATH=src pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mcmc_rvsd import (DIB, SpectralModel, StellarLine, aic, bic, classify,  # noqa: E402
                       dib_autocorr_times, gaussian_absorption, n_effective, run_mcmc,
                       voigt_absorption)
from mcmc_rvsd.io import load_mock, load_result                                   # noqa: E402

WAVE = np.arange(6611.5, 6617.0, 0.12)


def make_data(n_epochs=6, n_stars=1, voigt=False, seed=0):
    rng = np.random.default_rng(seed)
    rvs = np.linspace(-50, 50, n_epochs)
    rv_err = np.full(n_epochs, 1.0)
    stars = [StellarLine('FeI', 6613.8225)] if n_stars == 1 else [
        StellarLine('FeI', 6613.8225), StellarLine('ThI', 6613.416), StellarLine('YII', 6613.731)]
    dib = DIB(profile='voigt' if voigt else 'gaussian')
    model = SpectralModel(stellar_lines=stars, dib=dib)
    if n_stars == 1:
        p_stars = [6613.8225, 0.011, 0.38]
    else:
        p_stars = [6613.8225, 0.011, 0.38, 6613.416, 0.0111, 0.416, 6613.731, 0.0066, 0.378]
    p_true = np.array(p_stars + [6613.66, 0.049, 0.545] + ([0.10] if voigt else []))
    flux = np.array([model.flux(WAVE, rv, p_true) for rv in rvs])
    flux = flux + rng.normal(0, 1 / 200.0, flux.shape)
    return model, p_true, WAVE, flux, np.full_like(flux, 1 / 200.0), rvs, rv_err


def test_profiles_peak_at_center():
    x = np.linspace(6600, 6630, 4000)
    assert gaussian_absorption(np.array([6613.0]), 6613.0, 0.4, 0.5)[0] == pytest.approx(0.6)
    assert voigt_absorption(np.array([6613.0]), 6613.0, 0.4, 0.5, 0.2)[0] == pytest.approx(0.6, rel=1e-9)
    # far from the line both profiles are normalised to the continuum
    assert gaussian_absorption(np.array([6613.0]), 6650.0, 0.4, 0.5)[0] == pytest.approx(1.0)
    assert voigt_absorption(np.array([6613.0]), 6650.0, 0.4, 0.5, 0.2)[0] > 0.999


@pytest.mark.parametrize('n_stars,voigt,n_expected', [(1, False, 6), (1, True, 7), (3, False, 12)])
def test_parameter_bookkeeping(n_stars, voigt, n_expected):
    model, p_true, *_ = make_data(n_stars=n_stars, voigt=voigt)
    assert model.n_params == n_expected == p_true.size
    assert len(model.param_names) == n_expected
    assert len(model.bounds) == n_expected
    p0 = model.initial_guess()
    assert np.isfinite(model.lnprior(p0))


def test_prior_bounds_are_enforced():
    model, p_true, *_ = make_data()
    assert model.lnprior(p_true) == 0.0
    outside = p_true.copy()
    outside[0] = 6500.0
    assert model.lnprior(outside) == -np.inf
    outside = p_true.copy()
    outside[-1] = 5.0
    assert model.lnprior(outside) == -np.inf


def test_likelihood_and_rv_propagation():
    model, p_true, wave, flux, err, rvs, rv_err = make_data()
    good = model.lnlike(p_true, wave, flux, err, rvs, rv_err)
    bad = model.lnlike(p_true * 1.05, wave, flux, err, rvs, rv_err)
    assert np.isfinite(good) and bad < good
    # a larger RV error adds variance, so the same parameters become slightly less likely
    assert model.lnlike(p_true, wave, flux, err, rvs, rv_err * 10) < good
    # zero errors make the likelihood invalid
    assert model.lnlike(p_true, wave, flux, np.zeros_like(err), rvs, rv_err) == -np.inf


def test_short_mcmc_recovers_the_mock():
    model, p_true, wave, flux, err, rvs, rv_err = make_data()
    res = run_mcmc(model, wave, flux, err, rvs, rv_err, p0=model.initial_guess(),
                   snrs=np.full(len(rvs), 200.0), nwalkers=24, burnin=150, production=300, seed=1)
    # 5-sigma tolerance with a deliberately tiny chain
    assert np.all(np.abs(res.pfit - p_true) / res.perr < 5.0)
    assert res.pfit[-3] == pytest.approx(p_true[-3], abs=0.05)     # DIB depth
    assert res.label in ('Success', 'Mild Degeneracy', 'Severe Degeneracy')


def test_selection_and_quality_helpers():
    assert aic(-100.0, 6) == 12 + 200
    assert bic(-100.0, 6, 1000) == pytest.approx(6 * np.log(1000) + 200)

    # the label rule of the paper is on L / tau_DIB: max(L/tau) >= 50 -> Success,
    # min(L/tau) <= 25 -> Severe, otherwise Mild
    tau_ok = [60., 61., 62., 62., 59., 68.]          # min(tau_DIB) = 59  -> 85 samples
    tau_severe = [300., 206., 237., 221., 236., 135.]  # max(tau_DIB) = 236 -> 21 samples
    tau_mild = [170., 160., 150., 156., 160., 156.]
    assert classify(tau_ok, 5000) == 'Success'
    assert classify(tau_mild, 5000) == 'Mild Degeneracy'
    assert classify(tau_severe, 5000) == 'Severe Degeneracy'

    # asymmetry: Success uses the *smallest* DIB tau, Severe the *largest*
    assert classify([50., 50., 50., 30., 400., 40.], 5000) == 'Success'   # min -> 125 samples
    assert classify([50., 50., 50., 150., 150., 150.], 5000) == 'Mild Degeneracy'

    # the thresholds scale with the chain length
    assert classify(tau_mild, 5000) == 'Mild Degeneracy'
    assert classify(tau_mild, 20000) == 'Success'          # 20000/156 = 128 samples
    assert classify(tau_ok, 1000) == 'Severe Degeneracy'   # 1000/62 = 16 samples <= 25
    assert n_effective([100., 100., 100.], 5000)[0] == pytest.approx(50.0)
    assert list(dib_autocorr_times(np.arange(7) + 1.)) == [5.0, 6.0, 7.0]


def test_result_roundtrip(tmp_path):
    model, p_true, wave, flux, err, rvs, rv_err = make_data()
    res = run_mcmc(model, wave, flux, err, rvs, rv_err, p0=model.initial_guess(),
                   snrs=np.full(len(rvs), 200.0), nwalkers=16, burnin=100, production=150, seed=2)
    path = res.save(tmp_path / 'result.npz', thin=3)
    back = load_result(path)
    assert np.allclose(back.pfit, res.pfit)
    assert back.param_names == model.param_names
    assert back.label == res.label
    assert back.samples.shape[1] == model.n_params


def test_shipped_data_files_exist():
    for rel in ('data/mock/mock_6614_gaussian.npz', 'data/mock/mock_6614_voigt.npz',
                'data/mock/mock_multistar.npz', 'data/mock/dib_scan_results.json',
                'examples/results/mock_6614_gaussian.npz'):
        path = ROOT / rel
        if not path.exists():
            pytest.skip(f'{rel} not shipped yet')
        assert path.stat().st_size > 0


def test_mock_file_loads():
    path = ROOT / 'data/mock/mock_6614_gaussian.npz'
    if not path.exists():
        pytest.skip('mock file not generated yet')
    d = load_mock(path)
    assert {'wave_analysis', 'fluxs', 'y_errs', 'rv_obs', 'rv_err'} <= set(d)
    assert d['fluxs'].shape[1] == d['wave_analysis'].size
