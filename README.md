# MCMC-RVSD

**MCMC-RVSD** measures diffuse interstellar bands (DIBs) in multi-epoch spectra of
single-lined binary stars. The stellar lines move with the orbital radial velocity (RV)
while the DIB stays at its rest wavelength, so fitting all epochs simultaneously with an
ensemble MCMC sampler separates the two components and returns the DIB parameters
(centre, depth, width, and - for a non-Gaussian profile - the Lorentzian component)
together with a quality label for every measurement.

The package accompanies *"Disentangling Multi-epoch Single-lined Binary Spectra to Reveal
Interstellar Medium Features via MCMC"* (submitted; the citation will be updated here once the
paper is published).

## Installation

```bash
# from the repository (recommended while the examples are being developed)
pip install -e .
# or with conda
conda env create -f environment.yml && conda activate mcmc-rvsd
```

`pip install` only ships the library in `src/`. The notebooks (`examples/`, ~10 kB each) and the
spectra (`data/`: one raw LAMOST file and one preprocessed file per target, ~0.5 MB and ~0.03-0.16 MB)
are **not** part of the distribution - browse them on GitHub and download only what you need.

Preprocessing a raw LAMOST MRS spectrum takes one call:

```python
from mcmc_rvsd import preprocess
single = preprocess.preprocess_spectrum('data/real/G13619209590707/raw/G13619209590707_934314249.fits')
# -> dict with wave, flux_norm, flux_norm_err, continuum, rv, rv_err, snr (on the 6604-6664 A grid)
```

## Quick start

```python
import numpy as np
from mcmc_rvsd import SpectralModel, StellarLine, DIB, run_mcmc

# one stellar line + a Gaussian DIB (the fiducial model of the paper)
model = SpectralModel(stellar_lines=[StellarLine(name='FeI', center=6613.8225)],
                      dib=DIB(profile='gaussian'))

p0 = model.initial_guess(dib=[6613.66, 0.05, 0.55])
result = run_mcmc(model, wave, flux, flux_err, rvs, rv_errs, p0=p0, snrs=snrs)

print(result.summary())        # parameters, errors, lnL, AIC/BIC, tau, quality label
result.save('target_result.npz', thin=10)
```

## The four models used in the paper

| Case | How to configure |
|------|------------------|
| Fiducial DIB lambda6614 | `SpectralModel([StellarLine('FeI', 6613.8225)], DIB(profile='gaussian'))` |
| Multiple stellar lines | `SpectralModel([StellarLine('FeI', 6613.8225), StellarLine('YII', 6613.7310), StellarLine('ThI', 6613.4160)], DIB())` |
| Non-Gaussian DIB profile | `DIB(profile='voigt', gamma0=0.05)` - adds the Lorentzian half-width `gamma_DIB` (7 parameters instead of 6) |
| Other DIBs (6379, 6660, ...) | change `DIB(center=..., center_bounds=..., sigma_bounds=...)` and the fitting window |

Every parameter is controlled by the user: the initial values (`depth0`, `sigma0`,
`gamma0`), the flat prior bounds (`center_bounds`, `depth_bounds`, `sigma_bounds`,
`gamma_bounds`) and the number of stellar lines. `model.param_names` and
`model.bounds` list them in the order used by the sampler.

### Combining the epochs

* `rv_error_propagation=True` (default) adds the flux uncertainty induced by the RV
  error, `|d(model)/d(lambda) * lambda * sigma_RV / c|`, to the measurement error;
* `weight_method='variance_scaling'` (default) combines the epochs with SNR weights
  normalised to mean 1, `'likelihood_weight'` weights the log-likelihood directly and
  `'none'` leaves the epochs unweighted.

### Model selection and quality labels

Adding a stellar line always improves the fit, so the number of components is chosen
with AIC/BIC::

```python
from mcmc_rvsd import aic, bic
```

The chains are labelled from the **effective sample size of the DIB parameters**,
`L / tau_DIB`, where `L` is the chain length (production steps) and `tau_DIB` collects the
autocorrelation times of the three DIB parameters:

| condition | label |
|---|---|
| `max(L/tau_DIB) >= 50` | `Success` |
| `min(L/tau_DIB) <= 25` | `Severe Degeneracy` |
| otherwise | `Mild Degeneracy` |

`classify(tau, n_steps)` in `mcmc_rvsd.quality` implements exactly this rule. Both
conditions act on the three DIB parameters jointly: `max` for Success (one well-sampled
parameter is enough to accept the source) and `min` for Severe (one badly sampled
parameter is enough to reject it). For the catalogue chain length `L = 5000` the rule is
equivalent to `min(tau_DIB) <= 100` -> Success and `max(tau_DIB) >= 200` -> Severe, but
the thresholds move with `L` - pass the chain length that produced the samples.

A large autocorrelation time means that the parameters are degenerate, which is the final
safeguard for a measurement that cannot be modelled explicitly.

## Repository layout

```
src/mcmc_rvsd/          the library (packaged, `pip install` ships this)
  profiles.py           Gaussian / Voigt line profiles
  models.py             SpectralModel, StellarLine, DIB (user configuration + priors)
  likelihood.py         RV-error propagation and SNR weighting
  priors.py             box / RV-constrained / Gaussian priors
  sampler.py            emcee driver, MCMCResult, save/load
  selection.py          AIC / BIC
  quality.py            autocorrelation-time labels
  preprocess.py         LAMOST MRS preprocessing (read, cut, resample, normalise)
  plotting.py           spectrum + fit, per-epoch panels, corner plots
  specnorm.py           continuum normalisation (adapted from the LAMOST pipeline)
examples/               5 tutorial notebooks (not in the pip package)
data/                   mock data and the three real targets (not in the pip package)
scripts/                data generation and batch runners
tests/                  smoke tests
```

## Examples

| Notebook | Content |
|---|---|
| `examples/01_mock_6614_tutorial.ipynb` | mock spectrum, fiducial 6-parameter Gaussian model, posterior and corner plot |
| `examples/02_multi_stellar_lines.ipynb` | 2-3 stellar lines, stepwise model building with AIC/BIC |
| `examples/03_non_gaussian_dib.ipynb` | Gaussian vs Voigt, why the Gaussian width absorbs the wings |
| `examples/04_other_dibs.ipynb` | lambda6379 / lambda6660: a width-depth scan and its limits |
| `examples/05_real_targets.ipynb` | the three observed targets (Success / Mild / Severe) end to end, including the preprocessing of a raw LAMOST file |

Each notebook reads precomputed results by default (fast) and exposes a `RERUN = True`
switch to redo the MCMC.

## Validation in brief

* the fiducial model recovers all parameters of the mock data;
* a Gaussian fit to a DIB with a Lorentzian component overestimates `sigma_DIB` by
  `0.40 A` (`+44 sigma`) while the Voigt fit recovers it to within `0.9 sigma` - the
  equivalent width is barely affected (`-2.75%`);
* across a 50-point width-depth scan (`sigma_DIB = 0.4-0.8 A`, `A_DIB = 0.01-0.10`) all
  parameter deviations stay below `3 sigma`;
* a Gaussian fit to mock spectra with multiple stellar lines is biased, while adding the
  contaminating lines removes the bias (selected with AIC/BIC).

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # profiles, priors, likelihood, one short MCMC run, I/O round trip
```

## Reproducing the shipped results

```bash
python scripts/generate_mock_data.py     # data/mock/*.npz            (seconds)
python scripts/run_examples.py           # examples/results/*.npz     (~3 min per case)
python scripts/run_real_targets.py       # data/real/*/mcmc_result.npz (~1-2 min per target)
python scripts/compress_figures.py       # optional: shrink docs/figures before committing
```

`INDEX.md` lists every file of the repository and maps it to the research script it was
refactored from.

## Citation

If you use this code, please cite the accompanying paper (see the top of this file) and the
LAMOST DR12 data release. The full citation will be added here once the paper is published.

## Licence

MIT - see `LICENSE`.
