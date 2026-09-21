# File index

A lookup table for this package and for the research files it was refactored from.
Handy when you need "where was that thing again?".

## 1. Library (`src/mcmc_rvsd/`, shipped by `pip install`)

| File | Content | Original research file |
|---|---|---|
| `profiles.py` | Gaussian / Voigt line profiles, Doppler shift, `C_KMS` | `mock_data/mcmc.py` (`model`), `ai/Q03_非高斯轮廓/run_voigt_fit.py` |
| `models.py` | `SpectralModel`, `StellarLine`, `DIB`: parameter vector, bounds, forward model, priors | `mock_data/mcmc.py`, `ai/Q2_多恒星线实测/multi_star_mcmc.py` |
| `likelihood.py` | Gaussian log-likelihood, RV-error propagation, SNR weighting (`variance_scaling` / `likelihood_weight` / `none`) | `ai/Q06_真实数据分解展示/mcmc_q6.py` |
| `priors.py` | Box prior, RV-constrained centre prior, Gaussian prior | `mock_data/mcmc.py` (`lnprior`, commented block) |
| `sampler.py` | emcee driver, `MCMCResult` (summary / save / load), walker initialisation | `mock_data/mcmc.py` (`get_6614_mcmc`), `ai/Q2.../multi_star_mcmc.py` |
| `selection.py` | AIC / BIC / Delta-IC, chi2-style log-likelihood | `ai/Q2.../multi_star_mcmc.py` (`aic_bic`), `ai/Q03.../run_voigt_fit.py` |
| `quality.py` | Quality labels from `L / tau_DIB` (Success / Mild / Severe) and `N_eff` | paper Section "Quality Labels"; `cross/final_catalog/update_autocorr_neff.py` (`N_eff = 5000/tau`) |
| `preprocess.py` | LAMOST MRS reading, cut, resample, normalise, save/read preprocessed FITS | `cross/src/preprocess.py` |
| `specnorm.py` | Continuum normalisation and array helpers | `cross/src/specNormg.py` (verbatim copy) |
| `plotting.py` | Spectrum + fit, per-epoch grid, corner plots, decomposition, tau bars | figure style of `ai/Q06.../plot_q6_3x2.py`, `ai/Q03.../gamma0.6/plot_fig9_*.py` |
| `io.py` | `load_result`, `load_mock`, `save_summary_json` | new |
| `__main__.py` | `python -m mcmc_rvsd --version` | new |

## 2. Data (`data/`)

| Path | Content |
|---|---|
| `mock/mock_6614_gaussian.npz` | fiducial mock: 1 stellar line + Gaussian DIB, 24 epochs (tutorial 01) |
| `mock/mock_6614_voigt.npz` | DIB with a mild Lorentzian component, gamma = 0.10 A (tutorial 03) |
| `mock/mock_6614_voigt_gamma06.npz` | stress case of the paper, gamma = 0.60 A (tutorial 03) |
| `mock/mock_multistar.npz` | three *blended* stellar lines (Fe I, Th I, Y II) + one DIB, configuration (a) of the paper (tutorial 02) |
| `mock/mock_multistar_unblended.npz` | the same three lines spread over the window, configuration (b) (tutorial 02) |
| `mock/mock_6379.npz` | lambda 6379 mock on the 6350-6410 A window (tutorial 04) |
| `mock/dib_scan_results.json` | the 50-point width-depth scan (tutorial 04) |
| `real/<uid>/raw/<uid>_<obsid>.fits` | one raw LAMOST DR12 MRS file per target (red + blue channels, ~0.5 MB) |
| `real/<uid>/preprocessed.fits` | the three observed LAMOST MRS targets, normalised and resampled (all epochs) |
| `real/<uid>/mcmc_result.npz` | posterior chains of the fiducial fit (thinned by 10) |
| `real/<uid>/mcmc_summary.json` | fitted parameters, AIC/BIC, tau, quality label, runtime |
| `real/targets.csv` | uid, label, LAMOST obsids and spectrum names, epochs, RV span, SNR |

## 3. Results of the tutorials (`examples/results/`)

| File | Content |
|---|---|
| `mock_6614_gaussian.npz` | fiducial 6-parameter fit |
| `mock_6614_voigt_gauss.npz` / `mock_6614_voigt_voigt.npz` | Gaussian vs Voigt fit, gamma = 0.10 A |
| `mock_6614_voigt06_gauss.npz` / `mock_6614_voigt06_voigt.npz` | Gaussian vs Voigt fit, gamma = 0.60 A |
| `mock_multistar_1line.npz` / `mock_multistar_3line.npz` | one-line vs three-line model, blended configuration |
| `mock_unblended_1line.npz` / `mock_unblended_3line.npz` | one-line vs three-line model, unblended configuration (the 1-line model overestimates the DIB by +4 sigma) |
| `other_dib_6379.npz` | lambda 6379 configuration demo |
| `*.json` | the same summaries in JSON |

## 4. Scripts (`scripts/`)

| File | Purpose |
|---|---|
| `generate_mock_data.py` | rebuild `data/mock/*.npz` (deterministic, seed 42) |
| `run_examples.py` | rebuild `examples/results/*.npz` (48 walkers, 1500 + 5000 steps) |
| `run_real_targets.py` | refit the three observed targets from `data/real/*/preprocessed.fits` |
| `download_lamost_spectra.py` | fetch the raw LAMOST DR12 spectra by obsid |
| `compress_figures.py` | downscale `docs/figures/*.png` after re-executing notebooks (they are saved at dpi 300) |

## 5. Where the research files live (source project)

| What you are looking for | Path |
|---|---|
| Original MCMC for lambda 6614 | `mock_data/mcmc.py`, `mock_data/mock_data.ipynb` |
| Multi-stellar-line test | `ai/Q2_stellar_lines/`, `ai/Q2_多恒星线实测/` |
| Non-Gaussian (Voigt) test | `ai/Q03_非高斯轮廓/` (incl. `gamma0.6/`) |
| DIB width-depth scan, other DIBs | `ai/dib_scan/`, `ai/Q04_可测DIB数量/` |
| Real-data decomposition figures | `ai/Q06_真实数据分解展示/` |
| Full catalogue pipeline (434 targets) | `cross/src/`, `cross/final_catalog/` |
| LAMOST raw spectra pools | `mock_data/lamost/spec/`, `cross/data/spectra/` |
| Manuscript, response letter, tables | `ai/引用/paper.txt`, `ai/引用/letter.txt`, `raa/` |
