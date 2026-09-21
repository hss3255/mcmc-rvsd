"""LAMOST MRS preprocessing: normalise, cut and resample the red channel.

The pipeline reproduces what was applied to the observed spectra of the paper:

1. read the LAMOST medium-resolution FITS (the red channel, highest SNR);
2. convert the vacuum wavelengths to air wavelengths;
3. cut the region around the DIB (default 6604-6664 A);
4. resample onto the common 0.12 A grid;
5. normalise the continuum iteratively (:mod:`mcmc_rvsd.specnorm`);
6. interpolate over anomalous points (normalised flux > 1.15).

The result is written as a binary FITS table with the columns
``WAVES, FLUXS, Y_ERRS, RVS, RV_ERRS, SNRS, OBSDATES`` - one row per target,
one array element per epoch - which is the format read back by
:func:`read_preprocessed`.
"""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import specnorm

WAVE_START = 6604.0    # Angstrom
WAVE_END = 6664.0      # Angstrom
WAVE_STEP = 0.12       # Angstrom (same sampling as the mock data)


def create_wave_grid(wave_start=WAVE_START, wave_end=WAVE_END, step=WAVE_STEP):
    """Common wavelength grid of the analysis."""
    return np.arange(wave_start, wave_end + step / 2, step)


def read_lamost_fits(filepath):
    """Read one LAMOST MRS FITS file and return the red-channel arrays.

    Returns
    -------
    dict with ``wave`` (vacuum), ``flux``, ``flux_err``, ``mask``, ``snr``,
    ``rv`` (km/s) and ``rv_err`` (km/s); bad pixels are set to NaN in the flux.
    """
    with fits.open(filepath) as hdul:
        best_hdu, best_snr = None, 0.0
        for hdu in hdul:
            if not hasattr(hdu, 'data') or hdu.data is None:
                continue
            if not hasattr(hdu.data, 'dtype') or not hasattr(hdu.data.dtype, 'names'):
                continue
            if 'FLUX' not in hdu.data.dtype.names:
                continue
            flux_data = hdu.data['FLUX'][0] if hdu.data['FLUX'].ndim > 1 else hdu.data['FLUX']
            if 'IVAR' in hdu.data.dtype.names:
                ivar_data = hdu.data['IVAR'][0] if hdu.data['IVAR'].ndim > 1 else hdu.data['IVAR']
                snr_est = float(np.nanmedian(flux_data * np.sqrt(np.abs(ivar_data))))
            else:
                snr_est = float(np.nanmedian(flux_data) / np.nanstd(flux_data))
            is_red = 'R' in str(getattr(hdu, 'name', '')).upper()
            if is_red or snr_est > best_snr:
                best_snr, best_hdu = snr_est, hdu
        if best_hdu is None:
            raise ValueError(f'no HDU with a FLUX column found in {filepath}')

        data = best_hdu.data
        names = data.dtype.names
        flux = np.asarray(data['FLUX'][0] if data['FLUX'].ndim > 1 else data['FLUX'], dtype=float)
        if 'WAVELENGTH' in names:
            wave = np.asarray(data['WAVELENGTH'][0] if data['WAVELENGTH'].ndim > 1 else data['WAVELENGTH'], dtype=float)
        elif 'WAVE' in names:
            wave = np.asarray(data['WAVE'][0] if data['WAVE'].ndim > 1 else data['WAVE'], dtype=float)
        else:
            raise ValueError(f'no wavelength column found in {filepath}')

        if 'IVAR' in names:
            ivar = np.asarray(data['IVAR'][0] if data['IVAR'].ndim > 1 else data['IVAR'], dtype=float)
            flux_err = np.zeros_like(flux)
            good = ivar > 0
            flux_err[good] = 1.0 / np.sqrt(ivar[good])
            flux_err[~good] = np.median(flux_err[good]) if np.any(good) else 1.0
        else:
            flux_err = np.full_like(flux, np.std(flux) / 10.0)

        mask = np.zeros_like(flux, dtype=bool)
        for col in ('ANDMASK', 'ORMASK', 'PIXMASK'):
            if col in names:
                m = np.asarray(data[col][0] if data[col].ndim > 1 else data[col])
                mask = mask | (m > 0)
        flux[mask] = np.nan
        flux_err[mask] = np.nan

        rv, rv_err = 0.0, 1.0
        head = hdul[0].header
        if 'HELIO_RV' in head:
            rv = float(head['HELIO_RV'])
        elif 'RV' in head:
            rv = float(head['RV'])
        for key in ('RV', 'VRAD'):
            if rv == 0.0 and key in names:
                arr = data[key]
                rv = float(arr[0] if arr.ndim > 1 else arr)
        for key in ('RV_ERR', 'VRAD_ERR'):
            if key in names:
                arr = data[key]
                rv_err = float(arr[0] if arr.ndim > 1 else arr)
                break
    return dict(wave=wave, flux=flux, flux_err=flux_err, mask=mask, snr=best_snr,
                rv=rv, rv_err=rv_err)


def preprocess_spectrum(filepath, wave_grid=None, output_path=None, verbose=False):
    """Full preprocessing of one LAMOST MRS spectrum.

    Returns a dict with the normalised flux on the common grid plus the RV of the
    epoch, and optionally writes a preprocessed FITS file to ``output_path``.
    """
    wave_grid = create_wave_grid() if wave_grid is None else np.asarray(wave_grid, dtype=float)
    raw = read_lamost_fits(filepath)

    wave_air = specnorm.vac2air(raw['wave'])
    wave_cut, flux_cut, err_cut = specnorm.cut_array(wave_air, raw['flux'], raw['flux_err'],
                                                    wave_grid[0], wave_grid[-1])
    if len(wave_cut) < 10:
        raise ValueError(f'only {len(wave_cut)} pixels left after cutting {filepath}')

    flux_rebin = specnorm.rebin_array(wave_cut, flux_cut, wave_grid)
    err_rebin = specnorm.rebin_array(wave_cut, err_cut, wave_grid)

    finite = np.isfinite(flux_rebin) & np.isfinite(err_rebin)
    if not np.all(finite) and np.any(finite):
        flux_rebin[~finite] = np.nanmedian(flux_rebin)
        err_rebin[~finite] = np.nanmedian(err_rebin)

    ivar = 1.0 / (err_rebin ** 2)
    ivar[~np.isfinite(ivar)] = 1e-6
    try:
        continuum, flux_norm, flux_norm_err = specnorm.specNormg(
            order=1, spec=flux_rebin, ivar_norm=ivar, niter=20, lowrej=1, highrej=3)
    except Exception as exc:  # fall back to a simple median normalisation
        if verbose:
            print(f'  normalisation failed ({exc}); using the median')
        med = np.nanmedian(flux_rebin)
        continuum = np.full_like(flux_rebin, med)
        flux_norm, flux_norm_err = flux_rebin / med, err_rebin / med

    finite = np.isfinite(flux_norm) & np.isfinite(flux_norm_err)
    if not np.all(finite):
        if np.any(finite):
            flux_norm[~finite] = np.nanmean(flux_norm[finite])
            flux_norm_err[~finite] = np.nanmean(flux_norm_err[finite])
        else:
            flux_norm = np.ones_like(flux_norm)
            flux_norm_err = np.full_like(flux_norm, 0.1)

    # cosmic rays / spikes
    anomaly = flux_norm > 1.15
    if np.any(anomaly):
        good = np.where(~anomaly)[0]
        if len(good):
            flux_norm[anomaly] = np.interp(wave_grid[anomaly], wave_grid[good], flux_norm[good])
            flux_norm_err[anomaly] = np.interp(wave_grid[anomaly], wave_grid[good], flux_norm_err[good])

    result = dict(wave=wave_grid, flux_norm=flux_norm, flux_norm_err=flux_norm_err,
                  continuum=continuum, rv=raw['rv'], rv_err=raw['rv_err'], snr=raw['snr'],
                  original_file=os.path.basename(filepath))
    if output_path:
        save_preprocessed(result, output_path)
    return result


def save_preprocessed(result, output_path):
    """Write a preprocessed spectrum (or a stack of epochs) to a FITS table."""
    t = Table()
    t['WAVES'] = [np.asarray(result['wave'], dtype=float)]
    t['FLUXS'] = [np.asarray(result['flux_norm'], dtype=float)]
    t['Y_ERRS'] = [np.asarray(result['flux_norm_err'], dtype=float)]
    t['RVS'] = [np.atleast_1d(result['rv']).astype(float)]
    t['RV_ERRS'] = [np.atleast_1d(result['rv_err']).astype(float)]
    t['SNRS'] = [np.atleast_1d(result['snr']).astype(float)]
    t['OBSDATES'] = [str(result.get('original_file', ''))]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    t.write(output_path, overwrite=True)
    return output_path


def read_preprocessed(path):
    """Read a preprocessed FITS file into a dict of arrays.

    Works both for the multi-epoch files shipped with the examples and for the
    single-spectrum files written by :func:`save_preprocessed`.
    """
    with fits.open(path) as hdul:
        data = hdul[1].data
        names = set(data.dtype.names or ())
        wave = np.asarray(data['WAVES'][0], dtype=float)
        flux = np.asarray(data['FLUXS'], dtype=float)
        flux_err = np.asarray(data['Y_ERRS'], dtype=float)
        rvs = np.atleast_1d(np.asarray(data['RVS'], dtype=float).ravel())
        rv_errs = np.atleast_1d(np.asarray(data['RV_ERRS'], dtype=float).ravel())
        if 'SNRS' in names:
            snrs = np.atleast_1d(np.asarray(data['SNRS'], dtype=float).ravel())
        elif 'SNR' in names:
            snrs = np.atleast_1d(np.asarray(data['SNR'], dtype=float).ravel())
        else:
            snrs = np.ones(len(rvs))
        if 'OBSDATES' in names:
            obsdates = [str(s) for s in np.atleast_1d(data['OBSDATES'])]
        elif 'ORIGINAL_FILE' in names:
            obsdates = [str(s) for s in np.atleast_1d(data['ORIGINAL_FILE'])]
        else:
            obsdates = [f'epoch{i}' for i in range(len(rvs))]
    if flux.ndim == 1:
        flux = flux[np.newaxis, :]
        flux_err = flux_err[np.newaxis, :]
    return dict(wave=wave, flux=flux, flux_err=flux_err, rvs=rvs, rv_errs=rv_errs,
                snrs=snrs, obsdates=obsdates)
