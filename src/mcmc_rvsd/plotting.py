"""Figures: spectrum + fit, per-epoch panels, corner plots, decompositions.

The style follows the figures of the paper: black frames, bold axis labels, the
input (true) spectrum drawn as a thick translucent band at the bottom and the
models drawn on top of it.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt    # the backend is left to the caller: notebooks use
                                   # %matplotlib inline, batch scripts fall back to Agg
from matplotlib.lines import Line2D

C_DATA = '#546E7A'
C_STAR = '#E69F00'
C_DIB = '#009E73'
C_GAUSS = '#EC3232'
C_VOIGT = '#0787C3'
C_TRUTH = 'k'

LABEL_FS = 14
TICK_FS = 12


def style_axes(ax, spine=1.6):
    """Thick black frame, bold labels: the style used throughout the paper."""
    for s in ax.spines.values():
        s.set_linewidth(spine)
        s.set_color('black')
    ax.tick_params(axis='both', which='major', width=spine, length=6, labelsize=TICK_FS)


def _save(fig, out):
    if out:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig


def plot_spectrum_fit(wave, flux, flux_err, rvs, model, pfit, epoch=0, ptrue=None,
                      out=None, title=None, zoom=None):
    """Observed spectrum of one epoch with the model, plus the residuals below.

    ``ptrue`` (optional) draws the input components as dashed curves; ``zoom`` is
    the velocity/orbit index used in the legend of ``ptrue`` panels.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    y = flux[epoch]
    e = flux_err[epoch]
    mod = model.flux(wave, rvs[epoch], pfit)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1], hspace=0.0))
    if ptrue is not None:
        comp = model.component_fluxes(wave, rvs[epoch], ptrue)
        for key, value in comp.items():
            if key == 'total':
                ax1.plot(wave, value, color=C_TRUTH, ls='-', lw=8, alpha=0.4, zorder=1)
            else:
                ax1.plot(wave, value, ls='--', lw=1.8, zorder=3,
                         color=C_STAR if key != 'DIB' else C_DIB)
    ax1.errorbar(wave, y, yerr=e, fmt='o', ms=4.5, color=C_DATA, alpha=0.9, label='Observed',
                 markeredgewidth=0.6, markeredgecolor='#263238', capsize=2, elinewidth=0.5, zorder=10)
    ax1.plot(wave, mod, color=C_GAUSS, lw=2.2, zorder=9, label='Fit')
    handles, labels = ax1.get_legend_handles_labels()
    if ptrue is not None:
        handles += [Line2D([0], [0], color=C_STAR, ls='--', lw=1.8),
                    Line2D([0], [0], color=C_DIB, ls='--', lw=1.8)]
        labels += ['Input stellar line', 'Input DIB']
    style_axes(ax1)
    ax1.legend(handles, labels, frameon=False, fontsize=11, loc='lower left')
    ax1.set_ylabel('Normalised flux', fontsize=LABEL_FS, fontweight='bold')
    ax1.set_xlim(wave[0], wave[-1])
    if title:
        ax1.set_title(title, fontsize=LABEL_FS)

    ax2.errorbar(wave, (y - mod) * 1e3, yerr=e * 1e3, fmt='o', ms=3.4, color=C_GAUSS,
                 alpha=0.85, capsize=1.6, elinewidth=0.6, zorder=6)
    ax2.axhline(0, color='k', ls='--', lw=1)
    ax2.fill_between([wave[0], wave[-1]], -1e3 * np.median(e), 1e3 * np.median(e),
                     alpha=0.15, color='gray', zorder=1)
    style_axes(ax2)
    ax2.set_xlabel(r'Wavelength ($\rm\AA$)', fontsize=LABEL_FS, fontweight='bold')
    ax2.set_ylabel('Residual ($10^{-3}$)', fontsize=LABEL_FS - 2, fontweight='bold')
    ax2.text(0.995, 0.05, r'grey band: $\pm1\sigma$ noise level', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=10, color='0.35')
    return _save(fig, out)


def plot_all_epochs(wave, flux, flux_err, rvs, model, pfit, out=None, ncols=3,
                    rv_span=True, title=None):
    """Grid of panels, one per epoch, with the same model overplotted."""
    flux = np.asarray(flux, dtype=float)
    n = len(rvs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.9 * nrows),
                             sharex=True, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis('off')
            continue
        mod = model.flux(wave, rvs[i], pfit)
        ax.errorbar(wave, flux[i], yerr=flux_err[i], fmt='o', ms=3.0, color=C_DATA,
                    alpha=0.8, capsize=1.2, elinewidth=0.4, zorder=10)
        ax.plot(wave, mod, color=C_GAUSS, lw=1.6, zorder=9)
        style_axes(ax, spine=1.2)
        ax.text(0.03, 0.06, rf'$v={rvs[i]:.1f}$ km/s', transform=ax.transAxes, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel(r'Wavelength ($\rm\AA$)', fontsize=LABEL_FS - 2)
    for ax in axes[:, 0]:
        ax.set_ylabel('Normalised flux', fontsize=LABEL_FS - 2)
    if title:
        fig.suptitle(title, fontsize=LABEL_FS)
    fig.tight_layout()
    return _save(fig, out)


def plot_corner(samples, names, truths=None, out=None, label_kwargs=None):
    """Corner plot of the posterior samples (16/50/84 percentiles shown)."""
    import corner
    samples = np.asarray(samples)
    n = samples.shape[1]
    fig = corner.corner(samples, labels=list(names), truths=truths,
                        figsize=(min(2.2 * n + 1.0, 18.0),) * 2,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt='.3f', title_kwargs=dict(fontsize=10),
                        truth_color='red', truth_fmt='.3f',
                        label_kwargs=label_kwargs or dict(fontsize=14))
    return _save(fig, out)


def plot_decomposition(wave, rv, model, ptrue, out=None):
    """Dashed input components (stellar lines, DIB) under the input total."""
    wave = np.asarray(wave, dtype=float)
    comp = model.component_fluxes(wave, rv, ptrue)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wave, comp['total'], color=C_TRUTH, lw=10, alpha=0.4, zorder=1, label='Input total')
    for key, value in comp.items():
        if key == 'total':
            continue
        ax.plot(wave, value, ls='--', lw=1.8, zorder=3,
                color=C_STAR if key != 'DIB' else C_DIB,
                label=f'Input {key}')
    style_axes(ax)
    ax.set_xlabel(r'Wavelength ($\rm\AA$)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel('Normalised flux', fontsize=LABEL_FS, fontweight='bold')
    ax.legend(frameon=False, fontsize=12)
    return _save(fig, out)


def plot_autocorr(taus, names=None, out=None):
    """Bar chart of the autocorrelation times with the quality-label thresholds."""
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(taus)), taus, color=C_VOIGT, alpha=0.85)
    ax.axhline(100, ls=':', color='k', lw=1.5)
    ax.axhline(200, ls='--', color='k', lw=1.5)
    ax.text(0.02, 105, r'$\tau=100$ (Success $\to$ Mild)', transform=ax.get_yaxis_transform(), fontsize=10)
    ax.text(0.02, 205, r'$\tau=200$ (Mild $\to$ Severe)', transform=ax.get_yaxis_transform(), fontsize=10)
    ax.set_xticks(np.arange(len(taus)))
    ax.set_xticklabels(names or [f'p{i}' for i in range(len(taus))], fontsize=11)
    ax.set_ylabel(r'Autocorrelation time $\tau$', fontsize=LABEL_FS, fontweight='bold')
    style_axes(ax)
    return _save(fig, out)
