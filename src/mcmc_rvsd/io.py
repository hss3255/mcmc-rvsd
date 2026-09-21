"""Small helpers to read/write the mock data and the MCMC results."""
from __future__ import annotations

import json
import os

import numpy as np

from .sampler import MCMCResult


def save_summary_json(result, path, **extra):
    """Write the posterior summary of an MCMCResult to JSON (human readable)."""
    payload = result.as_dict()
    payload.update({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in extra.items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_result(path):
    """Load a saved result (as written by :meth:`MCMCResult.save`) as an MCMCResult."""
    with np.load(path, allow_pickle=False) as d:
        raw = {k: d[k] for k in d.files}
    names = [str(s) for s in raw.pop('param_names', raw.get('param_names', []))]
    samples = np.asarray(raw.pop('samples'))
    label = raw.pop('label', '')
    label = str(label) if isinstance(label, np.ndarray) else label
    scalars = {}
    for key in ('lnL', 'aic', 'bic', 'runtime'):
        if key in raw:
            value = raw.pop(key)
            scalars[key] = float(np.asarray(value).ravel()[0]) if np.size(value) else float('nan')
    for key in ('nwalkers', 'burnin', 'production'):
        if key in raw:
            value = raw.pop(key)
            scalars[key] = int(np.asarray(value).ravel()[0])
    known = {'pfit', 'perr', 'p16', 'p84', 'tau'}
    arrays = {k: np.asarray(v).ravel() for k, v in raw.items() if k in known}
    extra = {k: v for k, v in raw.items() if k not in known}
    if 'lnL_best' in extra:                   # keep it next to the other scalars
        extra['lnL_best'] = float(np.asarray(extra['lnL_best']).ravel()[0])
    for key in ('pfit', 'perr', 'p16', 'p84', 'tau'):      # be tolerant of partial files
        arrays.setdefault(key, np.full(len(names), np.nan))
    return MCMCResult(param_names=names, samples=samples, label=label, extra=extra,
                      **arrays, **scalars)


def load_mock(path):
    """Load a mock dataset written by ``scripts/generate_mock_data.py``."""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}
