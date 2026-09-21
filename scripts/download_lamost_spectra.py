#!/usr/bin/env python
"""Download the raw LAMOST DR12 medium-resolution spectra of the example targets.

The observed spectra used in the notebooks are public LAMOST DR12 data.  The
preprocessed versions are shipped inside ``data/real/<uid>/`` (small files); this
script fetches the *raw* FITS files (a few MB) if you want to redo the
preprocessing from scratch::

    python scripts/download_lamost_spectra.py --uids G13619209590707 ...

The obsids are resolved from ``data/real/targets.csv`` (columns: uid, label,
lamost_spec, obsids).  Files are written to ``data/real/<uid>/raw/<name>.fits``.
Note that the download URLs are the public DR12 medium-resolution spectrum
endpoint; a token may be required depending on the release.
"""
from __future__ import annotations

import argparse
import csv
import os
import urllib.request

URL = 'https://www.lamost.org/dr12/v1.1/medspectrum/fits/{obsid}'
HEADERS = {'User-Agent': 'Mozilla/5.0 (mcmc-rvsd)'}


def targets_from_csv(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def download(obsid, out_path, url=URL):
    req = urllib.request.Request(url.format(obsid=obsid), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp, open(out_path, 'wb') as fh:
        fh.write(resp.read())
    return out_path


def main(uids=None, datadir='data/real'):
    rows = targets_from_csv(os.path.join(datadir, 'targets.csv'))
    for row in rows:
        uid = row['uid']
        if uids and uid not in uids:
            continue
        outdir = os.path.join(datadir, uid, 'raw')
        os.makedirs(outdir, exist_ok=True)
        obsids = str(row.get('obsids', '')).replace(',', ' ').split()
        for obsid in obsids:
            out = os.path.join(outdir, f'med-{obsid}.fits')
            if os.path.exists(out):
                print(f'skip (exists): {out}')
                continue
            try:
                download(obsid, out)
                print(f'downloaded {out}')
            except Exception as exc:
                print(f'failed {obsid}: {exc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uids', nargs='*', default=None)
    parser.add_argument('--datadir', default='data/real')
    main(**vars(parser.parse_args()))
