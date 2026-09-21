#!/usr/bin/env python
"""Downscale the tutorial figures in ``docs/figures`` before committing.

The notebooks save their figures at dpi = 300 (good for a paper), which makes the
repository heavy once every notebook has been executed again.  This script resizes
every PNG to at most ``--max-width`` pixels and re-encodes it, cutting the figure
folder by roughly a factor of three.  Run it after re-executing notebooks::

    python scripts/compress_figures.py [--max-width 1600] [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

DEFAULT_DIR = Path(__file__).resolve().parents[1] / 'docs' / 'figures'


def compress(figdir=DEFAULT_DIR, max_width=1600, dry_run=False):
    figdir = Path(figdir)
    before = after = 0
    for path in sorted(figdir.glob('*.png')):
        size_before = path.stat().st_size
        image = Image.open(path)
        new_size = image.size
        if image.width > max_width:
            new_size = (max_width, round(image.height * max_width / image.width))
            image = image.resize(new_size, Image.LANCZOS)
        if not dry_run:
            image.convert('RGB').save(path, optimize=True, compress_level=9)
        size_after = path.stat().st_size if not dry_run else size_before
        before += size_before
        after += size_after
        print(f'{path.name:34s} {image.size[0]}x{image.size[1]:<5d} '
              f'{size_before / 1e6:6.2f} -> {size_after / 1e6:5.2f} MB')
    print(f'TOTAL {before / 1e6:.1f} -> {after / 1e6:.1f} MB')
    return before, after


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figdir', default=str(DEFAULT_DIR))
    parser.add_argument('--max-width', type=int, default=1600)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    compress(args.figdir, args.max_width, args.dry_run)
