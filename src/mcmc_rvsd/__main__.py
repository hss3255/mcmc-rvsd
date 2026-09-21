"""Command line entry point: ``python -m mcmc_rvsd`` prints the version/API."""
from __future__ import annotations

import argparse

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog='mcmc_rvsd',
                                     description='MCMC-RVSD: DIB measurements in binary-star spectra')
    parser.add_argument('--version', action='version', version=f'mcmc_rvsd {__version__}')
    args = parser.parse_args(argv)
    print(f'mcmc_rvsd {__version__} - see README.md and the notebooks in examples/')


if __name__ == '__main__':
    main()
