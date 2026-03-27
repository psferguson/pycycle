"""psearch — hybrid period-finding for multiband variable-star light curves.

Based on Saha & Vivas (2017, AJ 154, 231):
"A Hybrid Algorithm for Period Analysis from Multiband Data with Sparse and
Irregular Sampling for Arbitrary Light-curve Shapes"

Quick start::

    from psearch import PeriodSearch
    import numpy as np

    hjd, mag, magerr, filts = np.loadtxt('data.tab', unpack=True)
    ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=['V'])
    result = ps.run(pmin=0.2, dphi=0.02)
    print(result.best_period)
    result.plot_phased()
"""

from .core import PeriodSearch, PeriodSearchResult
from .results import results_table

__version__ = '0.24.0'

__all__ = [
    'PeriodSearch',
    'PeriodSearchResult',
    'results_table',
    '__version__',
]
