"""pycycle — hybrid period-finding for multiband variable-star light curves.

Based on Saha & Vivas (2017, AJ 154, 231):
"A Hybrid Algorithm for Period Analysis from Multiband Data with Sparse and
Irregular Sampling for Arbitrary Light-curve Shapes"

Quick start::

    from pycycle import PeriodSearch
    import numpy as np

    hjd, mag, magerr, filts = np.loadtxt('data.tab', unpack=True)
    ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=['V'])
    result = ps.run(pmin=0.2, dphi=0.02)
    print(result.best_period)
    result.plot_phased()
"""

from .core import PeriodSearch, PeriodSearchResult
from .results import results_table
from .templates import (load_rr_template, load_multiband_template,
                         load_multiband_templates, average_multiband_templates,
                         RRTemplate)
from .template_fit import TemplateFitter, TemplateFitResult

__version__ = '0.24.0'

__all__ = [
    'PeriodSearch',
    'PeriodSearchResult',
    'results_table',
    'RRTemplate',
    'load_rr_template',
    'load_multiband_template',
    'load_multiband_templates',
    'average_multiband_templates',
    'TemplateFitter',
    'TemplateFitResult',
    '__version__',
]
