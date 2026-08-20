"""pycycle — hybrid period-finding for multiband variable-star light curves.

Based on Saha & Vivas (2017, AJ 154, 231):
"A Hybrid Algorithm for Period Analysis from Multiband Data with Sparse and
Irregular Sampling for Arbitrary Light-curve Shapes"

Quick start::

    from pycycle import PeriodSearch
    import numpy as np

    hjd, mag, magerr = np.loadtxt('data.tab', usecols=(0, 1, 2), unpack=True)
    filts = np.loadtxt('data.tab', usecols=3, dtype=str)
    ps = PeriodSearch(hjd, mag, magerr, filts)
    result = ps.run(pmin=0.2, dphi=0.02)
    print(result.best_period)
    result.plot_phased()

Rubin DP2 catalogs (see :mod:`pycycle.dp2`)::

    from pycycle.dp2 import DP2Config, fit_object_ids

    cfg = DP2Config(template_dir='~/software/rr-templates/template_des')
    results = fit_object_ids([735954534639105918, 738184412939704186], cfg)
"""

from .core import PeriodSearch, PeriodSearchResult
from .results import results_table
from .templates import (load_rr_template, load_multiband_template,
                         load_multiband_templates, load_multiband_dir,
                         is_multiband_dir,
                         average_multiband_templates, RRTemplate)
from .template_fit import TemplateFitter, TemplateFitResult
from .dp2 import (DP2Config, clean_epochs, fit_lightcurve, fit_catalog,
                   fit_object_ids, make_dp2_fit_fn, open_dp2)

__version__ = '0.24.0'

__all__ = [
    'PeriodSearch',
    'PeriodSearchResult',
    'results_table',
    'RRTemplate',
    'load_rr_template',
    'load_multiband_template',
    'load_multiband_templates',
    'load_multiband_dir',
    'is_multiband_dir',
    'average_multiband_templates',
    'TemplateFitter',
    'TemplateFitResult',
    # Rubin DP2 / LSDB integration
    'DP2Config',
    'open_dp2',
    'clean_epochs',
    'fit_lightcurve',
    'make_dp2_fit_fn',
    'fit_catalog',
    'fit_object_ids',
    '__version__',
]
