"""Integration test: run PeriodSearch on the B1392 RR Lyrae gold dataset.

Expected result: best period ≈ 0.5016247 days (from the original self-test).
"""

import importlib.resources
import numpy as np
import pytest

from psearch import PeriodSearch


def _load_b1392():
    """Load the B1392all.tab sample data bundled with the package."""
    data_path = importlib.resources.files('psearch.data').joinpath('B1392all.tab')
    hjd, mag, magerr, filts = np.loadtxt(str(data_path), unpack=True)
    # apply the standard quality cut used in the original self-test
    ok = (magerr >= 0.0) & (magerr <= 0.2)
    return hjd[ok], mag[ok], magerr[ok], filts[ok]


@pytest.fixture(scope='module')
def b1392_result():
    hjd, mag, magerr, filts = _load_b1392()
    # Single-filter run (filter 0 = 'u') with n_thresh=0 for speed in CI
    ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=['u'])
    return ps.run(pmin=0.2, dphi=0.02, n_thresh=0)


def test_best_period_close_to_gold(b1392_result):
    """Best period should be within 0.001 days of the gold value 0.5016247."""
    gold = 0.5016247
    assert abs(b1392_result.best_period - gold) < 0.001, (
        "best_period = %.7f, expected ~%.7f" % (b1392_result.best_period, gold)
    )


def test_ptest_is_1d(b1392_result):
    assert b1392_result.ptest.ndim == 1
    assert len(b1392_result.ptest) > 0


def test_psi_shape_matches_ptest(b1392_result):
    assert b1392_result.psi_m.shape == b1392_result.ptest.shape


def test_top_periods_returns_table(b1392_result):
    tab = b1392_result.top_periods(n=5)
    assert len(tab) == 5
    assert 'period' in tab.colnames
    assert 'power' in tab.colnames
