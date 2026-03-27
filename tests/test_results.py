"""Unit tests for the results module."""

import numpy as np
import pytest

from pycycle.results import results_table


def _make_periodogram(n=500, seed=0):
    rng = np.random.default_rng(seed)
    periods = np.linspace(0.2, 2.0, n)
    # inject a peak at periods[250]
    psi = rng.uniform(0, 1, n)
    psi[249] = 0.8
    psi[250] = 10.0  # dominant peak
    psi[251] = 0.9
    thresh = np.ones(n) * 2.0
    return periods, psi, thresh


def test_top_period_is_injected_peak():
    periods, psi, thresh = _make_periodogram()
    tab = results_table(periods, psi, thresh, n=1)
    assert tab['rank'][0] == 1
    assert abs(tab['period'][0] - periods[250]) < 1e-9


def test_table_has_correct_columns():
    periods, psi, thresh = _make_periodogram()
    tab = results_table(periods, psi, thresh, n=3)
    for col in ['rank', 'period', 'period_err', 'power', 'index', 'freq', 'thresh']:
        assert col in tab.colnames


def test_table_length():
    periods, psi, thresh = _make_periodogram()
    tab = results_table(periods, psi, thresh, n=5)
    assert len(tab) == 5


def test_frequency_is_reciprocal_of_period():
    periods, psi, thresh = _make_periodogram()
    tab = results_table(periods, psi, thresh, n=3)
    for row in tab:
        assert abs(row['freq'] - 1.0 / row['period']) < 1e-10
