"""Integration test: run PeriodSearch on the B1392 RR Lyrae gold dataset.

Expected result: best period ≈ 0.5016247 days (from the original self-test).
"""

import importlib.resources
import numpy as np
import pytest

from pycycle import PeriodSearch
from pycycle.core import combine_periodograms, COMBINE_STRATEGIES


_BAND_NAMES = np.array(['u', 'g', 'r', 'i', 'z'])


def _load_b1392():
    """Load the B1392all.tab sample data bundled with the package."""
    data_path = importlib.resources.files('pycycle.data').joinpath('B1392all.tab')
    hjd, mag, magerr, filts_idx = np.loadtxt(str(data_path), unpack=True)
    filts = _BAND_NAMES[filts_idx.astype(int)]
    # apply the standard quality cut used in the original self-test
    ok = (magerr >= 0.0) & (magerr <= 0.2)
    return hjd[ok], mag[ok], magerr[ok], filts[ok]


@pytest.fixture(scope='module')
def b1392_result():
    hjd, mag, magerr, filts = _load_b1392()
    # Single-filter run on 'u' only, with n_thresh=0 for speed in CI
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


# ----------------------------------------------------------------------
# combine strategies / psi_per_band / psi_per_epoch (TODO.md items 1 & 2)
# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def multiband_search():
    hjd, mag, magerr, filts = _load_b1392()
    return PeriodSearch(hjd, mag, magerr, filts, filtnams=['g', 'r'])


@pytest.mark.parametrize('combine', COMBINE_STRATEGIES)
def test_combine_strategies_run_and_have_sane_shapes(multiband_search, combine):
    """Every documented `combine` strategy must run and return a usable result."""
    res = multiband_search.run(pmin=0.2, dphi=0.02, n_thresh=0, combine=combine)
    assert res.combine == combine
    assert res.psi_combined.shape == res.ptest.shape
    assert np.all(np.isfinite(res.psi_combined))
    # best_period must be a period actually on the grid
    assert res.best_period in res.ptest
    tab = res.top_periods(n=3)
    assert len(tab) == 3


def test_unknown_combine_strategy_raises(multiband_search):
    with pytest.raises(ValueError):
        multiband_search.run(pmin=0.2, dphi=0.02, n_thresh=0, combine='bogus')


def test_psi_per_band_shape_and_bands(multiband_search):
    res = multiband_search.run(pmin=0.2, dphi=0.02, n_thresh=0)
    assert res.psi_per_band.shape == (2, len(res.ptest))
    assert res.bands == ['g', 'r']


def test_psi_per_band_matches_single_band_runs(multiband_search):
    """Each row of psi_per_band should equal an independent single-band run."""
    res = multiband_search.run(pmin=0.2, dphi=0.02, n_thresh=0, periods=None)
    hjd, mag, magerr, filts = _load_b1392()
    for i, band in enumerate(['g', 'r']):
        single = PeriodSearch(hjd, mag, magerr, filts, filtnams=[band])
        single_res = single.run(pmin=0.2, dphi=0.02, n_thresh=0, periods=res.ptest)
        np.testing.assert_allclose(res.psi_per_band[i], single_res.psi_m)


def test_psi_per_band_always_2d_even_for_single_filter(b1392_result):
    """psi_m collapses to 1-D for one filter (back-compat); psi_per_band never does."""
    assert b1392_result.psi_m.ndim == 1
    assert b1392_result.psi_per_band.ndim == 2
    assert b1392_result.psi_per_band.shape[0] == 1
    np.testing.assert_allclose(b1392_result.psi_per_band[0], b1392_result.psi_m)


def test_combine_sum_matches_legacy_expression(multiband_search):
    """combine='sum' (the default) must exactly match the pre-existing
    `psi_m if ndim==1 else psi_m.sum(0)` idiom used by pycycle.dp2."""
    res = multiband_search.run(pmin=0.2, dphi=0.02, n_thresh=0)
    legacy = res.psi_m if res.psi_m.ndim == 1 else res.psi_m.sum(0)
    np.testing.assert_allclose(res.psi_combined, legacy)
    np.testing.assert_allclose(np.max(legacy), np.max(res.psi_combined))


def test_ranksum_and_normsum_resist_scale_domination():
    """A synthetic band with a huge-scale periodogram must not be able to
    dominate a rank- or scale-normalised combination the way it dominates
    a raw sum -- this is the core mechanism behind docs/TODO.md item 1.

    band 0 ("bad"): near-zero everywhere, a small-but-real value at the
    true index (150) representing weak genuine signal, and a spurious
    peak with a much larger raw magnitude at the wrong index (50) --
    exactly the shape a scale-dominant, wrong-period-favouring band has.
    band 1 ("good"): near-zero everywhere except a modest, but correct,
    peak at index 150.
    """
    psi_per_band = np.zeros((2, 200))
    psi_per_band[0, 150] = 0.5       # band 0's weak genuine signal at truth
    psi_per_band[0, 50] = 1.0e6      # band 0's much larger spurious peak
    psi_per_band[1, 150] = 5.0       # band 1 correctly and cleanly peaks at truth

    raw_sum = combine_periodograms(psi_per_band, method='sum')
    ranksum = combine_periodograms(psi_per_band, method='ranksum')
    normsum = combine_periodograms(psi_per_band, method='normsum')

    assert np.argmax(raw_sum) == 50          # raw sum is hijacked by band 0's scale
    assert np.argmax(ranksum) == 150         # ranksum is not
    assert np.argmax(normsum) == 150         # normsum is not


def test_combine_periodograms_single_band_is_a_noop():
    row = np.array([1.0, 5.0, 2.0])
    for method in COMBINE_STRATEGIES:
        out = combine_periodograms(row, method=method)
        np.testing.assert_allclose(out, row)
        assert out is not row  # documented to return a copy


def test_n_epochs_used_matches_quality_cut(multiband_search):
    res = multiband_search.run(pmin=0.2, dphi=0.02, n_thresh=0)
    hjd, mag, magerr, filts = _load_b1392()
    expected = int(np.sum(np.isin(filts, ['g', 'r']) & (magerr >= 0.0) & (magerr <= 0.2)))
    assert res.n_epochs_used == expected


def test_psi_per_epoch_scales_inversely_with_duplicated_epochs(multiband_search):
    """Duplicating every epoch of a light curve should roughly double the
    peak combined PSI (PSI ~ N) while leaving psi_per_epoch roughly fixed --
    that invariance is the whole point of the normalisation (TODO item 2)."""
    hjd, mag, magerr, filts = _load_b1392()
    ok = np.isin(filts, ['g', 'r'])
    hjd, mag, magerr, filts = hjd[ok], mag[ok], magerr[ok], filts[ok]

    ps1 = PeriodSearch(hjd, mag, magerr, filts, filtnams=['g', 'r'])
    res1 = ps1.run(pmin=0.2, dphi=0.02, n_thresh=0)

    hjd2 = np.concatenate([hjd, hjd + 1e-6])
    mag2 = np.concatenate([mag, mag])
    magerr2 = np.concatenate([magerr, magerr])
    filts2 = np.concatenate([filts, filts])
    ps2 = PeriodSearch(hjd2, mag2, magerr2, filts2, filtnams=['g', 'r'])
    res2 = ps2.run(pmin=0.2, dphi=0.02, n_thresh=0, periods=res1.ptest)

    assert res2.n_epochs_used == 2 * res1.n_epochs_used
    # psi_per_epoch should be much closer between the two runs than raw
    # peak PSI is -- that is the entire point of dividing by n_epochs_used.
    raw_ratio = np.max(res2.psi_combined) / np.max(res1.psi_combined)
    per_epoch_ratio = res2.psi_per_epoch / res1.psi_per_epoch
    assert raw_ratio > 1.5  # raw PSI grew substantially with N, as expected
    assert 0.5 < per_epoch_ratio < 2.0  # per-epoch PSI stayed roughly flat


def test_psi_per_epoch_nan_when_no_epochs():
    from pycycle.core import PeriodSearchResult
    ptest = np.linspace(0.3, 0.9, 10)
    psi_m = np.ones((2, 10))
    thresh_m = np.zeros((2, 10))
    res = PeriodSearchResult(ptest, psi_m, thresh_m,
                              np.array([]), np.array([]), np.array([]),
                              np.array([]), ['g', 'r'], n_epochs_used=0)
    assert np.isnan(res.psi_per_epoch)


def test_backward_compatible_dp2_usage_pattern(multiband_search):
    """Reproduce the exact expression pycycle.dp2.fit_lightcurve uses, to
    guarantee it keeps working identically regardless of new features."""
    res = multiband_search.run(pmin=0.2, dphi=0.02, n_thresh=1, pmax=2.0)
    period = float(res.best_period)
    psi = res.psi_m if res.psi_m.ndim == 1 else res.psi_m.sum(0)
    psi_peak = float(np.max(psi))
    tops = res.top_periods(n=2)
    alt_period = float(tops['period'][1])
    assert np.isfinite(period)
    assert np.isfinite(psi_peak)
    assert np.isfinite(alt_period)
