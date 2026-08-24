"""Unit tests for pycycle.periodogram.compute_periodogram."""

import importlib.resources
import numpy as np

from pycycle.periodogram import compute_periodogram

_BAND_NAMES = np.array(['u', 'g', 'r', 'i', 'z'])


def _load_b1392():
    data_path = importlib.resources.files('pycycle.data').joinpath('B1392all.tab')
    hjd, mag, magerr, filts_idx = np.loadtxt(str(data_path), unpack=True)
    filts = _BAND_NAMES[filts_idx.astype(int)]
    ok = (magerr >= 0.0) & (magerr <= 0.2)
    return hjd[ok], mag[ok], magerr[ok], filts[ok]


def test_compute_periodogram_returns_six_values():
    hjd, mag, magerr, filts = _load_b1392()
    out = compute_periodogram(hjd, mag, magerr, filts, fwant='u',
                               pmin=0.2, dphi=0.02, n_thresh=0)
    assert len(out) == 6
    x, fy, theta, psi, conf, nok = out
    assert x.shape == fy.shape == theta.shape == psi.shape == conf.shape
    assert isinstance(nok, (int, np.integer))


def test_nok_matches_quality_and_band_cut():
    hjd, mag, magerr, filts = _load_b1392()
    _, _, _, _, _, nok = compute_periodogram(hjd, mag, magerr, filts, fwant='g',
                                              pmin=0.2, dphi=0.02, n_thresh=0)
    expected = int(np.sum((filts == 'g') & (magerr >= 0.0) & (magerr <= 0.2)))
    assert nok == expected


def test_nok_zero_for_absent_band():
    hjd, mag, magerr, filts = _load_b1392()
    _, _, _, _, _, nok = compute_periodogram(hjd, mag, magerr, filts, fwant='not_a_band',
                                              pmin=0.2, dphi=0.02, n_thresh=0)
    assert nok == 0
