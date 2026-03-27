"""Per-filter hybrid PSI periodogram (Lomb-Scargle + Lafler-Kinman).

This module implements ``compute_periodogram``, which corresponds to the
``periodpsi2_py`` function in the original monolithic implementation.
"""

import time as _time

import numpy as np

from .scargle import scargle_fast
from .lafler_kinman import ctheta_slave
from .stats import scramble


def compute_periodogram(
    hjd,
    mag,
    magerr,
    filts,
    fwant,
    pmin,
    dphi,
    n_thresh=1,
    pmax=None,
    periods=None,
    verbose=False,
):
    """Compute the hybrid PSI periodogram for a single filter band.

    Parameters
    ----------
    hjd : ndarray of float64, shape (N,)
        Heliocentric Julian Dates of all observations.
    mag : ndarray of float64, shape (N,)
        Magnitudes co-aligned with *hjd*.
    magerr : ndarray of float64, shape (N,)
        Magnitude errors co-aligned with *hjd*.
    filts : ndarray of float64, shape (N,)
        Integer filter codes co-aligned with *hjd*.
    fwant : int
        Filter code selecting the band to analyse.
    pmin : float
        Minimum period to test [days].
    dphi : float
        Maximum allowed phase change between consecutive test periods.
    n_thresh : int, optional
        Number of Monte Carlo runs for the significance threshold (default 1).
        Set to 0 to skip threshold computation.
    pmax : float, optional
        Maximum period to test [days].
    periods : ndarray, optional
        Explicit array of test periods; overrides the auto-generated grid.
    verbose : bool, optional
        Print extra diagnostic information.

    Returns
    -------
    x : ndarray of float64
        Test periods [days].
    fy : ndarray of float64
        Lomb-Scargle power at each period.
    theta : ndarray of float64
        Lafler-Kinman theta at each period.
    psi : ndarray of float64
        Hybrid PSI = 2*fy/theta at each period.
    conf : ndarray of float64
        Significance threshold for PSI (sum of two noise realisations).
    """
    print('periodogram: BEGIN')

    # --- build period/frequency grid ---
    t0 = np.min(hjd)
    tspan = np.max(hjd) - t0
    maxfreq = 1.0 / pmin
    minfreq = 2.0 / tspan
    deltafreq = dphi / tspan
    nfreq = int((maxfreq - minfreq) / deltafreq)
    farray = minfreq + np.arange(nfreq) * deltafreq
    x = 1.0 / farray

    if periods is not None:
        x = periods.copy()
    elif pmax is not None:
        assert pmax > pmin
        idx = (x >= pmin) & (x <= pmax)
        assert np.count_nonzero(idx) > 0, "No periods in [pmin, pmax] range"
        x = x[idx].copy()
    farray = 1.0 / x
    nfreq = len(x)

    print(
        'periodogram: period range %14.8f – %14.8f days' % (x.min(), x.max())
    )
    print('periodogram: %d test periods' % nfreq)
    if verbose:
        print('periodogram: frequencies range %14.8f – %14.8f' %
              (farray.min(), farray.max()))

    omega = farray * 2.0 * np.pi  # scargle_fast uses angular frequencies

    # --- select observations for this filter (error cut: 0 <= err <= 0.2) ---
    ok = (filts == float(fwant)) & (magerr <= 0.2) & (magerr >= 0.0)
    tr = hjd[ok]
    yr = mag[ok]
    yr_err = magerr[ok]
    nok = len(tr)
    print('periodogram: %d observations for filter %s' % (nok, fwant))

    sss = np.argsort(tr)
    tr = tr[sss]
    yr = yr[sss]
    yr_err = yr_err[sss]

    # --- compute PSI ---
    t0_ = _time.time()
    fy = scargle_fast(tr, yr, omega, nfreq)
    print('scargle: done  %.3f s' % (_time.time() - t0_))

    t0_ = _time.time()
    theta = ctheta_slave(x, yr, tr)
    print('ctheta_slave: done  %.3f s' % (_time.time() - t0_))

    psi = 2.0 * fy / theta

    # --- Monte Carlo significance thresholds ---
    conf1 = np.zeros_like(psi)
    conf2 = np.zeros_like(psi)
    for count in range(n_thresh):
        print('periodogram: threshold run %d / %d' % (count + 1, n_thresh))

        # noise realisation
        er = yr_err * np.random.normal(0.0, 1.0, nok)
        fe = scargle_fast(tr, er, omega, nfreq)
        thetaerr = ctheta_slave(x, er, tr)
        conf1a = 2.0 * fe / thetaerr
        conf1b = conf1a * np.sum(psi) / np.sum(conf1a)
        conf1 = np.maximum(conf1, conf1b)

        # scrambled realisation
        zr, _ = scramble(yr)
        fz = scargle_fast(tr, zr, omega, nfreq)
        thetaz = ctheta_slave(x, zr, tr)
        conf2a = 2.0 * fz / thetaz
        conf2b = conf2a * np.sum(psi) / np.sum(conf2a)
        conf2 = np.maximum(conf2, conf2b)

    conf = conf1 + conf2

    print('periodogram: END')
    return x, fy, theta, psi, conf
