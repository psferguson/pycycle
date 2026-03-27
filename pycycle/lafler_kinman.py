"""Lafler-Kinman phase-dispersion (theta) statistic.

Uses the compiled C extension when available; otherwise falls back to a
numba-JIT-compiled pure-Python version.
"""

import numpy as np

try:
    from pycycle._ext._pycycle_c import ctheta_slave as _ctheta_slave_c
    _USE_C = True
except ImportError:
    _USE_C = False

try:
    import numba
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False


def _ctheta_slave_py(parray, mag, tobs):
    """Pure-Python Lafler-Kinman theta (optimised variant).

    Computes the phase-dispersion statistic for each candidate period in
    *parray*.  About 35% faster than the original IDL-literal translation.
    """
    t0 = np.min(tobs)
    tt = tobs - t0
    theta = np.zeros_like(parray)
    mmplus_km = np.zeros_like(mag)
    avm_km = np.sum(mag) / len(mag)
    denom_km = np.sum((mag - avm_km) ** 2)
    for k in range(len(parray)):
        period = parray[k]
        phi = tt / period
        nphi = phi.astype(np.int64)
        phi = phi - nphi
        ss = np.argsort(phi)
        mm = mag[ss]
        mmplus_km[:-1] = mm[1:]
        mmplus_km[-1] = mm[0]
        numer = np.sum((mmplus_km - mm) ** 2)
        theta[k] = numer / denom_km
    return theta


if _USE_NUMBA:
    _ctheta_slave_pyjit = numba.jit(nopython=True)(_ctheta_slave_py)
else:
    _ctheta_slave_pyjit = _ctheta_slave_py


def ctheta_slave(parray, mag, tobs):
    """Compute the Lafler-Kinman theta statistic for an array of test periods.

    Uses the compiled C extension when available, otherwise falls back to a
    numba-JIT (or plain Python) implementation.

    Parameters
    ----------
    parray : ndarray of float64, shape (N,)
        Candidate periods to test.
    mag : ndarray of float64, shape (M,)
        Observed magnitudes.
    tobs : ndarray of float64, shape (M,)
        Observation times co-aligned with *mag*.

    Returns
    -------
    theta : ndarray of float64, shape (N,)
        Phase-dispersion value at each test period.  Smaller values indicate
        a better-phased light curve.
    """
    parray = np.ascontiguousarray(parray, dtype=np.float64)
    mag = np.ascontiguousarray(mag, dtype=np.float64)
    tobs = np.ascontiguousarray(tobs, dtype=np.float64)
    if _USE_C:
        return _ctheta_slave_c(parray, mag, tobs)
    return _ctheta_slave_pyjit(parray, mag, tobs)
