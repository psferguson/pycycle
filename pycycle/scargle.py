"""Lomb-Scargle periodogram implementations.

The fast C/Cython implementation is used when available; otherwise falls back
to a numba-JIT-compiled pure-Python version.
"""

import numpy as np

try:
    from pycycle._ext._pycycle_c import scargle_fast as _scargle_fast_c
    _USE_C = True
except ImportError:
    _USE_C = False

try:
    import numba
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False


def _scargle_fast_py(t, c, omega, nfreq):
    """Pure-Python Lomb-Scargle periodogram (Press & Rybicki 1989)."""
    noise = np.sqrt(np.var(c))
    time = t - t[0]
    n0 = len(time)
    om = omega

    s2 = np.zeros(nfreq)
    c2 = np.zeros(nfreq)
    two_time = 2.0 * time
    for i in range(nfreq):
        s2[i] = np.sum(np.sin(two_time * om[i]))
        c2[i] = np.sum(np.cos(two_time * om[i]))

    omtau = np.arctan(s2 / c2) / 2.0
    cosomtau = np.cos(omtau)
    sinomtau = np.sin(omtau)
    tmp = c2 * np.cos(2.0 * omtau) + s2 * np.sin(2.0 * omtau)
    tc2 = 0.5 * (n0 + tmp)
    ts2 = 0.5 * (n0 - tmp)

    cn = c - np.mean(c)
    sh = np.zeros(nfreq)
    ch = np.zeros(nfreq)
    for i in range(nfreq):
        omi_time = om[i] * time
        sh[i] = np.sum(cn * np.sin(omi_time))
        ch[i] = np.sum(cn * np.cos(omi_time))

    px = ((ch * cosomtau + sh * sinomtau) ** 2 / tc2) + \
         ((sh * cosomtau - ch * sinomtau) ** 2 / ts2)
    px = 0.5 * px / (noise ** 2)
    return px


if _USE_NUMBA:
    _scargle_fast_pyjit = numba.jit(nopython=True)(_scargle_fast_py)
else:
    _scargle_fast_pyjit = _scargle_fast_py


def scargle_fast(t, c, omega, nfreq):
    """Compute the Lomb-Scargle periodogram.

    Uses the compiled C extension when available, otherwise falls back to a
    numba-JIT (or plain Python) implementation.

    Parameters
    ----------
    t : ndarray of float64, shape (N,)
        Observation times (e.g. HJD).
    c : ndarray of float64, shape (N,)
        Observed values (magnitudes, centred or uncentred).
    omega : ndarray of float64, shape (nfreq,)
        Angular frequencies (2*pi * frequency) at which to evaluate the PSD.
    nfreq : int
        Number of frequencies (must equal ``omega.size``).

    Returns
    -------
    px : ndarray of float64, shape (nfreq,)
        Lomb-Scargle power at each angular frequency.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    omega = np.ascontiguousarray(omega, dtype=np.float64)
    if _USE_C:
        return _scargle_fast_c(t, c, omega, nfreq)
    return _scargle_fast_pyjit(t, c, omega, nfreq)
