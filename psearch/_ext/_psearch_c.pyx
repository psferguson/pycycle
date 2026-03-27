# Cython wrapper for the psearch C extensions
# Wraps scargle_fast_c and ctheta_slave_c from psearch_py_c.c

import cython
import numpy as np
cimport numpy as np

# --- Lomb-Scargle ---

cdef extern void scargle_fast_c(
    double* tAD,
    long sz_tL,
    double* cAD,
    long sz_cL,
    double* omegaAD,
    long sz_omegaL,
    long nfreqL,
    double* pxAD,
    long sz_pxL)

@cython.boundscheck(False)
@cython.wraparound(False)
def scargle_fast(
    np.ndarray[np.double_t, ndim=1, mode="c"] tAD not None,
    np.ndarray[np.double_t, ndim=1, mode="c"] cAD not None,
    np.ndarray[np.double_t, ndim=1, mode="c"] omegaAD not None,
    long nfreqL,
):
    """Compute the Lomb-Scargle periodogram via the C implementation.

    Parameters
    ----------
    tAD : array of float64
        Observation times.
    cAD : array of float64
        Normalised magnitudes (centred).
    omegaAD : array of float64
        Angular frequencies (2*pi/period).
    nfreqL : int
        Number of frequencies (must equal omegaAD.size).

    Returns
    -------
    pxAD : array of float64
        Lomb-Scargle power at each frequency.
    """
    cdef long sz_tL = tAD.size
    cdef long sz_cL = cAD.size
    cdef long sz_omegaL = omegaAD.size
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] pxAD = omegaAD.copy(order='K')
    cdef long sz_pxL = pxAD.size
    assert nfreqL == omegaAD.size, "nfreqL must equal omegaAD.size"
    scargle_fast_c(
        &tAD[0], sz_tL,
        &cAD[0], sz_cL,
        &omegaAD[0], sz_omegaL,
        nfreqL,
        &pxAD[0], sz_pxL,
    )
    return pxAD


# --- Lafler-Kinman ---

cdef extern void ctheta_slave_c(
    double* parrayAD,
    long sz_parrayL,
    double* magAD,
    long sz_magL,
    double* tobsAD,
    long sz_tobsL,
    double* thetaAD,
    long sz_thetaL)

@cython.boundscheck(False)
@cython.wraparound(False)
def ctheta_slave(
    np.ndarray[np.double_t, ndim=1, mode="c"] parrayAD not None,
    np.ndarray[np.double_t, ndim=1, mode="c"] magAD not None,
    np.ndarray[np.double_t, ndim=1, mode="c"] tobsAD not None,
):
    """Compute the Lafler-Kinman theta statistic via the C implementation.

    Parameters
    ----------
    parrayAD : array of float64
        Test periods.
    magAD : array of float64
        Observed magnitudes.
    tobsAD : array of float64
        Observation times.

    Returns
    -------
    thetaAD : array of float64
        Theta (phase-dispersion) values at each test period.
    """
    cdef long sz_parrayL = parrayAD.size
    cdef long sz_magL = magAD.size
    cdef long sz_tobsL = tobsAD.size
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] thetaAD = parrayAD.copy(order='K')
    cdef long sz_thetaL = thetaAD.size
    ctheta_slave_c(
        &parrayAD[0], sz_parrayL,
        &magAD[0], sz_magL,
        &tobsAD[0], sz_tobsL,
        &thetaAD[0], sz_thetaL,
    )
    return thetaAD
