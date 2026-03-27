# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython inner loops for RR Lyrae template fitting.

Two public functions:

rss_grid_rr  — rr-templates model (Long / Stringer et al.)
               Params: mu (distance modulus), EBV (dust), A (amplitude)
               Requires dust coefficients and beta corrections.

rss_grid_mb  — Multiband model (Baeza-Villagra et al.)
               Params: per-band offset mu_b + shared amplitude A
               No dust or beta corrections.

Both perform a grid search over angular frequencies with Newton iterations
to optimise the nonlinear phase parameter phi.
"""

import numpy as np
cimport numpy as np
from libc.math cimport fmod, fabs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cdef inline double _interp(double[:, :] arr, int b, double ph, int n_ph) noexcept nogil:
    """Linear interpolation on a circular phase grid [0, 1)."""
    cdef long lo = <long>(ph * n_ph) % n_ph
    cdef long hi = (lo + 1) % n_ph
    cdef double f = ph * n_ph - lo
    return (1.0 - f) * arr[b, lo] + f * arr[b, hi]


cdef int _solve3(double a00, double a01, double a02,
                  double a11, double a12, double a22,
                  double b0,  double b1,  double b2,
                  double* x0, double* x1, double* x2) noexcept nogil:
    """Solve 3x3 symmetric positive definite system via Cramer's rule.
    Returns 0 on success, 1 if the matrix is singular."""
    cdef double det, inv_det
    det = (a00 * (a11 * a22 - a12 * a12)
         - a01 * (a01 * a22 - a12 * a02)
         + a02 * (a01 * a12 - a11 * a02))
    if fabs(det) < 1e-20:
        return 1
    inv_det = 1.0 / det
    x0[0] = inv_det * (b0 * (a11*a22 - a12*a12)
                     + b1 * (a02*a12 - a01*a22)
                     + b2 * (a01*a12 - a02*a11))
    x1[0] = inv_det * (b0 * (a02*a12 - a01*a22)
                     + b1 * (a00*a22 - a02*a02)
                     + b2 * (a01*a02 - a00*a12))
    x2[0] = inv_det * (b0 * (a01*a12 - a02*a11)
                     + b1 * (a01*a02 - a00*a12)
                     + b2 * (a00*a11 - a01*a01))
    return 0


# ---------------------------------------------------------------------------
# rr-templates mode
# ---------------------------------------------------------------------------

def rss_grid_rr(
    double[:] t,
    double[:] mag,
    double[:] w,
    long[:]   bidx,
    double[:, :] gamma,
    double[:, :] dgamma,
    double[:] dust,
    double[:, :] betas,
    double[:] omegas,
    int n_newton = 5,
):
    """RSS grid search — rr-templates model.

    Model per observation (band b, time t_i):
        mag_i = beta_b(omega) + mu + EBV * dust_b + A * gamma_b(omega*t_i + phi)
    where
        beta_b(omega) = c0_b + p1_b * omega + p2_b * omega^2

    Parameters
    ----------
    t : (N,) observation times
    mag : (N,) observed magnitudes
    w : (N,) weights = 1/sigma^2
    bidx : (N,) band indices into gamma rows
    gamma : (n_bands, n_phase) template
    dgamma : (n_bands, n_phase) template derivative d(gamma)/d(phase)
    dust : (n_bands,) per-band extinction coefficients
    betas : (n_bands, 3) columns are c0, p1, p2
    omegas : (N_freq,) angular frequencies = 2*pi/P
    n_newton : Newton iterations per frequency

    Returns
    -------
    rss : (N_freq,) residual sum of squares
    phi : (N_freq,) best phase per frequency
    coeffs : (N_freq, 3) [mu, EBV, A] at best phase
    """
    cdef int N      = t.shape[0]
    cdef int N_freq = omegas.shape[0]
    cdef int n_ph   = gamma.shape[1]

    rss_out    = np.empty(N_freq, dtype=np.float64)
    phi_out    = np.empty(N_freq, dtype=np.float64)
    coeffs_out = np.zeros((N_freq, 3), dtype=np.float64)

    cdef double[:] rss_v   = rss_out
    cdef double[:] phi_v   = phi_out
    cdef double[:, :] co_v = coeffs_out

    cdef int    i, k, it, err
    cdef long   b
    cdef double omega, phi, mu, ebv, A
    cdef double yi, wi, di, gi, dgi, phase_i
    cdef double X00, X01, X02, X11, X12, X22, Xy0, Xy1, Xy2
    cdef double numer, denom, rss_val, res_i

    for k in range(N_freq):
        omega = omegas[k]
        phi   = 0.0
        mu    = 0.0
        ebv   = 0.0
        A     = 0.0

        for it in range(n_newton):
            X00 = 0.0; X01 = 0.0; X02 = 0.0
            X11 = 0.0; X12 = 0.0; X22 = 0.0
            Xy0 = 0.0; Xy1 = 0.0; Xy2 = 0.0

            for i in range(N):
                b  = bidx[i]
                yi = mag[i] - (betas[b, 0]
                             + betas[b, 1] * omega
                             + betas[b, 2] * omega * omega)
                wi = w[i]
                di = dust[b]
                phase_i = fmod(omega * t[i] + phi, 1.0)
                if phase_i < 0.0:
                    phase_i += 1.0
                gi = _interp(gamma, b, phase_i, n_ph)

                X00 += wi
                X01 += wi * di
                X02 += wi * gi
                X11 += wi * di * di
                X12 += wi * di * gi
                X22 += wi * gi * gi
                Xy0 += wi * yi
                Xy1 += wi * di * yi
                Xy2 += wi * gi * yi

            err = _solve3(X00, X01, X02, X11, X12, X22,
                          Xy0, Xy1, Xy2, &mu, &ebv, &A)
            if err:
                break

            # Newton step for phi
            numer = 0.0
            denom = 0.0
            for i in range(N):
                b  = bidx[i]
                yi = mag[i] - (betas[b, 0]
                             + betas[b, 1] * omega
                             + betas[b, 2] * omega * omega)
                wi = w[i]
                phase_i = fmod(omega * t[i] + phi, 1.0)
                if phase_i < 0.0:
                    phase_i += 1.0
                gi  = _interp(gamma,  b, phase_i, n_ph)
                dgi = _interp(dgamma, b, phase_i, n_ph)
                res_i  = yi - mu - ebv * dust[b] - A * gi
                numer += wi * res_i * (A * dgi)
                denom += wi * (A * dgi) * (A * dgi)

            if fabs(denom) > 1e-20:
                phi -= numer / denom
            phi = fmod(phi, 1.0)
            if phi < 0.0:
                phi += 1.0

        # final RSS
        rss_val = 0.0
        for i in range(N):
            b  = bidx[i]
            yi = mag[i] - (betas[b, 0]
                         + betas[b, 1] * omega
                         + betas[b, 2] * omega * omega)
            phase_i = fmod(omega * t[i] + phi, 1.0)
            if phase_i < 0.0:
                phase_i += 1.0
            gi    = _interp(gamma, b, phase_i, n_ph)
            res_i = yi - mu - ebv * dust[b] - A * gi
            rss_val += w[i] * res_i * res_i

        rss_v[k]   = rss_val
        phi_v[k]   = phi
        co_v[k, 0] = mu
        co_v[k, 1] = ebv
        co_v[k, 2] = A

    return rss_out, phi_out, coeffs_out


# ---------------------------------------------------------------------------
# Multiband mode
# ---------------------------------------------------------------------------

def rss_grid_mb(
    double[:] t,
    double[:] mag,
    double[:] w,
    long[:]   bidx,
    double[:, :] gamma,
    double[:, :] dgamma,
    double[:] omegas,
    int n_bands,
    int n_newton = 5,
):
    """RSS grid search — Multiband model.

    Model per observation (band b, time t_i):
        mag_i = mu_b + A * gamma_b(omega*t_i + phi)
    where mu_b is a per-band offset and A is a shared amplitude.

    Updates alternate between:
      1. mu_b  (closed-form weighted mean per band)
      2. A     (1-D weighted regression)
      3. phi   (Newton step)

    Parameters
    ----------
    t, mag, w, bidx : per-observation arrays (N,)
    gamma, dgamma : (n_bands, n_phase) template and derivative
    omegas : (N_freq,) angular frequencies
    n_bands : number of bands
    n_newton : alternating update iterations per frequency

    Returns
    -------
    rss    : (N_freq,)
    phi    : (N_freq,)
    mu_out : (N_freq, n_bands) per-band offsets at best phi
    A_out  : (N_freq,) shared amplitude at best phi
    """
    cdef int N      = t.shape[0]
    cdef int N_freq = omegas.shape[0]
    cdef int n_ph   = gamma.shape[1]
    cdef int nb     = n_bands

    rss_out = np.empty(N_freq,       dtype=np.float64)
    phi_out = np.empty(N_freq,       dtype=np.float64)
    mu_out  = np.zeros((N_freq, nb), dtype=np.float64)
    A_out   = np.zeros(N_freq,       dtype=np.float64)

    cdef double[:] rss_v    = rss_out
    cdef double[:] phi_v    = phi_out
    cdef double[:, :] mu_v  = mu_out
    cdef double[:] A_v      = A_out

    # per-band accumulators (stack-allocated via typed memoryview on Python array)
    cdef double[:] mu_b   = np.zeros(nb)
    cdef double[:] wsum_b = np.zeros(nb)
    cdef double[:] ysum_b = np.zeros(nb)

    cdef int    i, k, it, bb
    cdef long   b
    cdef double omega, phi, A, gi, dgi, phase_i, wi
    cdef double A_num, A_den, res_i, rss_val, numer, denom

    for k in range(N_freq):
        omega = omegas[k]
        phi   = 0.0
        A     = 1.0
        for bb in range(nb):
            mu_b[bb] = 0.0

        for it in range(n_newton):

            # --- step 1: update mu_b for fixed A, phi ---
            for bb in range(nb):
                wsum_b[bb] = 0.0
                ysum_b[bb] = 0.0
            for i in range(N):
                b  = bidx[i]
                wi = w[i]
                phase_i = fmod(omega * t[i] + phi, 1.0)
                if phase_i < 0.0:
                    phase_i += 1.0
                gi = _interp(gamma, b, phase_i, n_ph)
                wsum_b[b] += wi
                ysum_b[b] += wi * (mag[i] - A * gi)
            for bb in range(nb):
                if wsum_b[bb] > 1e-30:
                    mu_b[bb] = ysum_b[bb] / wsum_b[bb]

            # --- step 2: update A for fixed mu_b, phi ---
            A_num = 0.0
            A_den = 0.0
            for i in range(N):
                b  = bidx[i]
                wi = w[i]
                phase_i = fmod(omega * t[i] + phi, 1.0)
                if phase_i < 0.0:
                    phase_i += 1.0
                gi     = _interp(gamma, b, phase_i, n_ph)
                A_num += wi * (mag[i] - mu_b[b]) * gi
                A_den += wi * gi * gi
            if A_den > 1e-30:
                A = A_num / A_den

            # --- step 3: Newton step for phi ---
            numer = 0.0
            denom = 0.0
            for i in range(N):
                b  = bidx[i]
                wi = w[i]
                phase_i = fmod(omega * t[i] + phi, 1.0)
                if phase_i < 0.0:
                    phase_i += 1.0
                gi    = _interp(gamma,  b, phase_i, n_ph)
                dgi   = _interp(dgamma, b, phase_i, n_ph)
                res_i = mag[i] - mu_b[b] - A * gi
                numer += wi * res_i * (A * dgi)
                denom += wi * (A * dgi) * (A * dgi)
            if fabs(denom) > 1e-20:
                phi -= numer / denom
            phi = fmod(phi, 1.0)
            if phi < 0.0:
                phi += 1.0

        # final RSS
        rss_val = 0.0
        for i in range(N):
            b  = bidx[i]
            phase_i = fmod(omega * t[i] + phi, 1.0)
            if phase_i < 0.0:
                phase_i += 1.0
            gi    = _interp(gamma, b, phase_i, n_ph)
            res_i = mag[i] - mu_b[b] - A * gi
            rss_val += w[i] * res_i * res_i

        rss_v[k] = rss_val
        phi_v[k] = phi
        A_v[k]   = A
        for bb in range(nb):
            mu_v[k, bb] = mu_b[bb]

    return rss_out, phi_out, mu_out, A_out
