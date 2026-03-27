"""RR Lyrae template fitting for pycycle.

Two fitting modes are supported, selected automatically based on the template:

rr-templates mode (template.dust is not None)
    Full physics model: mu (distance modulus), EBV (dust), A (amplitude), phi (phase).
    Model per observation (band b, time t):
        mag = beta_b(P) + mu + EBV * dust_b + A * gamma_b(freq*t + phi)
    where beta_b(P) = c0_b + p1_b*P + p2_b*P^2, P = 1/freq is the period in days,
    and freq = 1/P is in cycles/day.

Multiband mode (template.dust is None)
    Simplified model: per-band offset mu_b, shared amplitude A, phase phi.
    Model per observation:
        mag = mu_b + A * gamma_b(freq*t + phi)

In both modes the inner loop is accelerated by a Cython extension when available,
falling back to a pure-NumPy implementation otherwise.
"""
from __future__ import annotations

import numpy as np

from .templates import RRTemplate

try:
    from ._ext.template_fit_c import rss_grid_rr, rss_grid_mb
    _USE_C = True
except ImportError:
    _USE_C = False


# ---------------------------------------------------------------------------
# Period grid (mirrors periodogram.py)
# ---------------------------------------------------------------------------

def _make_period_grid(hjd, pmin, dphi, pmax=None):
    tspan = np.max(hjd) - np.min(hjd)
    maxfreq = 1.0 / pmin
    minfreq = 2.0 / tspan
    deltafreq = dphi / tspan
    nfreq = max(1, int((maxfreq - minfreq) / deltafreq))
    farray = minfreq + np.arange(nfreq) * deltafreq
    periods = 1.0 / farray
    if pmax is not None:
        periods = periods[periods <= pmax]
    return periods


# ---------------------------------------------------------------------------
# Pure-Python fallback inner loops
# ---------------------------------------------------------------------------

def _interp_template(gamma, bidx, phases):
    """Vectorised linear interpolation on the circular phase grid."""
    n_ph = gamma.shape[1]
    ph = np.asarray(phases) % 1.0
    lo = (ph * n_ph).astype(int) % n_ph
    hi = (lo + 1) % n_ph
    frac = ph * n_ph - lo
    return (1.0 - frac) * gamma[bidx, lo] + frac * gamma[bidx, hi]


def _rss_grid_rr_py(t, mag, w, bidx, gamma, dgamma, dust, betas, freqs, n_newton, n_start):
    N_freq = len(freqs)
    rss_out = np.full(N_freq, np.inf)
    phi_out = np.zeros(N_freq)
    coeffs_out = np.zeros((N_freq, 3))

    for k, freq in enumerate(freqs):
        period = 1.0 / freq
        for s in range(n_start):
            phi = s / n_start
            mu = ebv = A = 0.0

            for _ in range(n_newton):
                beta_corr = betas[bidx, 0] + betas[bidx, 1] * period + betas[bidx, 2] * period * period
                y = mag - beta_corr
                ph = (freq * t + phi) % 1.0
                g = _interp_template(gamma, bidx, ph)
                d = dust[bidx]
                X = np.column_stack([np.ones(len(t)), d, g])
                XtW = (X * w[:, None]).T
                try:
                    params = np.linalg.solve(XtW @ X, XtW @ y)
                except np.linalg.LinAlgError:
                    break
                mu, ebv, A = params
                dg = _interp_template(dgamma, bidx, ph)
                res = y - mu - ebv * d - A * g
                numer = np.sum(w * res * A * dg)
                denom = np.sum(w * (A * dg) ** 2)
                if abs(denom) > 1e-20:
                    phi = (phi - numer / denom) % 1.0

            beta_corr = betas[bidx, 0] + betas[bidx, 1] * period + betas[bidx, 2] * period * period
            y = mag - beta_corr
            ph = (freq * t + phi) % 1.0
            g = _interp_template(gamma, bidx, ph)
            res = y - mu - ebv * dust[bidx] - A * g
            rss = np.sum(w * res ** 2)
            if rss < rss_out[k]:
                rss_out[k] = rss
                phi_out[k] = phi
                coeffs_out[k] = [mu, ebv, A]

    return rss_out, phi_out, coeffs_out


def _rss_grid_mb_py(t, mag, w, bidx, gamma, dgamma, freqs, n_bands, n_newton, n_start):
    N_freq = len(freqs)
    rss_out = np.full(N_freq, np.inf)
    phi_out = np.zeros(N_freq)
    mu_out = np.zeros((N_freq, n_bands))
    A_out = np.zeros(N_freq)

    for k, freq in enumerate(freqs):
        for s in range(n_start):
            phi = s / n_start
            A = 1.0
            mu_b = np.zeros(n_bands)

            for _ in range(n_newton):
                ph = (freq * t + phi) % 1.0
                g = _interp_template(gamma, bidx, ph)
                for b in range(n_bands):
                    mask = bidx == b
                    if mask.any():
                        mu_b[b] = np.sum(w[mask] * (mag[mask] - A * g[mask])) / np.sum(w[mask])
                resid = mag - mu_b[bidx]
                A_num = np.sum(w * resid * g)
                A_den = np.sum(w * g * g)
                if A_den > 1e-20:
                    A = A_num / A_den
                dg = _interp_template(dgamma, bidx, ph)
                res = mag - mu_b[bidx] - A * g
                numer = np.sum(w * res * A * dg)
                denom = np.sum(w * (A * dg) ** 2)
                if abs(denom) > 1e-20:
                    phi = (phi - numer / denom) % 1.0

            ph = (freq * t + phi) % 1.0
            g = _interp_template(gamma, bidx, ph)
            res = mag - mu_b[bidx] - A * g
            rss = np.sum(w * res ** 2)
            if rss < rss_out[k]:
                rss_out[k] = rss
                phi_out[k] = phi
                mu_out[k] = mu_b
                A_out[k] = A

    return rss_out, phi_out, mu_out, A_out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class TemplateFitResult:
    """Result of a :class:`TemplateFitter` run.

    Attributes
    ----------
    periods : ndarray
        Test periods [days].
    rss : ndarray
        Weighted residual sum of squares per period (lower = better).
    best_period : float
        Period at the RSS minimum.
    best_phi : float
        Phase [0, 1) at the best period.
    best_coeffs : dict
        Fitted parameters at the best period.
        rr-templates mode: ``{'mu', 'EBV', 'A'}``.
        Multiband mode: ``{'mu_<band>', ..., 'A'}``.
    template : RRTemplate
        Template used for fitting.
    """

    def __init__(self, periods, rss, phi, coeffs_raw, template,
                 hjd, mag, magerr, filts, filtnams):
        self.periods = periods
        self.rss = rss
        self._phi = phi
        self._coeffs_raw = coeffs_raw  # (N_freq, ...) from fitter
        self.template = template
        self._hjd = hjd
        self._mag = mag
        self._magerr = magerr
        self._filts = filts
        self.filtnams = filtnams

        best_k = int(np.argmin(rss))
        self.best_period = float(periods[best_k])
        self.best_phi = float(phi[best_k])

        if template.dust is not None:
            mu, ebv, A = coeffs_raw[best_k]
            self.best_coeffs = {'mu': mu, 'EBV': ebv, 'A': A}
        else:
            # coeffs_raw is (mu_out, A_out)
            mu_arr, A_arr = coeffs_raw
            self.best_coeffs = {f'mu_{b}': float(mu_arr[best_k, i])
                                for i, b in enumerate(template.bands)}
            self.best_coeffs['A'] = float(A_arr[best_k])

    @property
    def best_freq(self):
        return 1.0 / self.best_period

    def predict(self, hjd, filts, filtnams):
        """Predict magnitudes at given times and bands.

        Parameters
        ----------
        hjd : array-like
        filts : array-like of int  (filter codes, matched to filtnams)
        filtnams : list of str

        Returns
        -------
        mag_pred : ndarray
        """
        hjd = np.asarray(hjd)
        filts = np.asarray(filts, dtype=int)
        freq = self.best_freq
        phi = self.best_phi
        template = self.template

        bidx = np.array([template.band_index(filtnams[int(f)]) for f in filts])
        ph = (freq * hjd + phi) % 1.0
        g = _interp_template(template.gamma, bidx, ph)

        if template.dust is not None:
            mu = self.best_coeffs['mu']
            ebv = self.best_coeffs['EBV']
            A = self.best_coeffs['A']
            period = 1.0 / freq
            beta = (template.betas[bidx, 0]
                    + template.betas[bidx, 1] * period
                    + template.betas[bidx, 2] * period ** 2)
            return beta + mu + ebv * template.dust[bidx] + A * g
        else:
            mu_arr = np.array([self.best_coeffs[f'mu_{b}'] for b in template.bands])
            A = self.best_coeffs['A']
            return mu_arr[bidx] + A * g

    def top_periods(self, n=10):
        """Return the *n* periods with lowest RSS as a dict-list."""
        idx = np.argsort(self.rss)[:n]
        return [{'rank': i + 1, 'period': float(self.periods[idx[i]]),
                 'rss': float(self.rss[idx[i]]),
                 'phi': float(self._phi[idx[i]])} for i in range(n)]

    def plot_rss(self, ax=None):
        """Plot RSS vs period."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3))
        ax.plot(self.periods, self.rss, color='dodgerblue', lw=0.7)
        ax.axvline(self.best_period, color='k', lw=1, ls='--',
                   label=f'best P = {self.best_period:.6f} d')
        ax.set_xlabel('Period [days]')
        ax.set_ylabel('RSS')
        ax.set_title(f'Template fit — {self.template.name}')
        ax.legend()
        return ax

    def plot_phased(self, period=None, ax=None):
        """Plot phased light curve with the best-fit template overlaid."""
        import matplotlib.pyplot as plt
        if period is None:
            period = self.best_period
        phi = self.best_phi
        freq = 1.0 / period
        template = self.template

        if ax is None:
            n = len(set(self.filtnams))
            fig, axes = plt.subplots(n, 1, sharex=True,
                                     figsize=(8, 2.5 * n), squeeze=False)
            axes = axes[:, 0]
        else:
            axes = [ax]

        colors = ['steelblue', 'seagreen', 'tomato', 'goldenrod', 'orchid']
        ph_dense = np.linspace(0.0, 1.0, 400, endpoint=False)

        for i, (fname, ax_) in enumerate(zip(self.filtnams, axes)):
            mask = self._filts == float(i)
            if not mask.any():
                continue
            ph_obs = (freq * self._hjd[mask] + phi) % 1.0
            c = colors[i % len(colors)]

            # data
            ax_.errorbar(ph_obs, self._mag[mask], yerr=self._magerr[mask],
                         fmt='o', ms=4, alpha=0.6, color=c, label=fname)
            ax_.errorbar(ph_obs + 1, self._mag[mask], yerr=self._magerr[mask],
                         fmt='o', ms=4, alpha=0.6, color=c)

            # template curve
            try:
                bi = template.band_index(fname)
            except ValueError:
                ax_.legend(); continue

            g_dense = _interp_template(template.gamma,
                                       np.full(400, bi, dtype=int), ph_dense)
            if template.dust is not None:
                mu = self.best_coeffs['mu']
                ebv = self.best_coeffs['EBV']
                A = self.best_coeffs['A']
                period = 1.0 / freq
                beta = (template.betas[bi, 0]
                        + template.betas[bi, 1] * period
                        + template.betas[bi, 2] * period ** 2)
                m_pred = beta + mu + ebv * template.dust[bi] + A * g_dense
            else:
                mu_b = self.best_coeffs.get(f'mu_{fname}', 0.0)
                A = self.best_coeffs['A']
                m_pred = mu_b + A * g_dense

            ax_.plot(ph_dense, m_pred, color='k', lw=1.2, zorder=5)
            ax_.plot(ph_dense + 1, m_pred, color='k', lw=1.2, zorder=5)
            ax_.invert_yaxis()
            ax_.set_xlim(-0.05, 2.05)
            ax_.set_ylabel('mag')
            ax_.legend(loc='upper right')

        axes[-1].set_xlabel(r'Phase $\phi$')
        plt.suptitle(f'P = {period:.6f} d  |  template: {template.name}',
                     y=1.01)
        plt.tight_layout()
        return axes


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

class TemplateFitter:
    """Fit an RR Lyrae template to a multiband light curve over a period grid.

    Parameters
    ----------
    template : RRTemplate
        Template loaded via :func:`~pycycle.templates.load_rr_template` or
        :func:`~pycycle.templates.load_multiband_templates`.
    n_newton : int
        Number of Newton iterations per frequency (default 5).
    use_errors : bool
        If True (default), weight observations by 1/sigma^2.
    """

    def __init__(self, template: RRTemplate, n_newton: int = 5, n_start: int = 4,
                 use_errors: bool = True):
        self.template = template
        self.n_newton = n_newton
        self.n_start = n_start
        self.use_errors = use_errors
        self._backend = 'C/Cython' if _USE_C else 'pure Python'

    def fit(self, hjd, mag, magerr, filts, filtnams,
            pmin: float = 0.2, dphi: float = 0.02,
            pmax: float = None, periods=None) -> TemplateFitResult:
        """Run the template fit over a period grid.

        Parameters
        ----------
        hjd : array-like, shape (N,)
        mag : array-like, shape (N,)
        magerr : array-like, shape (N,)
        filts : array-like of int, shape (N,)
            Integer filter codes matched to *filtnams*.
        filtnams : list of str
        pmin : float
            Minimum period [days].
        dphi : float
            Phase step controlling grid density.
        pmax : float, optional
        periods : array-like, optional
            Explicit period grid; overrides auto-generation.

        Returns
        -------
        TemplateFitResult
        """
        hjd = np.ascontiguousarray(hjd, dtype=np.float64)
        mag = np.ascontiguousarray(mag, dtype=np.float64)
        magerr = np.ascontiguousarray(magerr, dtype=np.float64)
        filts = np.asarray(filts, dtype=int)

        template = self.template

        # map filter codes → template band indices
        bidx = np.empty(len(hjd), dtype=np.int64)
        for obs_i, fcode in enumerate(filts):
            fname = filtnams[int(fcode)]
            bidx[obs_i] = template.band_index(fname)

        # weights
        if self.use_errors:
            sigma = np.where(magerr > 0, magerr, np.median(magerr[magerr > 0]))
            w = 1.0 / sigma ** 2
        else:
            w = np.ones(len(hjd))
        w = np.ascontiguousarray(w, dtype=np.float64)

        # period / omega grid
        if periods is not None:
            ptest = np.ascontiguousarray(periods, dtype=np.float64)
        else:
            ptest = _make_period_grid(hjd, pmin, dphi, pmax)
        print(f'TemplateFitter: backend = {self._backend}')
        print(f'TemplateFitter: template = {template.name}')
        print(f'TemplateFitter: {len(ptest)} test periods, '
              f'{self.n_newton} Newton iters × {self.n_start} starts')

        gamma = np.ascontiguousarray(template.gamma, dtype=np.float64)
        dg = np.ascontiguousarray(template.dgamma(), dtype=np.float64)

        # NOTE: freqs are in cycles/day (= 1/P), matching the template phase grid [0,1)
        freqs = np.ascontiguousarray(1.0 / ptest, dtype=np.float64)

        if template.dust is not None:
            # rr-templates mode
            dust = np.ascontiguousarray(template.dust, dtype=np.float64)
            betas = np.ascontiguousarray(template.betas, dtype=np.float64)
            if _USE_C:
                rss, phi, coeffs = rss_grid_rr(
                    hjd, mag, w, bidx, gamma, dg, dust, betas, freqs,
                    self.n_newton, self.n_start)
            else:
                rss, phi, coeffs = _rss_grid_rr_py(
                    hjd, mag, w, bidx, gamma, dg, dust, betas, freqs,
                    self.n_newton, self.n_start)
            coeffs_raw = coeffs
        else:
            # Multiband mode
            n_bands = template.n_bands
            if _USE_C:
                rss, phi, mu_out, A_out = rss_grid_mb(
                    hjd, mag, w, bidx, gamma, dg, freqs, n_bands,
                    self.n_newton, self.n_start)
            else:
                rss, phi, mu_out, A_out = _rss_grid_mb_py(
                    hjd, mag, w, bidx, gamma, dg, freqs, n_bands,
                    self.n_newton, self.n_start)
            coeffs_raw = (mu_out, A_out)

        return TemplateFitResult(
            periods=ptest, rss=rss, phi=phi, coeffs_raw=coeffs_raw,
            template=template,
            hjd=hjd, mag=mag, magerr=magerr, filts=filts, filtnams=filtnams,
        )
