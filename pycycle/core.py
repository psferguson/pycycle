"""Core pycycle API: PeriodSearch class and PeriodSearchResult."""

import numpy as np

from .periodogram import compute_periodogram
from .results import results_table
from .plotting import plot_observations, plot_periodogram, plot_phased

try:
    from pycycle._ext._pycycle_c import scargle_fast  # noqa: F401
    _BACKEND = 'C/Cython'
except ImportError:
    _BACKEND = 'pure Python'


class PeriodSearchResult:
    """Container for the output of a :class:`PeriodSearch` run.

    Attributes
    ----------
    ptest : ndarray of float64, shape (N,)
        Test periods [days] — the same grid for all filters.
    psi_m : ndarray of float64
        PSI periodogram.  Shape ``(M, N)`` for *M > 1* filters, or ``(N,)``
        for a single filter.
    thresh_m : ndarray of float64
        Significance threshold; same shape as *psi_m*.
    filtnams : list of str
        Filter names associated with each row of *psi_m*.
    """

    def __init__(self, ptest, psi_m, thresh_m, hjd, mag, magerr, filts, filtnams):
        self.ptest = ptest
        self.psi_m = psi_m
        self.thresh_m = thresh_m
        self._hjd = hjd
        self._mag = mag
        self._magerr = magerr
        self._filts = filts
        self.filtnams = filtnams

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------

    @property
    def freq(self):
        """Frequencies [days⁻¹] corresponding to :attr:`ptest`."""
        return 1.0 / self.ptest

    @property
    def best_period(self):
        """Period [days] with the highest combined PSI across all filters."""
        psi_combined = self.psi_m if self.psi_m.ndim == 1 else self.psi_m.sum(0)
        return self.ptest[np.argmax(psi_combined)]

    # ------------------------------------------------------------------
    # result extraction
    # ------------------------------------------------------------------

    def top_periods(self, n=10, filter_idx=None, write=False, filename='pycycle_results.csv'):
        """Return a table of the top *n* period candidates.

        Parameters
        ----------
        n : int
            Number of candidates.
        filter_idx : int or None
            Index into *filtnams* selecting a single filter.  When ``None``
            (default) the PSI values are summed across all filters.
        write : bool
            Write the table to *filename* as CSV.
        filename : str
            Output CSV path.

        Returns
        -------
        astropy.table.Table
        """
        if filter_idx is not None:
            psi = self.psi_m[filter_idx] if self.psi_m.ndim > 1 else self.psi_m
            thresh = self.thresh_m[filter_idx] if self.thresh_m.ndim > 1 else self.thresh_m
        else:
            psi = self.psi_m if self.psi_m.ndim == 1 else self.psi_m.sum(0)
            thresh = self.thresh_m if self.thresh_m.ndim == 1 else self.thresh_m.sum(0)
        return results_table(self.ptest, psi, thresh, n=n, write=write, filename=filename)

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def plot_observations(self, **kwargs):
        """Plot the raw multi-band light curve.  Passes keyword args to
        :func:`~pycycle.plotting.plot_observations`."""
        plot_observations(self._hjd, self._mag, self._filts, self.filtnams, **kwargs)

    def plot_periodogram(self, **kwargs):
        """Plot the PSI periodogram.  Passes keyword args to
        :func:`~pycycle.plotting.plot_periodogram`."""
        plot_periodogram(self.freq, self.psi_m, self.thresh_m, self.filtnams, **kwargs)

    def plot_phased(self, period=None, **kwargs):
        """Plot the phased light curve.

        Parameters
        ----------
        period : float, optional
            Folding period [days].  Defaults to :attr:`best_period`.
        """
        if period is None:
            period = self.best_period
        plot_phased(self._hjd, self._mag, self._magerr, self._filts,
                    self.filtnams, period=period, **kwargs)


class PeriodSearch:
    """Hybrid Lomb-Scargle / Lafler-Kinman period finder for variable stars.

    Based on Saha & Vivas (2017, AJ 154, 231).

    Parameters
    ----------
    hjd : ndarray of float64, shape (N,)
        Heliocentric Julian Dates of all observations.
    mag : ndarray of float64, shape (N,)
        Magnitudes co-aligned with *hjd*.
    magerr : ndarray of float64, shape (N,)
        Magnitude errors co-aligned with *hjd*.
    filts : array-like, shape (N,)
        Integer filter codes co-aligned with *hjd*.
    filtnams : list of str
        Filter names.  ``filtnams[i]`` corresponds to filter code *i*.

    Examples
    --------
    >>> import numpy as np
    >>> from pycycle import PeriodSearch
    >>> hjd = np.loadtxt('B1392all.tab', usecols=0)
    >>> # ... load mag, magerr, filts ...
    >>> ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=['B', 'V'])
    >>> result = ps.run(pmin=0.2, dphi=0.02)
    >>> print(result.best_period)
    """

    def __init__(self, hjd, mag, magerr, filts, filtnams):
        self.hjd = np.asarray(hjd, dtype=np.float64)
        self.mag = np.asarray(mag, dtype=np.float64)
        self.magerr = np.asarray(magerr, dtype=np.float64)
        self.filts = np.asarray(filts, dtype=np.float64)
        self.filtnams = list(filtnams)

        assert self.hjd.ndim == 1
        assert self.mag.shape == self.hjd.shape
        assert self.magerr.shape == self.hjd.shape
        assert self.filts.shape == self.hjd.shape

    def run(self, pmin, dphi, n_thresh=1, pmax=None, periods=None, verbose=False):
        """Run the period search across all filter bands.

        Parameters
        ----------
        pmin : float
            Minimum period to test [days].
        dphi : float
            Maximum allowed phase change between consecutive test periods.
        n_thresh : int, optional
            Number of Monte Carlo significance runs (default 1; use 0 to skip).
        pmax : float, optional
            Maximum period to test [days].
        periods : ndarray, optional
            Explicit array of test periods; overrides the auto-generated grid.
        verbose : bool, optional
            Print extra diagnostic output.

        Returns
        -------
        PeriodSearchResult
        """
        if verbose:
            print('PeriodSearch: backend = %s' % _BACKEND)
        nfilts = len(self.filtnams)
        psi_m = None
        thresh_m = None
        ptest = None

        for i, filtnam in enumerate(self.filtnams):
            if verbose:
                print('\nPeriodSearch: filter %s' % filtnam)
            x, fy, theta, psi, conf = compute_periodogram(
                self.hjd, self.mag, self.magerr, self.filts,
                fwant=i, pmin=pmin, dphi=dphi,
                n_thresh=n_thresh, pmax=pmax, periods=periods,
                verbose=verbose,
            )
            if i == 0:
                ptest = x
                psi_m = np.zeros((nfilts, len(x)))
                thresh_m = np.zeros((nfilts, len(x)))
            psi_m[i, :] = psi
            thresh_m[i, :] = conf

        if nfilts == 1:
            psi_m = psi_m.flatten()
            thresh_m = thresh_m.flatten()

        return PeriodSearchResult(
            ptest, psi_m, thresh_m,
            self.hjd, self.mag, self.magerr, self.filts, self.filtnams,
        )
