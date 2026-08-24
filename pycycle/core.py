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


#: Combination strategies understood by :func:`combine_periodograms` and
#: the ``combine=`` keyword of :meth:`PeriodSearch.run`.
COMBINE_STRATEGIES = ('sum', 'ranksum', 'normsum')


def combine_periodograms(psi_per_band, method='sum'):
    """Combine per-band PSI periodograms into a single across-band score.

    Background
    ----------
    PSI ≈ (N/2)(A/σ)² summed over the epochs of one band, so its *scale* is
    set by that band's epoch count, photometric noise, and amplitude -- not
    by how well it constrains the period.  Summing raw PSI across bands
    (``method='sum'``) therefore lets whichever band happens to have the
    largest scale dominate the pick, even when a different band's periodogram
    peaks squarely on the true period.

    This is reproducible, not hypothetical: a synthetic RRab mock with r and
    z sharing a ~2-day cadence (common in filter-rotating surveys -- a
    0.5 cycle/day sampling frequency) aliases their periodograms to the same
    wrong period, while g and i (sampled at continuous random epochs)
    independently recover the truth.  For an injected P = 0.6231 d this
    aliases to P ≈ 0.4750 d, matching the failure period quoted in
    ``docs/TODO.md`` item 1 (0.4753 d) to <0.1%.  In that specific draw:
    g and i individually recovered P = 0.62310 d (max PSI 66.9 and 93.2),
    but r and z both peaked at P ≈ 0.4751 d with far larger raw PSI (1636.7
    and 1094.2) purely from epoch count and cadence, not from evidence, and
    ``'sum'`` followed them.  ``'ranksum'`` and ``'normsum'`` both recovered
    P = 0.62310 d on that same draw.  See ``psearch_combine_eval.py``
    (measurement script referenced in the task record; reproduces this
    exact case as its "worked-example reproduction").

    Measured over 300 such mocks (period drawn uniformly in [0.44, 0.90] d),
    this shared-cadence-aliasing regime gives: ``'sum'`` 99.0% recovery
    (|P_fit-P_true|/P_true < 1%), ``'ranksum'`` 100.0%, ``'normsum'`` 100.0%.
    The effect is real but *far smaller* in this isolated synthetic test
    than the 23% -> 37% aggregate improvement quoted in ``docs/TODO.md``;
    the real DP2 catalogue evidently hits this failure mode (or others like
    it) more often than one specific alias geometry does in isolation.

    Rank/scale normalisation is not free, however: a second synthetic
    regime (one band with far more epochs but heavily damped amplitude and
    higher noise, still correctly peaking at the true period, just with a
    smaller peak) shows the opposite ordering -- ``'sum'`` 100.0% recovery,
    ``'ranksum'`` 93.3%, ``'normsum'`` 100.0%.  Converting to ranks discards
    the magnitude information that would otherwise let a low-S/N band's
    correct-period vote be down-weighted relative to a high-S/N band's, and
    that costs ``'ranksum'`` accuracy when every band is actually right.
    ``'normsum'`` did at least as well as ``'sum'`` in both regimes tested.
    This is why ``combine='sum'`` remains the default (see
    :meth:`PeriodSearch.run`) rather than switching to ``'ranksum'``: the
    measurement does not show it uniformly better, only better in a
    specific (real, but not universal) failure mode.

    Parameters
    ----------
    psi_per_band : ndarray of float64, shape (n_bands, n_periods) or (n_periods,)
        Per-band PSI periodograms, one row per band, on a shared period
        grid.  A 1-D input (single band) is returned unchanged (a copy) --
        there is nothing to combine.
    method : {'sum', 'ranksum', 'normsum'}, optional
        - ``'sum'`` (default): add the raw PSI values.  Matches pycycle's
          original behaviour; kept as the library default so that existing
          callers (e.g. ``pycycle.dp2.fit_lightcurve``) see identical
          ``best_period``/``psi_m`` semantics unless they opt in.  See
          "Background" above for why this is *not* recommended when bands
          differ substantially in epoch count, noise, or amplitude.
        - ``'ranksum'``: replace each band's PSI values with their rank
          (1 = worst period, n_periods = best) along the period axis, then
          sum the ranks across bands.  No single band can dominate by raw
          scale; a band only wins by consistently favouring a period more
          than the others do.
        - ``'normsum'``: divide each band's PSI by that band's own peak
          value (so each band's best period contributes exactly 1.0), then
          sum.  Cheaper than ``'ranksum'`` (no sort) and preserves each
          band's relative peak shape, unlike a full z-score, but a band
          that is pure noise still contributes an artificial peak of
          height 1.0 -- ``'ranksum'`` is more robust to that failure mode.

    Returns
    -------
    psi_combined : ndarray of float64, shape (n_periods,)
    """
    psi_per_band = np.asarray(psi_per_band, dtype=np.float64)
    if psi_per_band.ndim == 1:
        return psi_per_band.copy()

    if method == 'sum':
        return psi_per_band.sum(axis=0)
    elif method == 'ranksum':
        ranks = np.empty_like(psi_per_band)
        for i, row in enumerate(psi_per_band):
            order = np.argsort(row, kind='stable')
            ranks[i, order] = np.arange(1, row.shape[0] + 1, dtype=np.float64)
        return ranks.sum(axis=0)
    elif method == 'normsum':
        peak = psi_per_band.max(axis=1, keepdims=True)
        peak = np.where(peak > 0, peak, 1.0)  # guard an all-zero (or all-negative) band
        return (psi_per_band / peak).sum(axis=0)
    else:
        raise ValueError(
            "Unknown combine method %r; choose one of %s" % (method, COMBINE_STRATEGIES)
        )


class PeriodSearchResult:
    """Container for the output of a :class:`PeriodSearch` run.

    Attributes
    ----------
    ptest : ndarray of float64, shape (N,)
        Test periods [days] — the same grid for all filters.
    psi_m : ndarray of float64
        PSI periodogram.  Shape ``(M, N)`` for *M > 1* filters, or ``(N,)``
        for a single filter.  Unchanged from earlier pycycle versions --
        kept exactly as-is for backward compatibility.  New code should
        generally prefer :attr:`psi_per_band`, which has a stable
        ``(n_bands, n_periods)`` shape regardless of *M*.
    thresh_m : ndarray of float64
        Significance threshold; same shape as *psi_m*.
    filtnams : list of str
        Filter names associated with each row of *psi_m*.
    psi_per_band : ndarray of float64, shape (n_bands, n_periods)
        Per-band PSI periodograms, always 2-D (even for a single band), so
        callers can combine them deliberately instead of relying on the
        library's default.  Row order matches :attr:`bands`.
    bands : list of str
        Filter names for each row of :attr:`psi_per_band` (identical
        content to *filtnams*; provided under this name to pair naturally
        with *psi_per_band*).
    combine : {'sum', 'ranksum', 'normsum'}
        The strategy used by :attr:`best_period`, :attr:`psi_combined`, and
        ``top_periods(filter_idx=None)`` to reduce :attr:`psi_per_band`
        across bands.  See :func:`combine_periodograms`.
    n_epochs_used : int
        Total number of observations (summed over the searched bands) that
        passed the periodogram's quality cut and actually contributed to
        the PSI computation.  PSI is linear in epoch count at fixed
        sampling, so this is what :attr:`psi_per_epoch` divides by.
    """

    def __init__(self, ptest, psi_m, thresh_m, hjd, mag, magerr, filts, filtnams,
                 psi_per_band=None, bands=None, combine='sum', n_epochs_used=None):
        self.ptest = ptest
        self.psi_m = psi_m
        self.thresh_m = thresh_m
        self._hjd = hjd
        self._mag = mag
        self._magerr = magerr
        self._filts = filts
        self.filtnams = filtnams

        # psi_per_band is always 2-D, unlike the backward-compatible psi_m
        # (which collapses to 1-D for a single filter).
        if psi_per_band is not None:
            self.psi_per_band = np.atleast_2d(psi_per_band)
        else:
            self.psi_per_band = np.atleast_2d(psi_m)
        self.bands = list(bands) if bands is not None else list(filtnams)
        if combine not in COMBINE_STRATEGIES:
            raise ValueError(
                "Unknown combine method %r; choose one of %s" % (combine, COMBINE_STRATEGIES)
            )
        self.combine = combine
        self.n_epochs_used = int(n_epochs_used) if n_epochs_used is not None else len(hjd)

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------

    @property
    def freq(self):
        """Frequencies [days⁻¹] corresponding to :attr:`ptest`."""
        return 1.0 / self.ptest

    @property
    def psi_combined(self):
        """Across-band PSI, reduced from :attr:`psi_per_band` via :attr:`combine`.

        For ``combine='sum'`` (the default) this is numerically identical to
        the historical ``psi_m if psi_m.ndim == 1 else psi_m.sum(0)``
        expression used throughout pycycle and its callers.
        """
        return combine_periodograms(self.psi_per_band, method=self.combine)

    @property
    def best_period(self):
        """Period [days] with the highest combined PSI across all filters.

        Uses :attr:`combine` to reduce across bands (default ``'sum'``, the
        original behaviour -- see :func:`combine_periodograms` for why a
        rank- or scale-normalised combination is often a better *pick* even
        though the raw-PSI search itself finds the right period).
        """
        return self.ptest[np.argmax(self.psi_combined)]

    @property
    def psi_per_epoch(self):
        """Peak combined PSI divided by :attr:`n_epochs_used`.

        PSI ≈ (N/2)(A/σ)² is linear in epoch count at fixed sampling, so raw
        PSI is a good discriminator *within* one light curve (fixed N) but a
        poor one *across a catalogue* of objects with different epoch
        counts -- it actively promotes well-sampled non-variables over
        poorly-sampled real variables.  In a worked M49 comparison the
        visually-good objects had a median 336 epochs against 527 for the
        rejects, i.e. raw PSI was systematically biased toward the rejects;
        dividing by N moved a known RR Lyrae from rank 162 to rank 34 (see
        ``docs/TODO.md`` item 2).  This attribute exposes that
        normalisation directly so callers don't have to recompute it.
        """
        if not self.n_epochs_used:
            return np.nan
        return float(np.max(self.psi_combined)) / self.n_epochs_used

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
            Index into *filtnams* (equivalently, :attr:`bands`) selecting a
            single filter.  When ``None`` (default) the per-band PSI values
            are combined across all filters via :attr:`combine`.
        write : bool
            Write the table to *filename* as CSV.
        filename : str
            Output CSV path.

        Returns
        -------
        astropy.table.Table
        """
        if filter_idx is not None:
            psi = self.psi_per_band[filter_idx]
            thresh = self.thresh_m[filter_idx] if self.thresh_m.ndim > 1 else self.thresh_m
        else:
            psi = self.psi_combined
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
    filts : array-like of str, shape (N,)
        Filter name (band) per observation, co-aligned with *hjd*.
    filtnams : list of str, optional
        Bands to process, in the desired display/output order.  Defaults to
        the sorted unique values of *filts*.  Use this to restrict the search
        to a subset of bands or to fix a particular plot ordering.

    Examples
    --------
    >>> import numpy as np
    >>> from pycycle import PeriodSearch
    >>> hjd = np.loadtxt('B1392all.tab', usecols=0)
    >>> # ... load mag, magerr, filts (strings) ...
    >>> ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=['B', 'V'])
    >>> result = ps.run(pmin=0.2, dphi=0.02)
    >>> print(result.best_period)
    """

    def __init__(self, hjd, mag, magerr, filts, filtnams=None):
        self.hjd = np.asarray(hjd, dtype=np.float64)
        self.mag = np.asarray(mag, dtype=np.float64)
        self.magerr = np.asarray(magerr, dtype=np.float64)
        self.filts = np.asarray(filts).astype(str)
        if filtnams is None:
            self.filtnams = sorted(set(self.filts.tolist()))
        else:
            self.filtnams = list(filtnams)

        assert self.hjd.ndim == 1
        assert self.mag.shape == self.hjd.shape
        assert self.magerr.shape == self.hjd.shape
        assert self.filts.shape == self.hjd.shape

    def run(self, pmin, dphi, n_thresh=1, pmax=None, periods=None, verbose=False,
            combine='sum'):
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
        combine : {'sum', 'ranksum', 'normsum'}, optional
            How :attr:`PeriodSearchResult.best_period` and the default
            ``top_periods()`` reduce per-band PSI to a single across-band
            score; irrelevant for a single-filter search.  Default
            ``'sum'`` reproduces pycycle's original behaviour bit-for-bit
            and is kept as the default for backward compatibility with
            callers (e.g. ``pycycle.dp2.fit_lightcurve``) that read
            ``best_period``/``psi_m`` directly, *and* because a synthetic-mock
            comparison (``psearch_combine_eval.py``) did not show
            ``'ranksum'`` uniformly better: it wins when two bands share a
            sampling-driven alias that a raw sum lets dominate (99.0% ->
            100.0% recovery in that regime), but loses when a genuinely
            low-S/N band still peaks at the correct period, because
            converting to ranks throws away the magnitude evidence that
            would otherwise down-weight it (100.0% -> 93.3% in that
            regime).  ``'normsum'`` matched or beat ``'sum'`` in both
            regimes tested and is a reasonable alternative to try.  See
            :func:`combine_periodograms` for the full writeup and numbers.
            Pass the per-band periodograms
            (:attr:`PeriodSearchResult.psi_per_band`) to
            :func:`combine_periodograms` directly if you want to compare
            strategies after the fact without rerunning the search.

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
        n_epochs_used = 0

        for i, filtnam in enumerate(self.filtnams):
            if verbose:
                print('\nPeriodSearch: filter %s' % filtnam)
            x, fy, theta, psi, conf, nok = compute_periodogram(
                self.hjd, self.mag, self.magerr, self.filts,
                fwant=filtnam, pmin=pmin, dphi=dphi,
                n_thresh=n_thresh, pmax=pmax, periods=periods,
                verbose=verbose,
            )
            if i == 0:
                ptest = x
                psi_m = np.zeros((nfilts, len(x)))
                thresh_m = np.zeros((nfilts, len(x)))
            psi_m[i, :] = psi
            thresh_m[i, :] = conf
            n_epochs_used += nok

        # psi_per_band/thresh_per_band keep the full (nfilts, npoints) shape
        # regardless of nfilts; psi_m/thresh_m below stay bit-identical to
        # the historical API (1-D when nfilts == 1).
        psi_per_band = psi_m.copy()

        if nfilts == 1:
            psi_m = psi_m.flatten()
            thresh_m = thresh_m.flatten()

        return PeriodSearchResult(
            ptest, psi_m, thresh_m,
            self.hjd, self.mag, self.magerr, self.filts, self.filtnams,
            psi_per_band=psi_per_band, bands=list(self.filtnams),
            combine=combine, n_epochs_used=n_epochs_used,
        )
