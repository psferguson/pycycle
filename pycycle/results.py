"""Result extraction and formatting for pycycle periodograms."""

import numpy as np
from astropy.table import Table

__all__ = ["results_table"]


def results_table(periods, psi, thresh, n=10, write=False, filename='pycycle_results.csv'):
    """Return a table of the top *n* period candidates ranked by PSI power.

    Local maxima in *psi* are identified and the *n* strongest are returned.

    Parameters
    ----------
    periods : ndarray of float64, shape (N,)
        Test periods [days].
    psi : ndarray of float64, shape (N,)
        PSI periodogram values co-aligned with *periods*.
    thresh : ndarray of float64, shape (N,)
        Significance threshold co-aligned with *periods*.
    n : int
        Number of top candidates to return.
    write : bool, optional
        Write the table to *filename* as CSV (default ``False``).
    filename : str, optional
        Output CSV path (used only when *write* is ``True``).

    Returns
    -------
    tab : astropy.table.Table
        Columns: ``rank``, ``period``, ``period_err``, ``power``,
        ``index``, ``freq``, ``thresh``.
    """
    assert isinstance(periods, np.ndarray) and periods.ndim == 1
    assert psi.shape == periods.shape
    assert thresh.shape == periods.shape
    assert isinstance(n, int) and n >= 1

    sz = len(periods)
    assert sz >= 3

    # find local maxima
    lm_x, lm_y, lm_k = [], [], []
    for k in range(1, sz - 1):
        if psi[k] > psi[k - 1] and psi[k] > psi[k + 1]:
            lm_x.append(periods[k])
            lm_y.append(psi[k])
            lm_k.append(k)

    lm_x = np.array(lm_x)
    lm_y = np.array(lm_y)
    lm_k = np.array(lm_k, dtype=int)
    assert len(lm_y) >= n, (
        "Fewer than %d local maxima found (%d). Reduce n." % (n, len(lm_y))
    )

    idx = (-lm_y).argsort()[:n]

    rank, period, period_err, power, index, frequency, threshold = \
        [], [], [], [], [], [], []
    for j in range(n):
        k = idx[j]
        kk = lm_k[k]
        p0 = periods[kk]
        y0 = psi[kk]
        y0err = thresh[kk]
        sigma = abs((p0 - periods[kk + 1]) / 2.0)
        rank.append(j + 1)
        period.append(p0)
        period_err.append(sigma)
        power.append(y0)
        index.append(kk)
        frequency.append(1.0 / p0)
        threshold.append(y0err)

    tab = Table(
        [rank, period, period_err, power, index, frequency, threshold],
        names=['rank', 'period', 'period_err', 'power', 'index', 'freq', 'thresh'],
    )
    if write:
        tab.write(filename, format='csv', overwrite=True)
        print('Results written to %s' % filename)
    return tab
