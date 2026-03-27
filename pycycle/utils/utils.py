import numpy as np
from astropy.table import Table

__all__ = [
    "results_table",
]

def results_table(xx=None, yy=None, ee=None, n=None,
                  write=True, filename='tmp.csv'):
    """Return a table of the top n results from a pycycle run.
    The inputs (xx, yy, and ee) are the results from pycycle.

    Parameters
    ----------
    xx : `array` of `floats`
        periods (e.g., x)
    yy : `array` of `floats`
        power from periodogram (e.g., psi)
    ee : `array` of `floats`
        threshold values (e.g., conf)
    n : `int`
        number of ranked periods to show (e.g., 10)

    Returns
    -------
    tab : Astropy `Table`
        Table of top n results
    """

    assert isinstance(xx, np.ndarray)
    assert (xx.ndim == 1)
    xx_shape = xx.shape
    assert isinstance(yy, np.ndarray)
    assert (yy.shape == xx_shape)
    assert isinstance(ee, np.ndarray)
    assert (ee.shape == xx_shape)
    assert isinstance(n, int)
    assert (n >= 1)
    sz = len(xx)
    lm_x = np.zeros(sz)
    lm_y = np.zeros(sz)
    lm_k = np.zeros(sz,dtype=np.int_)
    j = 0
    assert (sz >= 3)
    for k in range(1, sz-1):
        ym1 = yy[k-1]
        y   = yy[k]
        yp1 = yy[k+1]
        if ((y>ym1) and (y>yp1)):
            lm_y[j] = yy[k]  # local maximum psiacc value
            lm_x[j] = xx[k]  # local maximum period value
            lm_k[j] = k
            j += 1
            lm_n = j
    lm_x = lm_x[:lm_n]
    lm_y = lm_y[:lm_n]
    lm_k = lm_k[:lm_n]
    assert (len(lm_y) >= n)
    idx = (-lm_y).argsort()[:n]  # indices (location) of the n largest values
    # print('TABLE: BEGIN')
    # print('rank   -------Period [days]------      Psi    index  Frequency  Thresh')
    # fmt = '%2d  %14.8f +- %11.8f %9.2f %8d %10.6f %7.2f'
    rank = []
    period = []
    period_err = []
    power = []
    index = []
    frequency = []
    thresh = []
    for j in range(n):
        k=idx[j]
        kk = lm_k[k]
        p0 = xx[kk]
        y0 = yy[kk]
        y0err = ee[kk]
        kkp1 = kk + 1
        p1 = xx[kkp1]
        sigma = abs((p0-p1)/2.)  # estimate of error (one standard deviation)
        rank.append(j+1)
        period.append(p0)
        period_err.append(sigma)
        power.append(y0)
        index.append(kk)
        frequency.append(1.0/p0)
        thresh.append(y0err)
        # print(fmt % ( j+1, p0, sigma, y0, kk, 1./p0, y0err))
    # print('TABLE: END')
    tab = Table([rank, period, period_err, power, index, frequency, thresh],
                names=['rank', 'period', 'period_err', 'power', 'index', 'freq', 'thresh'])
    if write:
        tab.write(filename, format='csv', overwrite=True)
        print(f"Results written to {filename}. Rename to avoid overwriting.")

    return tab
