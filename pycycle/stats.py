"""Statistical utilities: data scrambling and summary statistics."""

import numpy as np


def scramble(arr):
    """Randomly permute an array, returning both a sorted-index and pick-index version.

    Parameters
    ----------
    arr : ndarray
        Input 1-D array.

    Returns
    -------
    scrambarr : ndarray
        *arr* reordered by a sorted permutation of random indices.
    pickarr : ndarray
        *arr* sampled with replacement using the same random indices.
    """
    ns = len(arr)
    x = np.random.choice(ns, size=ns)
    s = np.argsort(x)
    scrambarr = arr[s]
    pickarr = arr[x]
    return scrambarr, pickarr


def summary(x, tag=None):
    """Print basic descriptive statistics for array *x*.

    Parameters
    ----------
    x : ndarray
        Data values.
    tag : str, optional
        Label prepended to each printed line.
    """
    tag = tag or ''
    if len(x) > 0:
        print(tag, 'median :', np.median(x))
        print(tag, '  mean :', np.mean(x))
        print(tag, '   std :', np.std(x))
        print(tag, '   min :', np.min(x))
        print(tag, '   max :', np.max(x))
        print(tag, '     n :', len(x))
        print(tag, '   NaN :', np.count_nonzero(np.isnan(x)))
