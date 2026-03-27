"""Plotting routines for pycycle results.

Three plot types are provided:

* :func:`plot_observations` — raw light curve (HJD vs magnitude).
* :func:`plot_periodogram` — PSI periodogram vs frequency.
* :func:`plot_phased` — phased light curve folded at a given period.
"""

import numpy as np
import matplotlib.pyplot as plt

_BLUE = 'dodgerblue'
_RED = 'salmon'


def plot_observations(hjd, mag, filts, filtnams, tag=None, plotfile=None, xlim=None):
    """Plot the raw multi-band light curve (HJD vs magnitude).

    Parameters
    ----------
    hjd : ndarray of float64, shape (N,)
        Heliocentric Julian Dates.
    mag : ndarray of float64, shape (N,)
        Magnitudes co-aligned with *hjd*.
    filts : ndarray, shape (N,)
        Integer filter codes co-aligned with *hjd*.
    filtnams : list of str
        Filter names; index corresponds to filter code.
    tag : str, optional
        Text label added to the bottom-right of the figure.
    plotfile : str, optional
        Path to save the figure (PNG); figure is not saved if ``None``.
    xlim : tuple, optional
        Custom x-axis limits ``(xmin, xmax)``.
    """
    nfilts = len(filtnams)
    hjd0 = int(np.min(hjd))
    x = hjd - hjd0
    dx = max(0.08 * np.max(x), 0.25)
    if xlim is None:
        xlim = [-dx, np.max(x) + dx]
    xlabel = 'HJD - %d [days]' % hjd0
    dy = 0.5

    if nfilts > 1:
        fig, axarr = plt.subplots(nfilts, sharex=True, figsize=(8.5, 11))
        for i in range(nfilts):
            ok = (filts == float(i))
            xx, yy = x[ok], mag[ok]
            axarr[i].scatter(xx, yy, color=_BLUE, alpha=0.5)
            axarr[i].set_xlim(xlim)
            axarr[i].set_ylim([np.max(yy) + dy, np.min(yy) - dy])
            axarr[i].set_ylabel('mag', size='x-large')
            axarr[i].text(0.97, 0.80, filtnams[i], ha='right',
                          size='x-large', transform=axarr[i].transAxes)
            if i == nfilts - 1:
                axarr[i].set_xlabel(xlabel, size='x-large')
    else:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ok = (filts == float(0))
        xx, yy = x[ok], mag[ok]
        ax.scatter(xx, yy, color=_BLUE, alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim([np.max(yy) + dy, np.min(yy) - dy])
        ax.set_ylabel('mag', size='x-large')
        ax.set_xlabel(xlabel, size='x-large')
        ax.text(0.97, 0.90, filtnams[0], ha='right',
                size='x-large', transform=ax.transAxes)

    if tag is not None:
        plt.figtext(0.95, 0.1, tag, ha='right', va='bottom',
                    color='grey', size='large', rotation=90)
    if plotfile is not None:
        plt.savefig(plotfile, dpi=300)
        print(plotfile, '<--- plotfile written')
    plt.close()


def plot_periodogram(freq, psi_m, thresh_m, filtnams, tag=None,
                     plotfile=None, ylim=None, verbose=False):
    """Plot the hybrid PSI periodogram vs frequency.

    Parameters
    ----------
    freq : ndarray of float64, shape (N,)
        Frequencies [days⁻¹].
    psi_m : ndarray of float64
        PSI periodogram.  Shape ``(M, N)`` for *M* filters, or ``(N,)`` for
        a single filter.
    thresh_m : ndarray of float64
        Significance thresholds; same shape as *psi_m*.
    filtnams : list of str
        Filter names.
    tag : str, optional
        Figure label.
    plotfile : str, optional
        Output file path.
    ylim : tuple, optional
        Custom y-axis limits.
    verbose : bool, optional
        Print peak frequency/period for each filter.
    """
    nfilts = len(filtnams)
    periods = 1.0 / freq

    if nfilts > 1:
        fig, axarr = plt.subplots(nfilts + 1, sharex=True, figsize=(8.5, 11))
        for i in range(nfilts):
            axarr[i].plot(freq, psi_m[i], color=_BLUE, zorder=0)
            if np.any(thresh_m[i]):
                axarr[i].plot(freq, thresh_m[i], color=_RED, zorder=10)
            if ylim is not None:
                axarr[i].set_ylim(ylim)
            axarr[i].set_ylabel(r'${\Psi}$', size=19)
            axarr[i].text(0.97, 0.80, filtnams[i], ha='right',
                          size='x-large', transform=axarr[i].transAxes)
            if verbose:
                idx = np.argmax(psi_m[i])
                print('%8s : %12.2f %11.6f %12.7f' %
                      (filtnams[i], psi_m[i][idx], freq[idx], periods[idx]))
        j = nfilts
        psi_all = psi_m.sum(0)
        thresh_all = thresh_m.sum(0)
        axarr[j].plot(freq, psi_all, color=_BLUE, zorder=0)
        if np.any(thresh_all):
            axarr[j].plot(freq, thresh_all, color=_RED, zorder=10)
        if ylim is not None:
            axarr[j].set_ylim(ylim)
        axarr[j].set_ylabel(r'${\Psi}$', size=19)
        axarr[j].set_xlabel(r'Frequency [days$^{-1}$]', size='x-large')
        axarr[j].text(0.985, 0.80, 'ALL', ha='right',
                      size='x-large', transform=axarr[j].transAxes)
        if verbose:
            idx = np.argmax(psi_all)
            print('%8s : %12.2f %11.6f %12.7f' %
                  ('ALL', psi_all[idx], freq[idx], periods[idx]))
    else:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.plot(freq, psi_m, color=_BLUE, zorder=0)
        if np.any(thresh_m):
            ax.plot(freq, thresh_m, color=_RED, zorder=10)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(r'${\Psi}$', size=19)
        ax.set_xlabel(r'Frequency [days$^{-1}$]', size='x-large')
        ax.text(0.97, 0.90, filtnams[0], ha='right',
                size='x-large', transform=ax.transAxes)
        if verbose:
            idx = np.argmax(psi_m)
            print('%8s : %12.2f %11.6f %12.7f' %
                  (filtnams[0], psi_m[idx], freq[idx], periods[idx]))

    if tag is not None:
        plt.figtext(0.95, 0.1, tag, ha='right', va='bottom',
                    color='grey', size='large', rotation=90)
    if plotfile is not None:
        plt.savefig(plotfile, dpi=300)
        print(plotfile, '<--- plotfile written')
    plt.close()


def plot_phased(hjd, mag, magerr, filts, filtnams, period,
                tag=None, plotfile=None):
    """Plot the phased light curve folded at *period*.

    Parameters
    ----------
    hjd : ndarray of float64, shape (N,)
        Heliocentric Julian Dates.
    mag : ndarray of float64, shape (N,)
        Magnitudes.
    magerr : ndarray of float64, shape (N,)
        Magnitude errors.
    filts : ndarray, shape (N,)
        Integer filter codes.
    filtnams : list of str
        Filter names.
    period : float
        Folding period [days].
    tag : str, optional
        Figure label.
    plotfile : str, optional
        Output file path.
    """
    nfilts = len(filtnams)
    hjd0 = int(np.min(hjd))
    x = hjd - hjd0
    dx = 0.1
    xlim = [-dx, 2.0 + dx]
    xlabel = r'${\phi}$'
    dy = 0.5

    if nfilts > 1:
        fig, axarr = plt.subplots(nfilts, sharex=True, figsize=(8.5, 11))
        for i in range(nfilts):
            ok = (filts == float(i))
            xx, yy, ee = x[ok], mag[ok], magerr[ok]
            phi = (xx / period) % 1.0
            axarr[i].errorbar(phi,     yy, yerr=ee, fmt='o',
                              color=_BLUE, alpha=0.5)
            axarr[i].errorbar(phi + 1, yy, yerr=ee, fmt='o',
                              color=_BLUE, alpha=0.5)
            axarr[i].set_xlim(xlim)
            axarr[i].set_ylim([np.max(yy + ee) + dy, np.min(yy - ee) - dy])
            axarr[i].set_ylabel('mag', size='x-large')
            axarr[i].text(0.97, 0.80, filtnams[i], ha='right',
                          size='x-large', transform=axarr[i].transAxes)
            if i == nfilts - 1:
                axarr[i].set_xlabel(xlabel, size=20)
    else:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ok = (filts == float(0))
        xx, yy, ee = x[ok], mag[ok], magerr[ok]
        phi = (xx / period) % 1.0
        ax.errorbar(phi,     yy, yerr=ee, fmt='o', color=_BLUE, alpha=0.5)
        ax.errorbar(phi + 1, yy, yerr=ee, fmt='o', color=_BLUE, alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim([np.max(yy + ee) + dy, np.min(yy - ee) - dy])
        ax.set_ylabel('mag', size='x-large')
        ax.set_xlabel(xlabel, size=20)
        ax.text(0.97, 0.90, filtnams[0], ha='right',
                size='x-large', transform=ax.transAxes)

    plt.figtext(0.5, 0.93, 'Period: %9.6f days' % period,
                ha='center', color='black', size='xx-large')
    if tag is not None:
        plt.figtext(0.95, 0.1, tag, ha='right', va='bottom',
                    color='grey', size='large', rotation=90)
    if plotfile is not None:
        plt.savefig(plotfile, dpi=300)
        print(plotfile, '<--- plotfile written')
    plt.close()
