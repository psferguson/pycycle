"""Plotting routines for pycycle results.

Three plot types are provided:

* :func:`plot_observations` — raw light curve (HJD vs magnitude).
* :func:`plot_periodogram` — PSI periodogram vs frequency.
* :func:`plot_phased` — phased light curve folded at a given period.

Each accepts an optional *axes* argument so the plot can be placed inside a
figure you already own — a validation grid, say, rather than a standalone
figure.  Pass a single ``Axes`` to overlay all bands on one panel (bands are
colour-coded and a legend is drawn), or a sequence of at least ``len(filtnams)``
axes for the usual one-band-per-panel layout.  With ``axes=None`` the functions
behave exactly as before: they build their own stacked figure and close it.
All three return the :class:`matplotlib.figure.Figure` they drew into.
"""

import numpy as np
import matplotlib.pyplot as plt

_BLUE = 'dodgerblue'
_RED = 'salmon'

#: Per-band colours used when several bands share one panel.  These are the
#: official Rubin colourblind-friendly mappings from RTN-045 -- see
#: :mod:`pycycle.lsst_style`, which prefers ``lsst.utils.plotting`` when the
#: stack is importable and falls back to a vendored copy otherwise.
from .lsst_style import band_color as _band_color  # noqa: E402


def _prepare_axes(axes, nfilts, figsize, sharex=True, npanels=None):
    """Resolve the *axes* argument into a concrete list of panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    panels : list of Axes
        One entry per band (length ``npanels``).  In overlay mode every entry
        is the *same* Axes.
    created : bool
        True when this call built the figure (and therefore owns it).
    overlay : bool
        True when all bands share a single panel.
    """
    npanels = nfilts if npanels is None else npanels

    if axes is None:
        fig, arr = plt.subplots(npanels, sharex=sharex, figsize=figsize,
                                squeeze=False)
        return fig, list(arr[:, 0]), True, False

    if isinstance(axes, plt.Axes):
        return axes.figure, [axes] * npanels, False, True

    panels = list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())
    if not panels:
        raise ValueError('axes sequence is empty')
    if len(panels) < npanels:
        # Not enough panels for one band each: overlay rather than silently
        # dropping the bands that do not fit.
        return panels[0].figure, [panels[0]] * npanels, False, True
    return panels[0].figure, panels[:npanels], False, False


def _finish(fig, created, overlay, tag, plotfile):
    """Shared tail: optional tag, optional save, and close only what we own."""
    if tag is not None:
        fig.text(0.95, 0.1, tag, ha='right', va='bottom',
                 color='grey', size='large', rotation=90)
    if plotfile is not None:
        fig.savefig(plotfile, dpi=300)
        print(plotfile, '<--- plotfile written')
    if created:
        plt.close(fig)
    return fig


def plot_observations(hjd, mag, filts, filtnams, tag=None, plotfile=None,
                      xlim=None, axes=None):
    """Plot the raw multi-band light curve (HJD vs magnitude).

    Parameters
    ----------
    hjd : ndarray of float64, shape (N,)
        Heliocentric Julian Dates.
    mag : ndarray of float64, shape (N,)
        Magnitudes co-aligned with *hjd*.
    filts : array-like of str, shape (N,)
        Filter name (band) per observation, co-aligned with *hjd*.
    filtnams : list of str
        Bands to plot, one panel each, in the desired display order.
    tag : str, optional
        Text label added to the bottom-right of the figure.
    plotfile : str, optional
        Path to save the figure (PNG); figure is not saved if ``None``.
    xlim : tuple, optional
        Custom x-axis limits ``(xmin, xmax)``.
    axes : Axes or sequence of Axes, optional
        Draw into these instead of creating a figure.  See the module
        docstring.

    Returns
    -------
    matplotlib.figure.Figure
    """
    filts = np.asarray(filts)
    nfilts = len(filtnams)
    hjd0 = int(np.min(hjd))
    x = hjd - hjd0
    dx = max(0.08 * np.max(x), 0.25)
    if xlim is None:
        xlim = [-dx, np.max(x) + dx]
    xlabel = 'HJD - %d [days]' % hjd0
    dy = 0.5

    fig, panels, created, overlay = _prepare_axes(axes, nfilts, (8.5, 11))

    ymin, ymax = np.inf, -np.inf
    for i, fname in enumerate(filtnams):
        ok = (filts == fname)
        if not np.any(ok):
            continue
        xx, yy = x[ok], mag[ok]
        ax = panels[i]
        color = _band_color(fname) if overlay else _BLUE
        ax.scatter(xx, yy, color=color, alpha=0.5,
                   label=fname if overlay else None)
        ymin, ymax = min(ymin, np.min(yy)), max(ymax, np.max(yy))
        if not overlay:
            ax.set_xlim(xlim)
            ax.set_ylim([np.max(yy) + dy, np.min(yy) - dy])
            ax.set_ylabel('mag', size='x-large')
            ax.text(0.97, 0.80, fname, ha='right', size='x-large',
                    transform=ax.transAxes)
            if i == nfilts - 1:
                ax.set_xlabel(xlabel, size='x-large')

    if overlay and np.isfinite(ymin):
        ax = panels[0]
        ax.set_xlim(xlim)
        ax.set_ylim([ymax + dy, ymin - dy])
        ax.set_ylabel('mag', size='x-large')
        ax.set_xlabel(xlabel, size='x-large')
        ax.legend(loc='best', fontsize='small', ncol=min(nfilts, 3))

    return _finish(fig, created, overlay, tag, plotfile)


def plot_periodogram(freq, psi_m, thresh_m, filtnams, tag=None,
                     plotfile=None, ylim=None, verbose=False, axes=None):
    """Plot the hybrid PSI periodogram vs frequency.

    With ``axes=None`` a figure of ``len(filtnams) + 1`` panels is built: one
    per band plus a combined "ALL" panel.  When a single Axes is supplied only
    the band-summed periodogram is drawn into it, since that is the quantity
    ``PeriodSearchResult.best_period`` is taken from.

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
    axes : Axes or sequence of Axes, optional
        Draw into these instead of creating a figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    nfilts = len(filtnams)
    periods = 1.0 / freq
    multi = np.asarray(psi_m).ndim > 1

    def _report(label, psi):
        idx = np.argmax(psi)
        print('%8s : %12.2f %11.6f %12.7f' %
              (label, psi[idx], freq[idx], periods[idx]))

    fig, panels, created, overlay = _prepare_axes(
        axes, nfilts, (8.5, 11), npanels=(nfilts + 1) if multi else 1)

    if overlay:
        # single panel: show the combined periodogram, which is what the
        # reported best period comes from
        psi_all = psi_m.sum(0) if multi else psi_m
        thresh_all = thresh_m.sum(0) if multi else thresh_m
        ax = panels[0]
        ax.plot(freq, psi_all, color=_BLUE, zorder=0)
        if np.any(thresh_all):
            ax.plot(freq, thresh_all, color=_RED, zorder=10)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(r'${\Psi}$', size=19)
        ax.set_xlabel(r'Frequency [days$^{-1}$]', size='x-large')
        ax.text(0.985, 0.90, 'ALL' if multi else filtnams[0], ha='right',
                size='x-large', transform=ax.transAxes)
        if verbose:
            _report('ALL' if multi else filtnams[0], psi_all)
        return _finish(fig, created, overlay, tag, plotfile)

    if multi:
        for i in range(nfilts):
            ax = panels[i]
            ax.plot(freq, psi_m[i], color=_BLUE, zorder=0)
            if np.any(thresh_m[i]):
                ax.plot(freq, thresh_m[i], color=_RED, zorder=10)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_ylabel(r'${\Psi}$', size=19)
            ax.text(0.97, 0.80, filtnams[i], ha='right', size='x-large',
                    transform=ax.transAxes)
            if verbose:
                _report(filtnams[i], psi_m[i])
        ax = panels[nfilts]
        psi_all = psi_m.sum(0)
        thresh_all = thresh_m.sum(0)
        ax.plot(freq, psi_all, color=_BLUE, zorder=0)
        if np.any(thresh_all):
            ax.plot(freq, thresh_all, color=_RED, zorder=10)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(r'${\Psi}$', size=19)
        ax.set_xlabel(r'Frequency [days$^{-1}$]', size='x-large')
        ax.text(0.985, 0.80, 'ALL', ha='right', size='x-large',
                transform=ax.transAxes)
        if verbose:
            _report('ALL', psi_all)
    else:
        ax = panels[0]
        ax.plot(freq, psi_m, color=_BLUE, zorder=0)
        if np.any(thresh_m):
            ax.plot(freq, thresh_m, color=_RED, zorder=10)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(r'${\Psi}$', size=19)
        ax.set_xlabel(r'Frequency [days$^{-1}$]', size='x-large')
        ax.text(0.97, 0.90, filtnams[0], ha='right', size='x-large',
                transform=ax.transAxes)
        if verbose:
            _report(filtnams[0], psi_m)

    return _finish(fig, created, overlay, tag, plotfile)


def plot_phased(hjd, mag, magerr, filts, filtnams, period,
                tag=None, plotfile=None, axes=None):
    """Plot the phased light curve folded at *period*.

    Parameters
    ----------
    hjd : ndarray of float64, shape (N,)
        Heliocentric Julian Dates.
    mag : ndarray of float64, shape (N,)
        Magnitudes.
    magerr : ndarray of float64, shape (N,)
        Magnitude errors.
    filts : array-like of str, shape (N,)
        Filter name (band) per observation.
    filtnams : list of str
        Bands to plot, one panel each, in the desired display order.
    period : float
        Folding period [days].
    tag : str, optional
        Figure label.
    plotfile : str, optional
        Output file path.
    axes : Axes or sequence of Axes, optional
        Draw into these instead of creating a figure.  A single Axes overlays
        all bands.

    Returns
    -------
    matplotlib.figure.Figure
    """
    filts = np.asarray(filts)
    nfilts = len(filtnams)
    hjd0 = int(np.min(hjd))
    x = hjd - hjd0
    dx = 0.1
    xlim = [-dx, 2.0 + dx]
    xlabel = r'${\phi}$'
    dy = 0.5

    fig, panels, created, overlay = _prepare_axes(axes, nfilts, (8.5, 11))

    ymin, ymax = np.inf, -np.inf
    for i, fname in enumerate(filtnams):
        ok = (filts == fname)
        if not np.any(ok):
            continue
        xx, yy, ee = x[ok], mag[ok], magerr[ok]
        phi = (xx / period) % 1.0
        ax = panels[i]
        color = _band_color(fname) if overlay else _BLUE
        ax.errorbar(phi, yy, yerr=ee, fmt='o', color=color, alpha=0.5,
                    label=fname if overlay else None)
        ax.errorbar(phi + 1, yy, yerr=ee, fmt='o', color=color, alpha=0.5)
        ymin = min(ymin, np.min(yy - ee))
        ymax = max(ymax, np.max(yy + ee))
        if not overlay:
            ax.set_xlim(xlim)
            ax.set_ylim([np.max(yy + ee) + dy, np.min(yy - ee) - dy])
            ax.set_ylabel('mag', size='x-large')
            ax.text(0.97, 0.80, fname, ha='right', size='x-large',
                    transform=ax.transAxes)
            if i == nfilts - 1:
                ax.set_xlabel(xlabel, size=20)

    if overlay and np.isfinite(ymin):
        ax = panels[0]
        ax.set_xlim(xlim)
        ax.set_ylim([ymax + dy, ymin - dy])
        ax.set_ylabel('mag', size='x-large')
        ax.set_xlabel(xlabel, size=20)
        ax.set_title('Period: %9.6f days' % period)
        ax.legend(loc='best', fontsize='small', ncol=min(nfilts, 3))
    else:
        fig.text(0.5, 0.93, 'Period: %9.6f days' % period,
                 ha='center', color='black', size='xx-large')

    return _finish(fig, created, overlay, tag, plotfile)
