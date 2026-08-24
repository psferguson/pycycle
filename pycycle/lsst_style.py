"""Official Rubin/LSST per-band plot colours, symbols and line styles.

These are the colourblind-friendly mappings defined in
`RTN-045 <https://rtn-045.lsst.io/#colorblind-friendly-plots>`_ and shipped in
``lsst.utils.plotting``.  pycycle does not depend on the LSST stack, so the
values are vendored here — but if ``lsst.utils.plotting`` *is* importable, its
definitions are used instead, so this module can never drift from the stack on a
machine that has it.

Only the **light background** palette is provided.  ``lsst.utils`` also defines a
dark-background variant; if you need it, take it from the stack directly.

Band naming
-----------
The LSST mappings are keyed by lowercase SDSS filter names ``ugrizy``.  pycycle
renames ``y`` to ``Y`` when loading templates (``DP2Config.band_map``), because
the DES template library spells it that way, so lookups here are
case-insensitive and accept either spelling.  Anything unrecognised falls back
to a neutral grey rather than raising — a plot should not be the thing that
fails a run.

    >>> from pycycle.lsst_style import band_color, band_symbol
    >>> band_color('g'), band_color('Y') == band_color('y')
    ('#31DE1F', True)
"""
from __future__ import annotations

__all__ = ['BAND_COLORS', 'BAND_SYMBOLS', 'BAND_LINESTYLES',
           'band_color', 'band_symbol', 'band_linestyle', 'FALLBACK_COLOR']

#: Colour for a band this module does not know about.
FALLBACK_COLOR = '#808080'
_FALLBACK_SYMBOL = 'o'
_FALLBACK_LINESTYLE = '-'


# --- vendored from lsst.utils.plotting.figures (RTN-045), light background ---
_COLORS = {
    'u': '#1600EA',
    'g': '#31DE1F',
    'r': '#B52626',
    'i': '#370201',
    'z': '#BA52FF',
    'y': '#61A2B3',
}

_SYMBOLS = {
    'u': 'o',
    'g': '^',
    'r': 'v',
    'i': 's',
    'z': '*',
    'y': 'p',
}

_LINESTYLES = {
    'u': '--',
    'g': (0, (3, 1, 1, 1)),
    'r': '-.',
    'i': '-',
    'z': (0, (3, 1, 1, 1, 1, 1)),
    'y': ':',
}


def _from_stack():
    """Prefer the real definitions when the LSST stack is importable."""
    try:
        from lsst.utils.plotting import (get_multiband_plot_colors,
                                         get_multiband_plot_symbols,
                                         get_multiband_plot_linestyles)
    except Exception:      # not installed, or a stack import error -- vendored is fine
        return None
    try:
        return (dict(get_multiband_plot_colors(dark_background=False)),
                dict(get_multiband_plot_symbols()),
                dict(get_multiband_plot_linestyles()))
    except Exception:
        return None


_stack = _from_stack()
if _stack is not None:
    _COLORS, _SYMBOLS, _LINESTYLES = _stack

#: Band -> colour, symbol, line style.  Light background only.
BAND_COLORS = dict(_COLORS)
BAND_SYMBOLS = dict(_SYMBOLS)
BAND_LINESTYLES = dict(_LINESTYLES)


def _key(band) -> str:
    """Normalise a band name to the lowercase LSST spelling."""
    return str(band).strip().lower()


def band_color(band, default: str = FALLBACK_COLOR) -> str:
    """Official LSST colour for *band*, or *default* if unrecognised."""
    return BAND_COLORS.get(_key(band), default)


def band_symbol(band, default: str = _FALLBACK_SYMBOL) -> str:
    """Official LSST marker for *band*, or *default* if unrecognised."""
    return BAND_SYMBOLS.get(_key(band), default)


def band_linestyle(band, default=_FALLBACK_LINESTYLE):
    """Official LSST line style for *band*, or *default* if unrecognised.

    Note these are matplotlib dash tuples for ``g`` and ``z``, not strings.
    """
    return BAND_LINESTYLES.get(_key(band), default)
