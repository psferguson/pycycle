"""RR Lyrae template loading for pycycle.

Two external template libraries are supported.  Neither is bundled here —
clone them separately and pass the path at load time.

rr-templates (Long / Stringer et al.)
    https://github.com/longjp/rr-templates
    CSV files: templates.csv, betas.csv, dust.csv
    Cite: Stringer et al. 2019, AJ 157, 187

Multiband-templates (Baeza-Villagra et al.)
    https://github.com/kbbaeza/Multiband-templates
    ZIP archives of individual normalised star templates
    Cite: Baeza-Villagra et al. 2025, A&A, arXiv:2501.03813
"""
from __future__ import annotations

import csv
import os
import zipfile

import numpy as np


class RRTemplate:
    """Normalised RR Lyrae light-curve template.

    Attributes
    ----------
    name : str
        Human-readable label.
    bands : list of str
        Band names in row order of *gamma*.
    phase : ndarray, shape (n_phase,)
        Uniform phase grid in [0, 1).
    gamma : ndarray, shape (n_bands, n_phase)
        Normalised template values.  Convention matches the source library:
        for rr-templates these are zero-mean offsets; for Multiband-templates
        these are magnitudes normalised to [0, 1] (0 = maximum brightness).
    dust : ndarray, shape (n_bands,) or None
        Per-band extinction coefficients (R_band).  None for Multiband-templates.
    betas : ndarray, shape (n_bands, 3) or None
        Frequency-dependent absolute magnitude corrections [c0, p1, p2] per band
        such that beta_b(omega) = c0 + p1*omega + p2*omega^2.
        None for Multiband-templates.
    """

    def __init__(self, name, bands, phase, gamma, dust=None, betas=None):
        self.name = name
        self.bands = list(bands)
        self.phase = np.asarray(phase, dtype=np.float64)
        self.gamma = np.asarray(gamma, dtype=np.float64)
        self.dust = None if dust is None else np.asarray(dust, dtype=np.float64)
        self.betas = None if betas is None else np.asarray(betas, dtype=np.float64)

    @property
    def n_bands(self):
        return len(self.bands)

    @property
    def n_phase(self):
        return len(self.phase)

    def band_index(self, band):
        return self.bands.index(band)

    def dgamma(self):
        """Return template derivative d(gamma)/d(phase) via central differences.

        Returns
        -------
        dg : ndarray, shape (n_bands, n_phase)
        """
        n = self.n_phase
        dg = np.empty_like(self.gamma)
        dg[:, 1:-1] = (self.gamma[:, 2:] - self.gamma[:, :-2]) * (n / 2.0)
        dg[:, 0] = (self.gamma[:, 1] - self.gamma[:, -1]) * (n / 2.0)
        dg[:, -1] = (self.gamma[:, 0] - self.gamma[:, -2]) * (n / 2.0)
        return dg

    def __repr__(self):
        return (f"RRTemplate(name={self.name!r}, bands={self.bands}, "
                f"n_phase={self.n_phase}, has_dust={self.dust is not None})")


# ---------------------------------------------------------------------------
# rr-templates loader
# ---------------------------------------------------------------------------

def load_rr_template(template_dir: str, name: str = '') -> RRTemplate:
    """Load an averaged rr-template from Long / Stringer et al. CSV files.

    Parameters
    ----------
    template_dir : str
        Directory containing ``templates.csv``, ``betas.csv``, ``dust.csv``.
        E.g. ``/path/to/rr-templates/template_sdss``.
    name : str, optional
        Label.  Defaults to the directory basename.

    Returns
    -------
    RRTemplate
    """
    if not name:
        name = os.path.basename(os.path.normpath(template_dir))

    # templates.csv — one row per band, 100 phase-point values per row
    gamma_dict: dict[str, np.ndarray] = {}
    with open(os.path.join(template_dir, 'templates.csv')) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            band = parts[0].strip('"')
            gamma_dict[band] = np.array([float(x) for x in parts[1:]])

    bands = list(gamma_dict.keys())
    n_phase = len(next(iter(gamma_dict.values())))

    # betas.csv — rows: c0 / p1 / p2;  columns: bands
    beta_rows: dict[str, list] = {}
    beta_bands: list[str] = []
    with open(os.path.join(template_dir, 'betas.csv')) as fh:
        reader = csv.reader(fh)
        header = [h.strip('"') for h in next(reader)]
        beta_bands = header[1:]
        for row in reader:
            key = row[0].strip('"')
            beta_rows[key] = [float(v) for v in row[1:]]

    betas_dict = {b: [beta_rows['c0'][i], beta_rows['p1'][i], beta_rows['p2'][i]]
                  for i, b in enumerate(beta_bands)}

    # dust.csv — one row per band
    dust_dict: dict[str, float] = {}
    with open(os.path.join(template_dir, 'dust.csv')) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            dust_dict[parts[0].strip('"')] = float(parts[1])

    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    gamma = np.array([gamma_dict[b] for b in bands])
    dust = np.array([dust_dict.get(b, 0.0) for b in bands])
    betas = np.array([betas_dict.get(b, [0.0, 0.0, 0.0]) for b in bands])

    return RRTemplate(name=name, bands=bands, phase=phase,
                      gamma=gamma, dust=dust, betas=betas)


# ---------------------------------------------------------------------------
# Multiband-templates loaders
# ---------------------------------------------------------------------------

def load_multiband_template(zip_path: str, star_id: str = None,
                             n_phase: int = 100) -> RRTemplate:
    """Load a single template from a Multiband-templates ZIP archive.

    Parameters
    ----------
    zip_path : str
        Path to ``RRab_normalized.zip`` or ``RRc_normalized.zip``.
    star_id : str, optional
        OGLE identifier substring (e.g. ``'OGLE-BLG-RRLYR-12786'``).
        If *None*, the first file in the archive is loaded.
    n_phase : int
        Number of phase grid points to resample to.

    Returns
    -------
    RRTemplate
    """
    with zipfile.ZipFile(zip_path) as z:
        fnames = sorted(f for f in z.namelist() if f.endswith('.txt'))
        if star_id is not None:
            matches = [f for f in fnames if star_id in f]
            if not matches:
                raise ValueError(f"Star {star_id!r} not found in {zip_path}")
            fname = matches[0]
        else:
            fname = fnames[0]
        data = z.read(fname).decode()

    name = os.path.splitext(os.path.basename(fname))[0]
    return _parse_mb_template(data, name, n_phase)


def load_multiband_templates(zip_path: str, n_phase: int = 100,
                              max_templates: int = None) -> list:
    """Load all templates from a Multiband-templates ZIP archive.

    Parameters
    ----------
    zip_path : str
        Path to ``RRab_normalized.zip`` or ``RRc_normalized.zip``.
    n_phase : int
        Number of phase grid points to resample to.
    max_templates : int, optional
        Cap on number of templates loaded (useful for quick tests).

    Returns
    -------
    list of RRTemplate
    """
    templates = []
    with zipfile.ZipFile(zip_path) as z:
        fnames = sorted(f for f in z.namelist() if f.endswith('.txt'))
        if max_templates is not None:
            fnames = fnames[:max_templates]
        for fname in fnames:
            name = os.path.splitext(os.path.basename(fname))[0]
            try:
                data = z.read(fname).decode()
                templates.append(_parse_mb_template(data, name, n_phase))
            except Exception:
                pass
    return templates


def average_multiband_templates(templates: list, n_phase: int = 100) -> RRTemplate:
    """Return an averaged RRTemplate from a list of Multiband templates.

    All templates must share the same band set.  The average is taken over the
    uniform phase grid used internally (resampling if necessary).

    Parameters
    ----------
    templates : list of RRTemplate
    n_phase : int
        Phase grid points of the output template.

    Returns
    -------
    RRTemplate
    """
    bands = templates[0].bands
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    gamma_sum = np.zeros((len(bands), n_phase))
    count = 0
    for t in templates:
        if t.bands != bands:
            continue
        if t.n_phase == n_phase:
            gamma_sum += t.gamma
        else:
            # resample
            for bi in range(len(bands)):
                gamma_sum[bi] += np.interp(phase, t.phase,
                                           t.gamma[bi], period=1.0)
        count += 1
    if count == 0:
        raise ValueError("No compatible templates to average.")
    return RRTemplate(
        name=f'avg_{os.path.basename(os.path.splitext(templates[0].name)[0])[:12]}',
        bands=bands,
        phase=phase,
        gamma=gamma_sum / count,
        dust=None,
        betas=None,
    )


def _parse_mb_template(data: str, name: str, n_phase: int) -> RRTemplate:
    """Parse raw text of a normalised Multiband-templates file."""
    rows = data.strip().split('\n')
    header = [h.lower().strip() for h in rows[0].split(',')]
    ph_col = header.index('phase')
    mg_col = header.index('mag')
    bd_col = header.index('band')

    ph_by_band: dict[str, list] = {}
    mg_by_band: dict[str, list] = {}

    for row in rows[1:]:
        parts = row.strip().split(',')
        if len(parts) <= max(ph_col, mg_col, bd_col):
            continue
        ph = float(parts[ph_col])
        mg = float(parts[mg_col])
        bd = parts[bd_col].strip()
        ph_by_band.setdefault(bd, []).append(ph)
        mg_by_band.setdefault(bd, []).append(mg)

    phase_grid = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    bands = sorted(ph_by_band.keys())
    gamma = np.zeros((len(bands), n_phase))

    for i, band in enumerate(bands):
        ph_arr = np.array(ph_by_band[band])
        mg_arr = np.array(mg_by_band[band])
        # keep only phase in [0, 1), sort, interpolate
        mask = ph_arr < 1.0
        ph_arr, mg_arr = ph_arr[mask], mg_arr[mask]
        order = np.argsort(ph_arr)
        gamma[i] = np.interp(phase_grid, ph_arr[order], mg_arr[order], period=1.0)

    return RRTemplate(name=name, bands=bands, phase=phase_grid,
                      gamma=gamma, dust=None, betas=None)
