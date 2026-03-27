# pycycle

**pycycle** is a hybrid Lomb-Scargle / Lafler-Kinman period finder for multiband
variable-star light curves, based on
[Saha & Vivas (2017, AJ 154, 231)](https://doi.org/10.3847/1538-3881/aa73d9).

It also includes a fast Cython-accelerated RR Lyrae template fitter and utilities
for catalog-scale searches on Rubin LSST data via [LSDB](https://lsdb.io).

---

## Period search

```python
from pycycle import PeriodSearch
import numpy as np

hjd, mag, magerr, filts = np.loadtxt('data.tab', unpack=True)
ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=['V'])
result = ps.run(pmin=0.2, dphi=0.02)
print(result.best_period)
result.plot_phased()
```

See `notebooks/tutorial.ipynb` for a full walkthrough.

---

## Template fitting

pycycle fits RR Lyrae light curves against two external template libraries.
Clone them separately:

### rr-templates (Long / Stringer et al.)

SDSS and DES averaged templates with full physics parameterisation
(distance modulus µ, dust E(B-V), amplitude A, phase φ).

```
git clone https://github.com/longjp/rr-templates
```

> Templates constructed for and used in
> [Stringer et al. 2019 (AJ 157, 187)](https://doi.org/10.3847/1538-3881/ab1f46).
> Please cite this work if you use these templates.

```python
from pycycle.templates import load_rr_template
from pycycle.template_fit import TemplateFitter

template = load_rr_template('/path/to/rr-templates/template_des', name='des')
fitter = TemplateFitter(template, n_newton=5, warm_start=True)
result = fitter.fit(hjd, mag, magerr, filts, filtnams,
                    pmin=0.44, dphi=0.02, pmax=0.89)
print(result.best_period, result.best_coeffs)
result.plot_phased()
```

**`warm_start=True`** carries the solution `(φ, µ, E(B-V), A)` from each
frequency as the starting point for the next — ~4× faster than the default
multi-start mode, following the optimisation in Stringer et al. (2019).

### Multiband-templates (Baeza-Villagra et al.)

136 RRab and 144 RRc individual DECam griz templates, normalised to [0, 1].

```
git clone https://github.com/KarinaBaezaV/Multiband-templates
```

> [Baeza-Villagra et al. 2025 (A&A, arXiv:2501.03813)](https://arxiv.org/abs/2501.03813).
> Please cite this work if you use these templates.

```python
from pycycle.templates import load_multiband_templates, average_multiband_templates
from pycycle.template_fit import TemplateFitter

templates = load_multiband_templates('/path/to/Multiband-templates/RRab_normalized.zip')
avg = average_multiband_templates(templates)
fitter = TemplateFitter(avg)
result = fitter.fit(hjd, mag, magerr, filts, filtnams, pmin=0.2, dphi=0.02)
```

See `notebooks/template_fitting.ipynb` for a complete worked example.

---

## When to use period search vs template fitting

| LSST phase | Obs/band | Recommended strategy |
|---|---|---|
| Year 1–3 | ~4–10 | **Template fitting directly** — period search is unreliable at this sparsity |
| Year 5–7 | ~15–30 | **Template fitting directly** |
| Year 10 | ~50–100 | **Period search → template fitting** on high-PSI candidates |
| Well-sampled (>30/band) | any | **Period search first** (~5 s/star), then template fit candidates |
| All variable types | any | **Period search** — template fitting is RRab-specific |

---

## LSST / Rubin utilities

`pycycle.lsdb_utils` provides helpers for running the template fitter at
catalog scale via [LSDB](https://lsdb.io):

```python
import lsdb
from pycycle.templates import load_rr_template
from pycycle.lsdb_utils import apply_des_to_lsst_correction, make_template_fit_fn

# Apply DES → LSST filter corrections (RTN-099) once at load time
template = load_rr_template('/path/to/rr-templates/template_des', name='des')
apply_des_to_lsst_correction(template)   # zero runtime cost during fitting

# Build a map_partitions-compatible function
fit_fn, meta = make_template_fit_fn(
    template,
    bands=['g', 'r', 'i', 'z', 'y'],
    pmin=0.44, dphi=0.02, pmax=0.89,
    n_newton=5, warm_start=True,
)

# Run on a joined object+source catalog
objects = lsdb.open_catalog('...dp01_object...',  columns=[...])
sources = lsdb.open_catalog('...dp01_forced_source...', columns=[...])
stars   = objects.query('extendedness == 0 and 21 < r_psfMag < 24.5')
joined  = stars.nest_sources(sources, source_id_col='objectId')
results = joined.map_partitions(fit_fn, meta=meta).compute()
```

Also provided: `flux_to_mag` (Rubin nJy → AB magnitude) and
`compute_variability_features` (χ²_ν and significance for pre-filtering
following Stringer et al. 2019 §3.2).

See `notebooks/lsdb_lsst_pipeline.ipynb` for the full two-stage pipeline.

---

## Installation

Requires Python ≥ 3.9. The C/Cython extensions are optional but give a
significant speed boost.

```bash
pip install -e .
# or with Cython extensions:
pip install -e ".[dev]"
```

The extensions are built automatically if Cython and a C compiler are available.

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/tutorial.ipynb` | Period finding with `PeriodSearch` on the bundled B1392 dataset |
| `notebooks/template_fitting.ipynb` | Template fitting with both rr-templates and Multiband-templates |
| `notebooks/lsdb_lsst_pipeline.ipynb` | Catalog-scale RR Lyrae search on Rubin LSST data via LSDB |

---

## Credits

- **Period search algorithm:** Saha & Vivas (2017, AJ 154, 231)
- **rr-templates:** Long/Stringer et al. (2019, AJ 157, 187)
- **Multiband-templates:** Baeza-Villagra et al. (2025, A&A, arXiv:2501.03813)
- **LSST pipeline strategy:** Stringer et al. (2019), Stringer & Drlica-Wagner et al. (2021, arXiv:2011.13930)
