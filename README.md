# pycycle

**pycycle** is a hybrid Lomb-Scargle / Lafler-Kinman period finder for multiband
variable-star light curves, based on
[Saha & Vivas (2017, AJ 154, 231)](https://doi.org/10.3847/1538-3881/aa73d9).

It also includes a fast Cython-accelerated RR Lyrae template fitter that works with
two external template libraries (see below).

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

pycycle can fit RR Lyrae light curves against two external template libraries.
You need to clone (or download) them separately:

### rr-templates (Long / Stringer et al.)

SDSS and DES averaged templates with full physics parameterisation
(distance modulus, dust, amplitude, phase).

```
git clone https://github.com/longjp/rr-templates
```

> Templates constructed for and used in
> [Stringer et al. 2019 (AJ 157, 187)](https://doi.org/10.3847/1538-3881/ab1f46).
> Please cite this work if you use these templates.
> Templates by James Long (jplong@mdanderson.org).

```python
from pycycle.templates import load_rr_template
from pycycle.template_fit import TemplateFitter

template = load_rr_template('/path/to/rr-templates/template_sdss', name='sdss')
fitter = TemplateFitter(template)
result = fitter.fit(hjd, mag, magerr, filts, filtnams, pmin=0.2, dphi=0.02)
print(result.best_period)
result.plot_phased()
```

### Multiband-templates (Baeza-Villagra et al.)

136 RRab and 144 RRc individual DECam griz templates, normalised to [0, 1].

```
git clone https://github.com/kbbaeza/Multiband-templates
```

> [Baeza-Villagra et al. 2025 (A&A)](https://arxiv.org/abs/2501.03813):
> *High-cadence stellar variability studies of RR Lyrae stars with DECam:
> New multiband templates.*
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

## Installation

Requires Python ≥ 3.9. The C/Cython extensions are optional but give a significant
speed boost.

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

---

## Credits

- **Period search algorithm:** Saha & Vivas (2017, AJ 154, 231)
- **rr-templates:** Long/Stringer et al. (2019, AJ 157, 187)
- **Multiband-templates:** Baeza-Villagra et al. (2025, A&A, arXiv:2501.03813)
