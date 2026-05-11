"""Parallel recovery test for the Multiband RRL template diversity study.

For each trial we:
  1. Pick a random truth template and a random period in the appropriate range.
  2. Simulate a sparse LSST-like multiband light curve.
  3. Fit ALL templates against the simulated curve in parallel (one process per
     template via multiprocessing.Pool).
  4. Take the best-RSS over the full library, and (by selecting indices) over
     each medoid subset for free.

Outputs per_template RSS for every trial so we can also study things like
"how often does the k=10 medoid set include the global best fit" in the
summary notebook.

Run under the search26 env. Reloads checkpoints written by analyze.py so the
slow clustering / PCA step doesn't have to repeat.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import pickle
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np

from pycycle.templates import load_multiband_templates
from pycycle.template_fit import TemplateFitter

OUT_DIR = _THIS
RRAB_ZIP = "/astro/store/shire/pferguso/software/Multiband-templates/RRab_normalized.zip"
RRC_ZIP = "/astro/store/shire/pferguso/software/Multiband-templates/RRc_normalized.zip"


# ---------------------------------------------------------------------------
# Light curve simulator (shared format with analyze.simulate_lc)
# ---------------------------------------------------------------------------

def simulate_lc(template, period_days, n_obs=30, noise=0.05, t_span=180.0,
                seed=1, mag_base=20.0, amplitude=1.0):
    rng = np.random.default_rng(seed)
    bands = template.bands
    per_band = n_obs // len(bands)
    extras = n_obs - per_band * len(bands)
    counts = [per_band + (1 if i < extras else 0) for i in range(len(bands))]
    t_all, mag_all, err_all, nam_all = [], [], [], []
    for i, b in enumerate(bands):
        t = rng.uniform(0, t_span, counts[i])
        ph = (t / period_days) % 1.0
        idx = (ph * template.n_phase).astype(int) % template.n_phase
        gamma_b = template.gamma[i, idx]
        mag = mag_base + amplitude * gamma_b + rng.normal(0, noise, size=counts[i])
        t_all.append(t)
        mag_all.append(mag)
        err_all.append(np.full(counts[i], noise))
        nam_all.extend([b] * counts[i])
    t = np.concatenate(t_all)
    mag = np.concatenate(mag_all)
    err = np.concatenate(err_all)
    filts = np.array(nam_all)
    order = np.argsort(t)
    return t[order], mag[order], err[order], filts[order]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Module-level globals populated by the pool initialiser so we don't have to
# pickle the entire template list per task.
_WORKER_TEMPLATES = None


def _init_worker(templates):
    global _WORKER_TEMPLATES
    _WORKER_TEMPLATES = templates


def _fit_one(args):
    """Run TemplateFitter against a single template; return (idx, rss_min, p_at_min)."""
    template_idx, hjd, mag, err, filts, pmin, pmax, dphi, n_newton, n_start = args
    tpl = _WORKER_TEMPLATES[template_idx]
    obs_bands = set(np.asarray(filts).tolist())
    if not obs_bands.issubset(set(tpl.bands)):
        return template_idx, np.inf, np.nan
    try:
        fitter = TemplateFitter(tpl, n_newton=n_newton, n_start=n_start)
        with contextlib.redirect_stdout(io.StringIO()):
            res = fitter.fit(hjd, mag, err, filts,
                             pmin=pmin, pmax=pmax, dphi=dphi)
        k = int(np.argmin(res.rss))
        return template_idx, float(res.rss[k]), float(res.periods[k])
    except Exception:
        return template_idx, np.inf, np.nan


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_class(zip_path, label, medoids_by_k, *, n_trials=16, n_workers=32,
              seed=2026, pmin=0.3, pmax=0.9, dphi=0.05,
              n_newton=3, n_start=4, log=print):
    log(f"\n=== Parallel recovery: {label} ===", flush=True)
    t0 = time.time()
    templates = load_multiband_templates(zip_path)
    log(f"  loaded {len(templates)} templates in {time.time()-t0:.1f}s", flush=True)
    N = len(templates)

    p_range = (0.4, 0.8) if label.startswith("RRab") else (0.25, 0.45)
    rng = np.random.default_rng(seed)

    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(templates,)) as pool:
        for trial in range(n_trials):
            truth_i = int(rng.integers(N))
            period_true = float(rng.uniform(*p_range))
            tpl_true = templates[truth_i]
            hjd, mag, err, filts = simulate_lc(
                tpl_true, period_days=period_true,
                n_obs=30, noise=0.05, seed=trial + 100)

            tasks = [(i, hjd, mag, err, filts, pmin, pmax, dphi, n_newton, n_start)
                     for i in range(N)]
            t_start = time.time()
            results = pool.map(_fit_one, tasks)
            dt = time.time() - t_start

            rss_arr = np.array([r[1] for r in results])
            p_arr = np.array([r[2] for r in results])
            full_best = int(np.argmin(rss_arr))
            rss_full = float(rss_arr[full_best])
            p_full = float(p_arr[full_best])

            row = dict(
                trial=trial,
                truth_idx=truth_i,
                truth_name=tpl_true.name,
                period_true=period_true,
                rss_full=rss_full,
                p_full=p_full,
                k_full=full_best,
                dt_parallel=dt,
                rss_per_template=rss_arr.copy(),
                p_per_template=p_arr.copy(),
            )

            for k, meds in medoids_by_k.items():
                meds = np.asarray(meds)
                sub_rss = rss_arr[meds]
                sub_p = p_arr[meds]
                local_best = int(np.argmin(sub_rss))
                row[f"rss_k{k}"] = float(sub_rss[local_best])
                row[f"p_k{k}"] = float(sub_p[local_best])
                row[f"k_pick_k{k}"] = int(meds[local_best])
                row[f"frac_excess_k{k}"] = (
                    (sub_rss[local_best] - rss_full) / rss_full
                    if rss_full > 0 else np.nan
                )
                row[f"truth_in_set_k{k}"] = bool(truth_i in meds.tolist())

            log(f"  trial {trial:>2}: truth={truth_i:>3} ({tpl_true.name[:30]:<30s}) "
                f"P_true={period_true:.3f} P_full={p_full:.3f} "
                f"rss_full={rss_full:.3f}  dt={dt:.2f}s "
                + " ".join(f"k{k}={row[f'frac_excess_k{k}']*100:+.1f}%"
                           for k in medoids_by_k),
                flush=True)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=16)
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--pmin", type=float, default=0.3)
    ap.add_argument("--pmax", type=float, default=0.9)
    ap.add_argument("--dphi", type=float, default=0.05)
    ap.add_argument("--n-newton", type=int, default=3)
    ap.add_argument("--n-start", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "recovery_parallel.pkl"))
    ap.add_argument("--only", choices=["RRab", "RRc", "both"], default="both")
    args = ap.parse_args()

    with open(os.path.join(OUT_DIR, "medoid_indices.pkl"), "rb") as fh:
        rec = pickle.load(fh)

    out = {}
    if args.only in ("RRab", "both"):
        out["RRab"] = run_class(
            RRAB_ZIP, "RRab",
            {int(k): v for k, v in rec["RRab"]["medoids_by_k"].items()},
            n_trials=args.n_trials, n_workers=args.n_workers,
            pmin=args.pmin, pmax=args.pmax, dphi=args.dphi,
            n_newton=args.n_newton, n_start=args.n_start,
        )
        # Checkpoint after RRab in case RRc takes a long time.
        with open(args.out, "wb") as fh:
            pickle.dump(out, fh)
        print(f"  checkpointed -> {args.out}", flush=True)
    if args.only in ("RRc", "both"):
        out["RRc"] = run_class(
            RRC_ZIP, "RRc",
            {int(k): v for k, v in rec["RRc"]["medoids_by_k"].items()},
            n_trials=args.n_trials, n_workers=args.n_workers,
            pmin=args.pmin, pmax=args.pmax, dphi=args.dphi,
            n_newton=args.n_newton, n_start=args.n_start,
        )
        with open(args.out, "wb") as fh:
            pickle.dump(out, fh)
        print(f"  checkpointed -> {args.out}", flush=True)

    print("done.")


if __name__ == "__main__":
    main()
