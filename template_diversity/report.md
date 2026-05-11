# Multiband RRL Template Diversity Analysis
Templates: Baeza-Villagra et al. 2025, DECam griz, normalized.

## Headline recommendations
- **RRab:** 10 medoid templates capture the bulk of shape variance; 20 is essentially redundant with the full library.
- **RRc:** 5 medoid templates are sufficient; the population is more nearly sinusoidal so PCA is very low-rank.

## RRab
- Loaded 136 templates on band set [np.str_('g'), np.str_('i'), np.str_('r'), np.str_('z')].
- PCA: 95% of variance in top 6 components, 99% in top 19.
- Pairwise L2 (aligned): median=2.456, max=7.161.
- Mean intra-cluster distance:
    - k=2: 1.400
    - k=5: 0.835
    - k=10: 0.704
    - k=20: 0.578
- Recovery on simulated LSST-like LCs (n=8 trials, fractional excess RSS vs full library):
    - k=5: median=61.9%, p90=165.3%, max=171.8%
    - k=10: median=50.9%, p90=105.6%, max=202.4%
    - k=20: median=40.5%, p90=69.0%, max=107.6%

## RRc
- Loaded 144 templates on band set [np.str_('g'), np.str_('i'), np.str_('r'), np.str_('z')].
- PCA: 95% of variance in top 11 components, 99% in top 19.
- Pairwise L2 (aligned): median=1.632, max=4.113.
- Mean intra-cluster distance:
    - k=2: 1.037
    - k=5: 0.922
    - k=10: 0.822
    - k=20: 0.688
- Recovery on simulated LSST-like LCs (n=8 trials, fractional excess RSS vs full library):
    - k=5: median=34.5%, p90=152.8%, max=266.6%
    - k=10: median=12.3%, p90=76.7%, max=116.5%
    - k=20: median=18.4%, p90=58.6%, max=59.6%

## Output files
- RRab_dendro.png
- RRab_distmat.png
- RRab_medoids_k10.png
- RRab_medoids_k2.png
- RRab_medoids_k20.png
- RRab_medoids_k5.png
- RRab_pca_scree.png
- RRab_result.pkl
- RRc_dendro.png
- RRc_distmat.png
- RRc_medoids_k10.png
- RRc_medoids_k2.png
- RRc_medoids_k20.png
- RRc_medoids_k5.png
- RRc_pca_scree.png
- RRc_result.pkl
- __pycache__
- analyze.py
- medoid_indices.npz
- medoid_indices.pkl
- recovery.pkl
- recovery_parallel.log
- recovery_parallel.pkl
- recovery_parallel.py
- report.md
- run.log
- run_analyze.log
