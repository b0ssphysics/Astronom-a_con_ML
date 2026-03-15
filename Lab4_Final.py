# # Lab 3 — HDBSCAN Parameter Exploration, Silhouette Analysis & Soft Membership Probabilities
# ## M41 (NGC 2287) — Gaia DR3
# 
# 

# %% [markdown]
# ## Section 1 — Dependencies and Data Loading
# 
# Same dataset as Labs 1–3. We add an optional **parallax SNR cut** (`parallax / parallax_error > PLX_SNR_CUT`) to control data quality before clustering.  
# Set `PLX_SNR_CUT = 0` to skip the cut entirely.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import hdbscan
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
file = "/home/fernando-x390/Desktop/Astro with ML/LAb3/M41_query_result_full.csv"
df   = pd.read_csv(file)
print(f"Stars loaded from file          : {len(df)}")

# ── Optional parallax quality cut ─────────────────────────────────────────────
PLX_SNR_CUT = 5

if PLX_SNR_CUT > 0:
    if 'parallax_over_error' in df.columns:
        df = df[df['parallax_over_error'] > PLX_SNR_CUT].copy()
    else:
        df = df[(df['parallax'] / df['parallax_error'].abs()) > PLX_SNR_CUT].copy()
    print(f"After parallax SNR > {PLX_SNR_CUT} cut        : {len(df)} stars")
else:
    print("No parallax SNR cut applied")

# ── Select features, drop NaN rows ────────────────────────────────────────────
features = ['pmra', 'pmdec', 'parallax']
photo    = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']

df_clean = df[features + photo].dropna().copy()
df_clean.reset_index(drop=True, inplace=True)
print(f"Stars after dropping NaN rows   : {len(df_clean)}")

df_clean[features + photo].describe()

# %% [markdown]
# ## Section 2 — Feature Preparation and Standardisation

X_raw = df_clean[features].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Feature statistics after scaling (should be ≈0 mean, ≈1 std):")
for i, f in enumerate(features):
    print(f"  {f:12s}  mean={X_scaled[:, i].mean():+.4f}   std={X_scaled[:, i].std():.4f}")

# %% [markdown]
# ## Section 3 — HDBSCAN Parameter Guide and Grid Search

M41_REF = np.array([-4.37, -1.34, 1.60])   # [pmra, pmdec, parallax]

mcs_values = [50, 80, 100, 150, 200]
ms_values  = [10, 30, 50, 70, 100]

grid_results = []

print(f"{'MCS':>5}  {'MS':>5}  {'N_clust':>7}  {'N_noise':>8}  {'M41_size':>9}  {'Silhouette':>11}")
print("-" * 55)

for mcs in mcs_values:
    for ms in ms_values:
        if ms > mcs:
            continue

        cl  = hdbscan.HDBSCAN(
            min_cluster_size         = mcs,
            min_samples              = ms,
            metric                   = 'euclidean',
            cluster_selection_method = 'eom',
            allow_single_cluster     = True,
        )
        lbl = cl.fit_predict(X_scaled)

        unique  = np.unique(lbl)
        n_cl    = len(unique[unique >= 0])
        n_ns    = np.sum(lbl == -1)

        best_lbl_g, best_d = -1, np.inf
        for c in unique[unique >= 0]:
            cen  = X_raw[lbl == c].mean(axis=0)
            dist = np.linalg.norm(cen - M41_REF)
            if dist < best_d:
                best_d, best_lbl_g = dist, c
        m41_size = np.sum(lbl == best_lbl_g) if best_lbl_g >= 0 else 0

        mask_v = lbl >= 0
        if n_cl >= 2 and mask_v.sum() >= 2:
            sil = silhouette_score(X_scaled[mask_v], lbl[mask_v])
            sil_str = f"{sil:>11.4f}"
        else:
            sil = np.nan
            sil_str = f"{'N/A':>11}"

        grid_results.append(dict(mcs=mcs, ms=ms, n_clusters=n_cl,
                                 n_noise=n_ns, m41_size=m41_size, silhouette=sil))
        print(f"{mcs:>5}  {ms:>5}  {n_cl:>7}  {n_ns:>8}  {m41_size:>9}  {sil_str}")

grid_df = pd.DataFrame(grid_results)

# %% [markdown]
# ## Section 4 — Final HDBSCAN Run with Chosen Parameters

MCS = 100
MS  = 70
CSE = 0.0

clusterer = hdbscan.HDBSCAN(
    min_cluster_size         = MCS,
    min_samples              = MS,
    metric                   = 'euclidean',
    cluster_selection_method = 'eom',
    cluster_selection_epsilon= CSE,
    allow_single_cluster     = True,
    prediction_data          = True,
)

labels     = clusterer.fit_predict(X_scaled)
unique_lbl = np.unique(labels)
n_clusters = len(unique_lbl[unique_lbl >= 0])
n_noise    = np.sum(labels == -1)

print(f"Parameters used  →  MCS={MCS}, MS={MS}, epsilon={CSE}")
print(f"Clusters found               : {n_clusters}")
print(f"Noise points (field)         : {n_noise}  ({n_noise/len(labels)*100:.1f}%)")

for cl in unique_lbl[unique_lbl >= 0]:
    mk = labels == cl
    print(f"\n  Cluster {cl}  ({np.sum(mk)} stars)")
    print(f"    mean pmra     = {X_raw[mk,0].mean():+.3f} mas/yr")
    print(f"    mean pmdec    = {X_raw[mk,1].mean():+.3f} mas/yr")
    print(f"    mean parallax = {X_raw[mk,2].mean():+.3f} mas")

best_lbl, best_dist = -1, np.inf
for cl in unique_lbl[unique_lbl >= 0]:
    cen  = X_raw[labels == cl].mean(axis=0)
    dist = np.linalg.norm(cen - M41_REF)
    if dist < best_dist:
        best_dist, best_lbl = dist, cl

is_m41   = labels == best_lbl
is_noise = labels == -1
print(f"\nM41 identified as cluster label : {best_lbl}")
print(f"M41 hard members                : {np.sum(is_m41)}")