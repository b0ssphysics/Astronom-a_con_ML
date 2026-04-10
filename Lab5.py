# =============================================================================
# Lab 5 — Blind Search for Open Clusters within 200 pc of the Sun
# Gaia DR3 | Continuation from Lab 4
#
# Known sanity-check targets:
#   Hyades      (Mel 25)  d ~46 pc   pmra ~+100  pmdec ~-27
#   Coma Ber    (Mel 111) d ~86 pc   pmra ~ -12  pmdec ~ +9
#   Pleiades    (Mel 22)  d ~136 pc  pmra ~ +20  pmdec ~-45
#
# Targets from Hunt & Ryu 2023:
#   HSC 396  d=99.14 pc  RA=317.19  Dec=-3.67   pmra=+21.8   pmdec=-8.72
#   HSC 759  d=95.96 pc  RA=225.15  Dec=+59.39  pmra=-16.19  pmdec=-3.64
# =============================================================================

import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyarrow.parquet as pq

from astropy.coordinates import SkyCoord
import astropy.units as u

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import hdbscan as hd

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
data_path  = '/home/fernando-x390/Desktop/Astro with ML/Lab5/GDR3_200pc.parquet'
cols_needed = [
    'source_id', 'ra', 'dec', 'pmra', 'pmdec', 'parallax',
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
]

df = pd.read_parquet(data_path, columns=cols_needed)
print(f'Total stars loaded : {len(df)}')

# Downcast to float32 — halves memory, negligible precision loss for astrometry
for col in ['ra', 'dec', 'pmra', 'pmdec', 'parallax',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']:
    if col in df.columns:
        df[col] = df[col].astype('float32')

print(f'Memory after downcast : {df.memory_usage(deep=True).sum()/1e9:.2f} GB')


# =============================================================================
# 2. GALACTIC COORDINATES  (ICRS → Galactic via IAU rotation)
# =============================================================================
coords   = SkyCoord(ra=df['ra'].values * u.deg,
                    dec=df['dec'].values * u.deg,
                    frame='icrs')
df['l']  = coords.galactic.l.deg.astype('float32')   # 0° → 360°
df['b']  = coords.galactic.b.deg.astype('float32')   # −90° → +90°

print(f'l range : {df["l"].min():.2f} → {df["l"].max():.2f} deg')
print(f'b range : {df["b"].min():.2f} → {df["b"].max():.2f} deg')


# =============================================================================
# 3. 8 × 8 GALACTIC SKY GRID
# =============================================================================
N      = 8
l_bins = np.linspace(0,   360, N + 1)   # 45° wide per cell
b_bins = np.linspace(-90,  90, N + 1)   # 22.5° wide per cell

df['grid_col'] = pd.cut(df['l'], bins=l_bins,
                         labels=False, include_lowest=True).astype('int16')
df['grid_row'] = pd.cut(df['b'], bins=b_bins,
                         labels=False, include_lowest=True).astype('int16')
df['zone_id']  = (df['grid_row'] * N + df['grid_col']).astype('int16')

zone_counts = df.groupby('zone_id').size()
print(f'\nl bin width : {l_bins[1]-l_bins[0]:.1f} deg')
print(f'b bin width : {b_bins[1]-b_bins[0]:.1f} deg')
print(f'Zones occupied : {len(zone_counts)} / {N*N}')
print('Stars per zone:\n', zone_counts.describe().round(0))

# Report zones for known clusters
for name, l_t, b_t in [('Hyades',   180.0, -22.0),
                        ('Coma Ber', 221.0,  84.0),
                        ('Pleiades', 167.0, -24.0)]:
    col = max(0, min(N-1, np.searchsorted(l_bins, l_t, side='right') - 1))
    row = max(0, min(N-1, np.searchsorted(b_bins, b_t, side='right') - 1))
    zid = row * N + col
    print(f'{name:12s}: zone {zid:>3d}  '
          f'l=[{l_bins[col]:.0f},{l_bins[col+1]:.0f})  '
          f'b=[{b_bins[row]:.0f},{b_bins[row+1]:.0f})  '
          f'N={zone_counts.get(zid, 0)}')


# =============================================================================
# 4. GALACTIC SKY MAP  (Fig 01)
# =============================================================================
fig = plt.figure(figsize=(14, 7))
ax  = fig.add_subplot(111, projection='mollweide')

l_rad = np.deg2rad(df['l'].values.astype(float))
l_rad[l_rad > np.pi] -= 2 * np.pi
b_rad = np.deg2rad(df['b'].values.astype(float))

scatter = ax.scatter(
    l_rad, b_rad,
    c=df['phot_g_mean_mag'].values.astype(float),
    cmap='plasma_r', s=0.3, alpha=0.25,
    vmin=float(df['phot_g_mean_mag'].quantile(0.05)),
    vmax=float(df['phot_g_mean_mag'].quantile(0.95)),
    rasterized=True
)
ax.set_xticklabels([
    '150°','120°','90°','60°','30°','GC\n0°',
    '330°','300°','270°','240°','210°'
], fontsize=8)
ax.set_ylabel('Galactic latitude  $b$', fontsize=11)
ax.set_title('GAIA DR3 — Galactic sky map  (200 pc)', fontsize=12)
ax.grid(True, color='white', lw=0.4, alpha=0.5)

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.2, 0.06, 0.6, 0.03])
cbar    = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label('$G$ magnitude', fontsize=11, labelpad=6)
cbar.ax.tick_params(labelsize=9)
fig.text(0.5, 0.11, 'Galactic longitude  $l$',
         ha='center', va='bottom', fontsize=11)

plt.savefig('Fig01_galactic_skymap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: Fig01_galactic_skymap.png')


# =============================================================================
# 5. HDBSCAN PER ZONE
#
# Parameter rationale
# -------------------
# MIN_CLUSTER_SIZE = 50  : Coma Ber has ~200 true members; after field
#                          dilution a floor of 50 ensures we don't miss it.
# MIN_SAMPLES      =  5  : Low value keeps the density estimate sensitive to
#                          tight kinematic clumps like Hyades.
# No silhouette filter in the loop — sil is recorded for diagnostics only.
# Diameter filter (≤30 pc) in the candidate step rejects field artefacts.
# =============================================================================
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES      =  5
MAX_SIL_SAMPLE   = 2000
MAX_ZONE_STARS   = 50_000

features = ['pmra', 'pmdec', 'parallax',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']


def smart_subsample(df_z, max_stars, random_state=42):
    """Keep ALL nearby stars (parallax ≥ 5 mas, d ≤ 200 pc) intact;
    subsample only the distant background to stay within the memory cap."""
    near   = df_z[df_z['parallax'] >= 5].copy()
    far    = df_z[df_z['parallax'] <  5].copy()
    budget = max_stars - len(near)
    if budget <= 0:
        return near.reset_index(drop=True)
    if len(far) > budget:
        far = far.sample(budget, random_state=random_state)
    return pd.concat([near, far]).reset_index(drop=True)


zone_results = {}

print(f"\n{'Zone':>6}  {'Lbl':>5}  {'N':>7}  {'N_cl':>5}  {'Cl_size':>8}  {'Sil':>8}")
print('─' * 55)

for zid in sorted(df['zone_id'].unique()):

    df_z = df[df['zone_id'] == zid].copy()
    if len(df_z) < MIN_CLUSTER_SIZE:
        continue

    df_z = df_z.dropna(subset=features).reset_index(drop=True)
    if len(df_z) < MIN_CLUSTER_SIZE:
        continue

    if len(df_z) > MAX_ZONE_STARS:
        df_z = smart_subsample(df_z, MAX_ZONE_STARS)

    X = RobustScaler().fit_transform(df_z[features].values.astype(float))

    cl = hd.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom',
        allow_single_cluster=False,
        core_dist_n_jobs=4
    )
    labels = cl.fit_predict(X)
    df_z['hdb_label'] = labels

    n_clusters = len(np.unique(labels[labels >= 0]))
    if n_clusters == 0:
        del X, df_z; gc.collect()
        continue

    for lbl in np.unique(labels[labels >= 0]):
        cl_mask = labels == lbl
        cl_size = cl_mask.sum()

        cl_idx  = np.where(cl_mask)[0]
        fld_idx = np.where(~cl_mask)[0]
        n_cl_s  = min(len(cl_idx),  MAX_SIL_SAMPLE // 2)
        n_fld_s = min(len(fld_idx), MAX_SIL_SAMPLE // 2)

        sil = np.nan
        if n_cl_s >= 2 and n_fld_s >= 2:
            sample_idx = np.concatenate([
                np.random.choice(cl_idx,  n_cl_s,  replace=False),
                np.random.choice(fld_idx, n_fld_s, replace=False),
            ])
            sil_labels = np.where(cl_mask, 0, 1)
            try:
                sil = silhouette_score(
                    X[sample_idx], sil_labels[sample_idx])
            except Exception:
                sil = np.nan

        # Store ALL clusters — no sil rejection here
        zone_results[(zid, int(lbl))] = dict(
            df         = df_z,
            labels     = labels,
            best_lbl   = int(lbl),
            cl_mask    = cl_mask,
            n_clusters = n_clusters,
            n_noise    = int(np.sum(labels == -1)),
            cl_size    = cl_size,
            sil        = sil,
        )
        sil_str = f'{sil:.4f}' if not np.isnan(sil) else '   N/A'
        print(f'{zid:>6}  {lbl:>5}  {len(df_z):>7}  {n_clusters:>5}  '
              f'{cl_size:>8}  {sil_str:>8}')

    del X, df_z; gc.collect()

print(f'\nTotal clusters stored: {len(zone_results)}')

# ── Sanity check: Hyades, Coma, Pleiades ─────────────────────────────────────
for name, l_t, b_t, pmra_t, pmdec_t, d_t in [
        ('Hyades',   180.0, -22.0, 100.0, -27.0,  46.0),
        ('Coma Ber', 221.0,  84.0, -12.0,   9.0,  86.0),
        ('Pleiades', 167.0, -24.0,  20.0, -45.0, 136.0)]:
    col = max(0, min(N-1, np.searchsorted(l_bins, l_t, side='right') - 1))
    row = max(0, min(N-1, np.searchsorted(b_bins, b_t, side='right') - 1))
    zid = row * N + col
    hits = {k: v for k, v in zone_results.items() if k[0] == zid}
    if not hits:
        print(f'{name}: zone {zid} → no cluster found')
        continue
    print(f'\n{name} (d={d_t}pc pmra={pmra_t} pmdec={pmdec_t}) → zone {zid} '
          f'({len(hits)} cluster(s)):')
    for (z, lbl), res in hits.items():
        df_cl   = res['df'][res['cl_mask']]
        dpm     = np.sqrt((df_cl['pmra'].median()  - pmra_t)**2 +
                          (df_cl['pmdec'].median() - pmdec_t)**2)
        d_found = 1000 / df_cl['parallax'].median()
        match   = '✓ MATCH' if dpm < 8.0 and abs(d_found - d_t) < 20 else '✗'
        sil_str = f'{res["sil"]:.4f}' if not np.isnan(res['sil']) else 'N/A'
        print(f'  lbl={lbl}  N={res["cl_size"]:>5}  '
              f'pmra={df_cl["pmra"].median():>7.2f}  '
              f'pmdec={df_cl["pmdec"].median():>7.2f}  '
              f'd={d_found:>6.1f}pc  Δpm={dpm:.1f}  sil={sil_str}  {match}')


# =============================================================================
# 6. CANDIDATE FILTERING  (diameter only — sil already informational)
# =============================================================================
DIAM_MAX   = 30.0
candidates = []

print(f"\n{'Zone':>6}  {'Lbl':>5}  {'N':>7}  {'Sil':>7}  "
      f"{'Diam_pc':>8}  {'d_pc':>8}  Status")
print('─' * 65)

for (zid, lbl), res in zone_results.items():
    df_z    = res['df']
    cl_mask = res['cl_mask']
    df_cl   = df_z[cl_mask].copy()

    plx_mean = df_cl['parallax'].mean()
    if plx_mean <= 0:
        continue

    d_mean_pc = 1000.0 / plx_mean
    sep_deg   = np.sqrt(
        (df_cl['ra'].values  - np.median(df_cl['ra'].values))**2 +
        (df_cl['dec'].values - np.median(df_cl['dec'].values))**2
    )
    diam_pc = 2 * np.deg2rad(np.percentile(sep_deg, 90)) * d_mean_pc

    sil_str = f'{res["sil"]:.4f}' if not np.isnan(res['sil']) else '   N/A'

    if diam_pc > DIAM_MAX:
        print(f'{zid:>6}  {lbl:>5}  {len(df_cl):>7}  {sil_str:>7}  '
              f'{diam_pc:>8.1f}  {d_mean_pc:>8.1f}  REJECTED diam')
        continue

    candidates.append(dict(
        zone_id=zid, lbl=lbl, df_cl=df_cl, df_z=df_z, cl_mask=cl_mask,
        sil=res['sil'], diam_pc=diam_pc, d_mean_pc=d_mean_pc, N=len(df_cl)
    ))
    print(f'{zid:>6}  {lbl:>5}  {len(df_cl):>7}  {sil_str:>7}  '
          f'{diam_pc:>8.1f}  {d_mean_pc:>8.1f}  *** CANDIDATE ***')

print(f'\nCandidates after filters: {len(candidates)}')


# =============================================================================
# 7. CANDIDATE SKY MAP  (Fig 02)
# =============================================================================
fig = plt.figure(figsize=(14, 7))
ax  = fig.add_subplot(111, projection='mollweide')

l_rad = np.deg2rad(df['l'].values.astype(float))
l_rad[l_rad > np.pi] -= 2 * np.pi
b_rad = np.deg2rad(df['b'].values.astype(float))
ax.scatter(l_rad, b_rad, c='white', s=0.1, alpha=0.06, rasterized=True)

colors = plt.cm.tab20(np.linspace(0, 1, max(len(candidates), 1)))

for i, cand in enumerate(candidates):
    df_cl = cand['df_cl']
    l_cl  = np.deg2rad(df_cl['l'].values.astype(float))
    l_cl[l_cl > np.pi] -= 2 * np.pi
    b_cl  = np.deg2rad(df_cl['b'].values.astype(float))

    ax.scatter(l_cl, b_cl, s=6, alpha=0.8,
               color=colors[i % 20], zorder=3)

    l_cen = np.median(l_cl)
    b_cen = np.median(b_cl)
    ax.text(l_cen, b_cen,
            f'z{cand["zone_id"]}l{cand["lbl"]}\n'
            f'{cand["diam_pc"]:.1f}pc  {cand["d_mean_pc"]:.0f}pc',
            fontsize=5.5, color='yellow', ha='center', va='bottom',
            fontweight='bold', zorder=4)

ax.set_xticklabels([
    '150°','120°','90°','60°','30°','GC\n0°',
    '330°','300°','270°','240°','210°'
], fontsize=8)
ax.set_ylabel('Galactic latitude  $b$', fontsize=11)
ax.set_title('Candidate clusters — Galactic sky map', fontsize=12)
ax.grid(True, color='gray', lw=0.4, alpha=0.5)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.tick_params(colors='white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
fig.subplots_adjust(bottom=0.15)
fig.text(0.5, 0.06, 'Galactic longitude  $l$',
         ha='center', fontsize=11, color='white')

plt.savefig('Fig02_candidates_skymap.png', dpi=150,
            bbox_inches='tight', facecolor='black')
plt.show()
print('Saved: Fig02_candidates_skymap.png')


# =============================================================================
# 8. PER-CANDIDATE DIAGNOSTIC PLOTS  (VPD · CMD · Distance histogram)
# =============================================================================
for cand in candidates:
    zid     = cand['zone_id']
    lbl     = cand['lbl']
    df_cl   = cand['df_cl']
    df_z    = cand['df_z']
    cl_mask = cand['cl_mask']
    d_pc    = cand['d_mean_pc']
    tag     = f'z{zid:02d}_l{lbl}'
    sil_str = f'{cand["sil"]:.3f}' if not np.isnan(cand['sil']) else 'N/A'

    print(f'\nZone {zid}  lbl={lbl}  N={cand["N"]}  '
          f'd={d_pc:.1f}pc  diam={cand["diam_pc"]:.1f}pc  sil={sil_str}')
    print(f'  pmra  median : {df_cl["pmra"].median():+.2f} mas/yr')
    print(f'  pmdec median : {df_cl["pmdec"].median():+.2f} mas/yr')
    print(f'  parallax med : {df_cl["parallax"].median():.3f} mas')

    # ── Figure A: VPD ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_z['pmra'][~cl_mask], df_z['pmdec'][~cl_mask],
               s=3, c='lightgrey', alpha=0.3, label='Field', zorder=1)
    ax.scatter(df_cl['pmra'], df_cl['pmdec'],
               s=20, c='crimson', alpha=0.85,
               edgecolors='black', linewidths=0.2,
               label=f'Cluster  N={cand["N"]}', zorder=3)
    ax.set_xlabel(r'$\mu_{\alpha}\cos\delta$  (mas yr$^{-1}$)', fontsize=11)
    ax.set_ylabel(r'$\mu_{\delta}$  (mas yr$^{-1}$)', fontsize=11)
    ax.set_title(f'VPD  Zone {zid}  lbl={lbl}  d={d_pc:.1f} pc', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(linestyle=':', alpha=0.4)
    pad = 5.0
    ax.set_xlim(df_cl['pmra'].min()  - pad, df_cl['pmra'].max()  + pad)
    ax.set_ylim(df_cl['pmdec'].min() - pad, df_cl['pmdec'].max() + pad)
    plt.tight_layout()
    plt.savefig(f'FigA_{tag}_VPD.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved FigA_{tag}_VPD.png')

    # ── Figure B: CMD ────────────────────────────────────────────────────────
    photo    = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
    df_cl_ph = df_cl.dropna(subset=photo)
    df_z_ph  = df_z.dropna(subset=photo)

    if len(df_cl_ph) > 3:
        col_cl  = df_cl_ph['phot_bp_mean_mag'] - df_cl_ph['phot_rp_mean_mag']
        mag_cl  = df_cl_ph['phot_g_mean_mag']
        col_fld = df_z_ph['phot_bp_mean_mag']  - df_z_ph['phot_rp_mean_mag']
        mag_fld = df_z_ph['phot_g_mean_mag']
        mu0     = 5 * np.log10(d_pc) - 5
        abs_mag = mag_cl - mu0

        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        axes[0].scatter(col_fld, mag_fld, s=2, c='lightgrey',
                        alpha=0.2, zorder=1, label='Field')
        axes[0].scatter(col_cl,  mag_cl,  s=18, c='crimson',
                        alpha=0.85, edgecolors='black', linewidths=0.2,
                        zorder=3, label=f'Cluster N={len(df_cl_ph)}')
        axes[0].invert_yaxis()
        axes[0].set_xlabel(r'$G_{\rm BP} - G_{\rm RP}$  (mag)', fontsize=11)
        axes[0].set_ylabel(r'$G$ apparent  (mag)', fontsize=11)
        axes[0].set_title('CMD apparent', fontsize=11)
        axes[0].set_xlim(-0.5, 3.5)
        axes[0].legend(fontsize=9)
        axes[0].grid(linestyle=':', alpha=0.4)

        axes[1].scatter(col_cl, abs_mag, s=18, c='steelblue',
                        alpha=0.85, edgecolors='black', linewidths=0.2,
                        zorder=3, label=f'd={d_pc:.1f} pc')
        axes[1].invert_yaxis()
        axes[1].set_xlabel(r'$G_{\rm BP} - G_{\rm RP}$  (mag)', fontsize=11)
        axes[1].set_ylabel(r'$M_G$  (mag)', fontsize=11)
        axes[1].set_title('CMD absolute', fontsize=11)
        axes[1].set_xlim(-0.5, 3.5)
        axes[1].legend(fontsize=9)
        axes[1].grid(linestyle=':', alpha=0.4)

        fig.suptitle(f'CMD  Zone {zid}  lbl={lbl}  '
                     f'N={len(df_cl_ph)}  d={d_pc:.1f} pc', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'FigB_{tag}_CMD.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f'  Saved FigB_{tag}_CMD.png')

    # ── Figure C: distance histogram ─────────────────────────────────────────
    d_cl  = 1000.0 / df_cl['parallax'][df_cl['parallax'] > 0]
    d_fld = 1000.0 / df_z['parallax'][df_z['parallax']  > 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d_fld, bins=50, density=True, color='lightgrey',
            alpha=0.7, label='Field', edgecolor='grey')
    ax.hist(d_cl,  bins=20, density=True, color='crimson', alpha=0.85,
            label=f'Cluster N={len(d_cl)}',
            edgecolor='black', linewidth=0.5)
    ax.axvline(d_pc, color='black', linestyle='--', lw=1.5,
               label=f'd={d_pc:.1f} pc')
    ax.set_xlabel('Distance (pc)', fontsize=11)
    ax.set_ylabel('Normalised density', fontsize=11)
    ax.set_title(f'Distance histogram  Zone {zid}  lbl={lbl}', fontsize=11)
    pad_pc = 10.0
    ax.set_xlim(d_cl.min() - pad_pc, d_cl.max() + pad_pc)
    ax.legend(fontsize=9)
    ax.grid(linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'FigC_{tag}_distance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved FigC_{tag}_distance.png')


# =============================================================================
# 9. TARGETED SEARCH — HSC 396 & HSC 759  (using galactic coords)
# =============================================================================
hsc_targets = {
    'HSC 396': dict(ra=317.19, dec= -3.67, pmra= 21.80, pmdec= -8.72, d=99.14),
    'HSC 759': dict(ra=225.15, dec= 59.39, pmra=-16.19, pmdec= -3.64, d=95.96),
}

PM_WINDOW  = 5.0
PLX_WINDOW = 0.15

for name, tgt in hsc_targets.items():
    # Convert target RA/Dec to galactic
    c_tgt = SkyCoord(ra=tgt['ra']*u.deg, dec=tgt['dec']*u.deg, frame='icrs')
    l_tgt = float(c_tgt.galactic.l.deg)
    b_tgt = float(c_tgt.galactic.b.deg)
    plx_tgt = 1000.0 / tgt['d']

    col = max(0, min(N-1, np.searchsorted(l_bins, l_tgt, side='right') - 1))
    row = max(0, min(N-1, np.searchsorted(b_bins, b_tgt, side='right') - 1))
    zid = row * N + col

    print(f'\n{name}  d={tgt["d"]}pc  pmra={tgt["pmra"]}  pmdec={tgt["pmdec"]}')
    print(f'  Galactic: l={l_tgt:.2f}  b={b_tgt:.2f}')
    print(f'  Zone {zid}  '
          f'l=[{l_bins[col]:.0f},{l_bins[col+1]:.0f})  '
          f'b=[{b_bins[row]:.0f},{b_bins[row+1]:.0f})  '
          f'N_zone={zone_counts.get(zid, 0)}')

    # Check if blind search found anything in this zone
    hits = {k: v for k, v in zone_results.items() if k[0] == zid}
    if hits:
        for (z, lbl), res in hits.items():
            df_cl   = res['df'][res['cl_mask']]
            dpm     = np.sqrt((df_cl['pmra'].median()  - tgt['pmra'])**2 +
                              (df_cl['pmdec'].median() - tgt['pmdec'])**2)
            d_found = 1000 / df_cl['parallax'].median()
            match   = '✓ MATCH' if dpm < 5 and abs(d_found - tgt['d']) < 15 else '✗'
            print(f'  Blind: lbl={lbl}  N={res["cl_size"]}  '
                  f'pmra={df_cl["pmra"].median():.2f}  '
                  f'pmdec={df_cl["pmdec"].median():.2f}  '
                  f'd={d_found:.1f}pc  Δpm={dpm:.1f}  {match}')
    else:
        print(f'  Blind search: no cluster found in zone {zid}')

    # Targeted window search within the zone
    df_zone = df[df['zone_id'] == zid].copy()
    df_zone = df_zone.dropna(subset=features)
    pm_mask = (
        (df_zone['pmra']     > tgt['pmra']  - PM_WINDOW) &
        (df_zone['pmra']     < tgt['pmra']  + PM_WINDOW) &
        (df_zone['pmdec']    > tgt['pmdec'] - PM_WINDOW) &
        (df_zone['pmdec']    < tgt['pmdec'] + PM_WINDOW) &
        (df_zone['parallax'] > plx_tgt      - PLX_WINDOW) &
        (df_zone['parallax'] < plx_tgt      + PLX_WINDOW)
    )
    df_win = df_zone[pm_mask].copy().reset_index(drop=True)
    print(f'  Targeted window: {len(df_win)} stars  '
          f'(pm±{PM_WINDOW}, plx±{PLX_WINDOW})')
    if len(df_win) >= 10:
        print(f'  → Possible detection — run dedicated HDBSCAN on this window')