"""
Generate addendum_validation.png with numbers exactly matching the addendum text.

Key numbers from the addendum that this figure MUST match:
- 882 residential-scale high-value sales ($800K-$3M RMV)
- 152 land-dominant, 730 building-dominant (all high-value)
- 17 teardowns (sale <= land + 5%), 865 normal
- Teardown median ratio: 0.39, median year 1941
- Normal median ratio: 1.047
- Non-teardown LD: 141, median ratio 1.046; BD: 724, median ratio 1.047
- Mann-Whitney p=0.69
- 97 Viewcrest-like comps (non-teardown, LD, bldg $150K-$500K)
- 13 closest comps (RMV $1.05M-$1.35M, land >=60%)

Filter: ~(YearBuilt >= 2024) to exclude new construction while keeping NaN years.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── Load and filter ──────────────────────────────────────────────────────────
df = pd.read_csv('data/multnomah/multnomah_research_db.csv')

has_rmv = df['RMV'].notna() & (df['RMV'] > 0)
has_sale = df['SalePrice'].notna() & (df['SalePrice'] > 0)
df = df[has_rmv & has_sale].copy()
df = df[~(df['YearBuilt'] >= 2024)]  # exclude new construction, keep NaN years

df['Ratio'] = df['SalePrice'] / df['RMV']
df['LandPct'] = df['RMVLAND'] / df['RMV'] * 100
df['LandDom'] = df['LandPct'] >= 50
df['SaleLandRatio'] = df['SalePrice'] / df['RMVLAND']

# Residential-scale high-value: $800K-$3M RMV
hv = df[(df['RMV'] >= 800_000) & (df['RMV'] < 3_000_000)].copy()
print(f"High-value residential: {len(hv)}")

ld = hv[hv['LandDom']]
bd = hv[~hv['LandDom']]
print(f"  Land-dominant: {len(ld)}")
print(f"  Building-dominant: {len(bd)}")

teardowns = hv[hv['SaleLandRatio'] <= 1.05]
normal = hv[hv['SaleLandRatio'] > 1.05]
print(f"  Teardowns: {len(teardowns)}")
print(f"  Normal: {len(normal)}")
print(f"  Teardown median ratio: {teardowns['Ratio'].median():.3f}")
print(f"  Normal median ratio: {normal['Ratio'].median():.3f}")

nt_ld = normal[normal['LandDom']]
nt_bd = normal[~normal['LandDom']]
print(f"  Non-teardown LD: {len(nt_ld)}, median ratio: {nt_ld['Ratio'].median():.3f}")
print(f"  Non-teardown BD: {len(nt_bd)}, median ratio: {nt_bd['Ratio'].median():.3f}")

u, p = stats.mannwhitneyu(nt_ld['Ratio'], nt_bd['Ratio'], alternative='two-sided')
print(f"  Mann-Whitney p={p:.4f}")

# Viewcrest-like comps
vc_like = nt_ld[(nt_ld['RMVIMPR'] >= 150_000) & (nt_ld['RMVIMPR'] <= 500_000)]
print(f"  Viewcrest-like comps: {len(vc_like)}")

# Closest comps
closest = nt_ld[(nt_ld['RMV'] >= 1_050_000) & (nt_ld['RMV'] <= 1_350_000) & (nt_ld['LandPct'] >= 60)]
print(f"  Closest comps: {len(closest)}")

VIEWCREST_RMV = 1_181_560

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle("Addendum: County-Wide Validation of Viewcrest Price Analysis",
             fontsize=15, fontweight='bold', y=0.98)

# ── Panel 1: High-value scatter (top-left) ──────────────────────────────────
ax = axes[0, 0]

# Plot building-dominant first (background), then land-dominant
ax.scatter(bd['RMV'] / 1000, bd['SalePrice'] / 1000,
           c='#5DA5DA', alpha=0.35, s=20, label=f'Building-dominant (n={len(bd)})')
ax.scatter(ld['RMV'] / 1000, ld['SalePrice'] / 1000,
           c='#F15854', alpha=0.5, s=25, label=f'Land-dominant (n={len(ld)})')

# Viewcrest marker
ax.scatter(VIEWCREST_RMV / 1000, VIEWCREST_RMV / 1000,
           marker='*', c='gold', edgecolors='black', s=300, zorder=10,
           label='Viewcrest est.')

# Reference line
lims = [700, 3100]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='RMV = Sale')
ax.set_xlim(700, 3100)
ax.set_ylim(0, 5500)
ax.set_xlabel('County RMV ($K)')
ax.set_ylabel('Sale Price ($K)')
ax.set_title(f'{len(hv)} Residential Sales with RMV ≥ $800K\n(vs. 11 in Gresham-only)',
             fontsize=11)
ax.legend(fontsize=8, loc='upper left')

# ── Panel 2: Teardown identification (top-right) ────────────────────────────
ax = axes[0, 1]

# Normal sales
norm_bd = normal[~normal['LandDom']]
norm_ld = normal[normal['LandDom']]
ax.scatter(norm_bd['SaleLandRatio'], norm_bd['Ratio'],
           c='#5DA5DA', alpha=0.25, s=15, label=f'Bldg-dom, normal (n={len(norm_bd)})')
ax.scatter(norm_ld['SaleLandRatio'], norm_ld['Ratio'],
           c='#60BD68', alpha=0.4, s=20, label=f'Land-dom, normal (n={len(norm_ld)})')

# Teardowns
ax.scatter(teardowns['SaleLandRatio'], teardowns['Ratio'],
           c='#F15854', marker='x', s=60, linewidths=2, zorder=5,
           label=f'Teardown/distressed (n={len(teardowns)})')

# Viewcrest position (estimated)
vc_sale_land = VIEWCREST_RMV / 900_100  # ~1.31
ax.scatter(vc_sale_land, 1.046, marker='*', c='gold', edgecolors='black',
           s=300, zorder=10, label='Viewcrest position')

# Threshold line
ax.axvline(x=1.05, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(1.05, 0.05, '  Sale = Land value', color='red', fontsize=8, alpha=0.7,
        va='bottom')

ax.set_xlabel('Sale Price / Land Value')
ax.set_ylabel('Sale Price / RMV')
ax.set_title('Identifying Teardown Sales (paid only for dirt)', fontsize=11)
ax.legend(fontsize=7.5, loc='lower right')
ax.set_xlim(0, max(normal['SaleLandRatio'].quantile(0.98), 2.5))
ax.set_ylim(0, 2.0)

# ── Panel 3: Box plots (bottom-left) ────────────────────────────────────────
ax = axes[1, 0]

# Four groups: BD normal, LD normal, Viewcrest-like, Teardowns
groups = [nt_bd['Ratio'], nt_ld['Ratio'], vc_like['Ratio'], teardowns['Ratio']]
labels = [
    f'Building-dom\n(n={len(nt_bd)})',
    f'Land-dom\n(n={len(nt_ld)})',
    f'Viewcrest-like\nLD, bldg $150K-$500K\n(n={len(vc_like)})',
    f'Teardown/\ndistressed\n(n={len(teardowns)})',
]
colors = ['#5DA5DA', '#60BD68', '#FAA43A', '#F15854']

bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2),
                showfliers=True, flierprops=dict(marker='.', markersize=4, alpha=0.4))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Viewcrest reference line
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.text(4.5, 1.0, 'RMV = Sale', fontsize=8, color='gray', ha='right', va='bottom')

ax.set_ylabel('Sale Price / RMV')
ax.set_title(f'Sale/RMV by Property Type ($800K+ RMV, <$3M)\nMann-Whitney p = {p:.2f} (LD vs BD, excl. teardowns)',
             fontsize=11)
ax.set_ylim(0, 2.0)

# ── Panel 4: Closest comps bar chart (bottom-right) ─────────────────────────
ax = axes[1, 1]

if len(closest) > 0:
    closest_sorted = closest.sort_values('Ratio')

    # Shorten addresses for display
    addrs = []
    for _, row in closest_sorted.iterrows():
        city_short = row['City'][:4] if pd.notna(row['City']) else ''
        addr = f"{row['Address']}"
        if len(addr) > 30:
            addr = addr[:28] + '..'
        addrs.append(f"{addr} ({city_short})")

    bar_colors = ['#F15854' if r < 0.9 else '#FAA43A' if r < 1.0 else '#60BD68'
                  for r in closest_sorted['Ratio']]

    bars = ax.barh(range(len(closest_sorted)), closest_sorted['Ratio'],
                   color=bar_colors, alpha=0.7, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(closest_sorted)))
    ax.set_yticklabels(addrs, fontsize=7.5)

    # Reference lines
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=closest_sorted['Ratio'].median(), color='green', linestyle='--',
               alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Sale Price / RMV')
    ax.set_title(f'{len(closest)} Closest Comps (RMV $1.05M–$1.35M, ≥60% land)\n'
                 f'Median: {closest_sorted["Ratio"].median():.3f} '
                 f'→ ${closest_sorted["Ratio"].median() * VIEWCREST_RMV:,.0f} for Viewcrest',
                 fontsize=11)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linestyle='--', label=f'Median: {closest_sorted["Ratio"].median():.3f}'),
        plt.Rectangle((0, 0), 1, 1, fc='#F15854', alpha=0.7, label='Possible teardown'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/figures/addendum_validation.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/figures/addendum_validation.png")
