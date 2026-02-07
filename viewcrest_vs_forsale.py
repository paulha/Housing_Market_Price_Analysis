"""Compare 555 SW Viewcrest Dr to the current for-sale listings."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SUBJECT = {
    'Address': '555 SW VIEWCREST DR',
    'AskingPrice': None,  # not for sale
    'RMV': 1_181_560,
    'RMVLAND': 900_100,
    'RMVIMPR': 281_460,
    'YearBuilt': 1964,
    'SqFt': 2660,
    'Acres': 1.13,
    'Beds': None,
    'Baths': None,
}
SUBJECT['LandPct'] = SUBJECT['RMVLAND'] / SUBJECT['RMV'] * 100
SUBJECT['BldgPerSqFt'] = SUBJECT['RMVIMPR'] / SUBJECT['SqFt']

df = pd.read_csv('results/reports/gresham_forsale_enriched.csv')
df['LandPct'] = df['RMVLAND'] / df['RMV'] * 100
df['BldgPerSqFt'] = df['RMVIMPR'] / df['SqFt']

print(f"=== 555 SW Viewcrest Dr vs Current For-Sale Listings (n={len(df)}) ===\n")

print(f"{'Metric':<22} {'Viewcrest':>12} {'Listings Med':>14} {'Listings Min':>14} {'Listings Max':>14}")
print("-" * 80)

metrics = [
    ('RMV', 'RMV', SUBJECT['RMV'], '${:,.0f}'),
    ('Asking Price', 'AskingPrice', None, '${:,.0f}'),
    ('Land Value', 'RMVLAND', SUBJECT['RMVLAND'], '${:,.0f}'),
    ('Building Value', 'RMVIMPR', SUBJECT['RMVIMPR'], '${:,.0f}'),
    ('Land % of RMV', 'LandPct', SUBJECT['LandPct'], '{:.1f}%'),
    ('Year Built', 'YearBuilt', SUBJECT['YearBuilt'], '{:.0f}'),
    ('Building SqFt', 'SqFt', SUBJECT['SqFt'], '{:,.0f}'),
    ('Bldg RMV $/SqFt', 'BldgPerSqFt', SUBJECT['BldgPerSqFt'], '${:.0f}'),
]

for label, col, subj_val, fmt in metrics:
    med = df[col].median()
    mn = df[col].min()
    mx = df[col].max()
    if subj_val is not None:
        sv = fmt.format(subj_val)
    else:
        sv = 'N/A'
    print(f"{label:<22} {sv:>12} {fmt.format(med):>14} {fmt.format(mn):>14} {fmt.format(mx):>14}")

# Percentile rankings
print(f"\n--- Where Viewcrest Ranks Among Listings ---")
for label, col, subj_val, fmt in metrics:
    if subj_val is not None:
        pctile = (df[col] < subj_val).mean() * 100
        above = (df[col] > subj_val).sum()
        below = (df[col] < subj_val).sum()
        print(f"  {label:<22} {pctile:.0f}th percentile ({below} below, {above} above)")

# Key differences
print(f"\n--- Key Differences ---")
print(f"\n1. PRICE TIER:")
print(f"   Viewcrest RMV ($1.18M) is far above the listing median RMV (${df['RMV'].median():,.0f}).")
n_above = (df['RMV'] >= 1_000_000).sum()
print(f"   Only {n_above} of {len(df)} listings have RMV >= $1M.")

print(f"\n2. LAND vs BUILDING COMPOSITION:")
print(f"   Viewcrest: {SUBJECT['LandPct']:.1f}% land / {100-SUBJECT['LandPct']:.1f}% building")
print(f"   Listings:  {df['LandPct'].median():.1f}% land / {100-df['LandPct'].median():.1f}% building (median)")
high_land = (df['LandPct'] >= 50).sum()
print(f"   Only {high_land} of {len(df)} listings are >=50% land value.")

# Find the most similar listing
print(f"\n3. MOST SIMILAR LISTINGS (by land %):")
df_sorted = df.sort_values('LandPct', ascending=False).head(5)
print(f"   {'Address':<35} {'RMV':>10} {'Ask':>10} {'Land%':>7} {'YrBlt':>6} {'SqFt':>6}")
print(f"   {'-'*80}")
print(f"   {'>>> VIEWCREST (not for sale) <<<':<35} {'$1,181K':>10} {'N/A':>10} {'76.2%':>7} {'1964':>6} {'2,660':>6}")
print(f"   {'-'*80}")
for _, row in df_sorted.iterrows():
    print(f"   {row['Address']:<35} ${row['RMV']/1000:>9,.0f}K ${row['AskingPrice']/1000:>9,.0f}K {row['LandPct']:>6.1f}% {int(row['YearBuilt']):>6} {int(row['SqFt']):>6}")

print(f"\n4. BUILDING VALUE PER SQFT:")
print(f"   Viewcrest: ${SUBJECT['BldgPerSqFt']:.0f}/sqft (county's building RMV per sqft)")
print(f"   Listings:  ${df['BldgPerSqFt'].median():.0f}/sqft (median)")
print(f"   Viewcrest's building is valued at {SUBJECT['BldgPerSqFt']/df['BldgPerSqFt'].median()*100:.0f}% of the typical listing's building $/sqft.")

print(f"\n5. AGE:")
print(f"   Viewcrest: built 1964 ({2026-1964} years old)")
print(f"   Listings:  median built {df['YearBuilt'].median():.0f} ({2026-df['YearBuilt'].median():.0f} years old)")
older = (df['YearBuilt'] <= 1964).sum()
print(f"   Only {older} of {len(df)} listings are as old or older.")

# What would Viewcrest's asking price be if it followed the listing pattern?
print(f"\n--- Implied Asking Price for Viewcrest ---")
med_ratio = df['AskToRMV'].median()
mean_ratio = df['AskToRMV'].mean()
print(f"  If priced at listing median Ask/RMV ratio ({med_ratio:.3f}): ${SUBJECT['RMV'] * med_ratio:,.0f}")
print(f"  If priced at listing mean Ask/RMV ratio ({mean_ratio:.3f}):   ${SUBJECT['RMV'] * mean_ratio:,.0f}")

# The two most comparable high-value listings
print(f"\n--- High-Value Listings (closest comparables) ---")
high = df[df['RMV'] >= 700_000].sort_values('RMV', ascending=False)
if len(high) > 0:
    print(f"   {'Address':<30} {'RMV':>12} {'Asking':>12} {'Ask/RMV':>8} {'Land%':>7} {'YrBlt':>6} {'SqFt':>6} {'Days':>5}")
    print(f"   {'-'*90}")
    for _, row in high.iterrows():
        print(f"   {row['Address']:<30} ${row['RMV']:>11,.0f} ${row['AskingPrice']:>11,.0f} {row['AskToRMV']:>8.3f} {row['LandPct']:>6.1f}% {int(row['YearBuilt']):>6} {int(row['SqFt']):>6} {int(row['DaysOnMarket']):>5}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('555 SW Viewcrest Dr vs Current Gresham For-Sale Listings', fontsize=14, fontweight='bold')

def vline(ax, val, label='Viewcrest'):
    ax.axvline(val, color='red', linewidth=2, linestyle='--', label=label)
    ax.legend(fontsize=8)

# 1. RMV distribution
ax = axes[0, 0]
ax.hist(df['RMV'] / 1000, bins=15, color='darkorange', edgecolor='white', alpha=0.7)
vline(ax, SUBJECT['RMV'] / 1000)
ax.set_xlabel('County RMV ($K)')
ax.set_ylabel('Count')
ax.set_title('RMV Distribution of For-Sale Listings')

# 2. Land % distribution
ax = axes[0, 1]
ax.hist(df['LandPct'], bins=15, color='darkorange', edgecolor='white', alpha=0.7)
vline(ax, SUBJECT['LandPct'])
ax.set_xlabel('Land % of RMV')
ax.set_ylabel('Count')
ax.set_title('Land Value as % of RMV')

# 3. Land vs Building scatter
ax = axes[1, 0]
ax.scatter(df['RMVLAND'] / 1000, df['RMVIMPR'] / 1000, alpha=0.6, s=40, color='darkorange',
           edgecolors='white', linewidth=0.3, label='For-sale listings')
ax.scatter(SUBJECT['RMVLAND'] / 1000, SUBJECT['RMVIMPR'] / 1000, color='red', s=150,
           marker='*', zorder=5, label='Viewcrest')
ax.plot([0, 1100], [0, 1100], 'g--', alpha=0.3, label='50/50 line')
ax.set_xlabel('Land Value ($K)')
ax.set_ylabel('Building Value ($K)')
ax.set_title('Land vs Building Value')
ax.legend(fontsize=8)

# 4. Building $/sqft distribution
ax = axes[1, 1]
ax.hist(df['BldgPerSqFt'], bins=15, color='darkorange', edgecolor='white', alpha=0.7)
vline(ax, SUBJECT['BldgPerSqFt'])
ax.set_xlabel('Building RMV per SqFt ($)')
ax.set_ylabel('Count')
ax.set_title('Building Value per Square Foot')

plt.tight_layout()
out = 'results/figures/viewcrest_vs_forsale.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to {out}")
