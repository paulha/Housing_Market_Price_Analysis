"""Compare 555 SW Viewcrest Dr to its RMV class ($900K-$1.5M Gresham residential)."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Subject property
SUBJECT = {
    'PropertyID': 'R113487',
    'Address': '555 SW VIEWCREST DR',
    'TotalRMV': 1_181_560,
    'LandVal': 900_100,
    'BldgVal': 281_460,
    'YearBuilt': 1964,
    'SqFt': 2660,
    'Acres': 1.13,
}
SUBJECT['LandPct'] = SUBJECT['LandVal'] / SUBJECT['TotalRMV'] * 100
SUBJECT['BldgPerSqFt'] = SUBJECT['BldgVal'] / SUBJECT['SqFt']

# Load comparables
df = pd.read_csv('data/gresham_comps.csv')
df['YEARBUILT'] = pd.to_numeric(df['YEARBUILT'], errors='coerce')

# Clean: replace 0.0 acres with NaN (missing data, not actually zero-acre lots)
df.loc[df['A_T_ACRES'] == 0, 'A_T_ACRES'] = np.nan

# Derived columns
df['LandPct'] = df['LANDVAL3'] / df['TOTALVAL3'] * 100
df['BldgPerSqFt'] = df['BLDGVAL3'] / df['BLDGSQFT']

# Filter out year 9999 (bad data)
df = df[df['YEARBUILT'] < 2026]

print(f"=== 555 SW Viewcrest Dr vs RMV Class ($900K-$1.5M) ===")
print(f"Comparable properties: {len(df)}")
print()

# Key metrics comparison
metrics = [
    ('Total RMV', 'TOTALVAL3', SUBJECT['TotalRMV'], '${:,.0f}'),
    ('Land Value', 'LANDVAL3', SUBJECT['LandVal'], '${:,.0f}'),
    ('Building Value', 'BLDGVAL3', SUBJECT['BldgVal'], '${:,.0f}'),
    ('Land % of RMV', 'LandPct', SUBJECT['LandPct'], '{:.1f}%'),
    ('Year Built', 'YEARBUILT', SUBJECT['YearBuilt'], '{:.0f}'),
    ('Building SqFt', 'BLDGSQFT', SUBJECT['SqFt'], '{:,.0f}'),
    ('Bldg $/SqFt (RMV)', 'BldgPerSqFt', SUBJECT['BldgPerSqFt'], '${:.0f}'),
]

# Only include acres for properties that have the data
acres_df = df[df['A_T_ACRES'].notna()]

print(f"{'Metric':<22} {'Viewcrest':>12} {'Median':>12} {'Mean':>12} {'Min':>12} {'Max':>12}  {'Percentile':>10}")
print("-" * 102)

for label, col, subj_val, fmt in metrics:
    med = df[col].median()
    mean = df[col].mean()
    mn = df[col].min()
    mx = df[col].max()
    pctile = (df[col] < subj_val).mean() * 100

    sv = fmt.format(subj_val)
    md = fmt.format(med)
    mn_s = fmt.format(mn)
    mx_s = fmt.format(mx)
    mean_s = fmt.format(mean)

    print(f"{label:<22} {sv:>12} {md:>12} {mean_s:>12} {mn_s:>12} {mx_s:>12}  {pctile:>8.0f}th")

# Acres separately (many missing)
acres_med = acres_df['A_T_ACRES'].median()
acres_mean = acres_df['A_T_ACRES'].mean()
acres_pctile = (acres_df['A_T_ACRES'] < SUBJECT['Acres']).mean() * 100
print(f"{'Lot Size (acres)':<22} {'1.13':>12} {acres_med:>12.2f} {acres_mean:>12.2f} {acres_df['A_T_ACRES'].min():>12.2f} {acres_df['A_T_ACRES'].max():>12.2f}  {acres_pctile:>8.0f}th")
print(f"  (acres data available for {len(acres_df)} of {len(df)} properties)")

print()
print("=== What Makes Viewcrest Stand Out ===")
print()

# Land-heavy analysis
high_land_pct = (df['LandPct'] >= 70).sum()
print(f"Land-dominant (>=70% land): Viewcrest is {SUBJECT['LandPct']:.1f}% land. "
      f"Only {high_land_pct} of {len(df)} comps ({high_land_pct/len(df)*100:.1f}%) are also >=70% land.")

# Low building value analysis
low_bldg = (df['BLDGVAL3'] <= 300_000).sum()
print(f"Low building value (<=300K): Viewcrest building is ${SUBJECT['BldgVal']:,}. "
      f"{low_bldg} of {len(df)} comps ({low_bldg/len(df)*100:.1f}%) also have <=300K building value.")

# Age analysis
older = (df['YEARBUILT'] <= 1964).sum()
print(f"Built 1964 or earlier: {older} of {len(df)} comps ({older/len(df)*100:.1f}%)")

# Small building for the price class
small_bldg = (df['BLDGSQFT'] <= 2700).sum()
print(f"Building <=2,700 sqft: {small_bldg} of {len(df)} comps ({small_bldg/len(df)*100:.1f}%)")

# Large lot analysis
if len(acres_df) > 0:
    large_lot = (acres_df['A_T_ACRES'] >= 1.0).sum()
    print(f"Lot >=1 acre: {large_lot} of {len(acres_df)} comps with data ({large_lot/len(acres_df)*100:.1f}%)")

# Find the most similar properties (land-dominant, older, large lot)
print()
print("=== Most Similar Properties (land-dominant, older homes) ===")
similar = df[(df['LandPct'] >= 60) & (df['YEARBUILT'] <= 1975) & (df['A_T_ACRES'].notna()) & (df['A_T_ACRES'] >= 0.8)]
if len(similar) > 0:
    similar = similar.sort_values('LandPct', ascending=False)
    print(f"Found {len(similar)} properties with >=60% land, built <=1975, lot >=0.8 acres:")
    print()
    print(f"{'Address':<35} {'RMV':>12} {'Land%':>7} {'BldgVal':>10} {'YrBlt':>6} {'SqFt':>6} {'Acres':>6}")
    print("-" * 92)
    print(f"{'>>> 555 SW VIEWCREST DR <<<':<35} {'$1,181,560':>12} {'76.2%':>7} {'$281,460':>10} {'1964':>6} {'2,660':>6} {'1.13':>6}")
    print("-" * 92)
    for _, row in similar.iterrows():
        print(f"{row['SITEADDR']:<35} ${row['TOTALVAL3']:>11,.0f} {row['LandPct']:>6.1f}% ${row['BLDGVAL3']:>9,.0f} {int(row['YEARBUILT']):>6} {int(row['BLDGSQFT']):>6} {row['A_T_ACRES']:>6.2f}")
else:
    print("No closely matching properties found.")

# ---- Visualization ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('555 SW Viewcrest Dr vs RMV Class ($900K-$1.5M Gresham)', fontsize=14, fontweight='bold')

def add_subject_line(ax, val, label='Viewcrest'):
    ax.axvline(val, color='red', linewidth=2, linestyle='--', label=label)
    ax.legend(fontsize=8)

# 1. Land % distribution
ax = axes[0, 0]
ax.hist(df['LandPct'], bins=25, color='steelblue', edgecolor='white', alpha=0.7)
add_subject_line(ax, SUBJECT['LandPct'])
ax.set_xlabel('Land % of Total RMV')
ax.set_ylabel('Count')
ax.set_title('Land Value as % of RMV')

# 2. Building value distribution
ax = axes[0, 1]
ax.hist(df['BLDGVAL3'] / 1000, bins=25, color='steelblue', edgecolor='white', alpha=0.7)
add_subject_line(ax, SUBJECT['BldgVal'] / 1000)
ax.set_xlabel('Building Value ($K)')
ax.set_ylabel('Count')
ax.set_title('Building Value')

# 3. Year built distribution
ax = axes[0, 2]
ax.hist(df['YEARBUILT'], bins=25, color='steelblue', edgecolor='white', alpha=0.7)
add_subject_line(ax, SUBJECT['YearBuilt'])
ax.set_xlabel('Year Built')
ax.set_ylabel('Count')
ax.set_title('Year Built')

# 4. Building sqft distribution
ax = axes[1, 0]
ax.hist(df['BLDGSQFT'], bins=25, color='steelblue', edgecolor='white', alpha=0.7)
add_subject_line(ax, SUBJECT['SqFt'])
ax.set_xlabel('Building SqFt')
ax.set_ylabel('Count')
ax.set_title('Building Square Footage')

# 5. Land vs Building value scatter
ax = axes[1, 1]
ax.scatter(df['LANDVAL3'] / 1000, df['BLDGVAL3'] / 1000, alpha=0.4, s=20, color='steelblue', label='Comps')
ax.scatter(SUBJECT['LandVal'] / 1000, SUBJECT['BldgVal'] / 1000, color='red', s=100, zorder=5,
           marker='*', label='Viewcrest')
ax.plot([0, 1500], [0, 1500], 'g--', alpha=0.3, label='50/50 line')
ax.set_xlabel('Land Value ($K)')
ax.set_ylabel('Building Value ($K)')
ax.set_title('Land vs Building Value')
ax.legend(fontsize=8)

# 6. Lot size distribution (where available)
ax = axes[1, 2]
if len(acres_df) > 0:
    # Cap display at 5 acres for readability
    display_acres = acres_df['A_T_ACRES'].clip(upper=6)
    ax.hist(display_acres, bins=25, color='steelblue', edgecolor='white', alpha=0.7)
    add_subject_line(ax, SUBJECT['Acres'])
    ax.set_xlabel('Lot Size (acres)')
    ax.set_ylabel('Count')
    ax.set_title(f'Lot Size (n={len(acres_df)} with data)')

plt.tight_layout()
output_path = 'results/figures/viewcrest_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to {output_path}")
