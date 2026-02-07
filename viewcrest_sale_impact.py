"""Does land-to-building ratio affect how properties sell relative to RMV?"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Load Gresham research DB (actual sales with RMV)
db = pd.read_csv('data/gresham/gresham_research_db.csv')

# Filter: valid sales, exclude new construction
db = db[(db['SalePrice'] > 0) & (db['RMV'] > 0)]
db['YearBuilt'] = pd.to_numeric(db['YearBuilt'], errors='coerce')
db = db[db['YearBuilt'] < 2024]  # exclude new construction

db['SaleToRMV'] = db['SalePrice'] / db['RMV']
db['LandPct'] = db['RMVLAND'] / db['RMV'] * 100
db['BldgPerSqFt'] = db['RMVIMPR'] / db['SqFt']

print(f"=== Impact of Land % on Sale Price Relative to RMV ===")
print(f"Gresham sales (existing homes): {len(db)}")
print()

# Split into land-dominant vs building-dominant
land_heavy = db[db['LandPct'] >= 50]
bldg_heavy = db[db['LandPct'] < 50]

print(f"Land-dominant (>=50% land): {len(land_heavy)} sales")
print(f"  Median Sale/RMV: {land_heavy['SaleToRMV'].median():.3f}")
print(f"  Mean Sale/RMV:   {land_heavy['SaleToRMV'].mean():.3f}")
print(f"  Median RMV:      ${land_heavy['RMV'].median():,.0f}")
print()
print(f"Building-dominant (<50% land): {len(bldg_heavy)} sales")
print(f"  Median Sale/RMV: {bldg_heavy['SaleToRMV'].median():.3f}")
print(f"  Mean Sale/RMV:   {bldg_heavy['SaleToRMV'].mean():.3f}")
print(f"  Median RMV:      ${bldg_heavy['RMV'].median():,.0f}")
print()

# Statistical test
if len(land_heavy) >= 5 and len(bldg_heavy) >= 5:
    t_stat, p_val = stats.mannwhitneyu(land_heavy['SaleToRMV'], bldg_heavy['SaleToRMV'], alternative='two-sided')
    print(f"Mann-Whitney U test (land-heavy vs bldg-heavy): p = {p_val:.3f}")
    print()

# Correlation: Land% vs Sale/RMV ratio
r, p = stats.pearsonr(db['LandPct'], db['SaleToRMV'])
print(f"Correlation (Land% vs Sale/RMV): r = {r:.3f}, p = {p:.3f}")
print()

# Look at high-value properties specifically (>$800K RMV, closer to Viewcrest's range)
high_val = db[db['RMV'] >= 800_000]
print(f"=== High-Value Properties (RMV >= $800K) ===")
print(f"Count: {len(high_val)}")
if len(high_val) > 0:
    print(f"  Median Sale/RMV: {high_val['SaleToRMV'].median():.3f}")
    print(f"  Range: {high_val['SaleToRMV'].min():.3f} - {high_val['SaleToRMV'].max():.3f}")
    print()
    high_val_sorted = high_val.sort_values('LandPct', ascending=False)
    print(f"{'Address':<35} {'RMV':>12} {'SalePrice':>12} {'Ratio':>7} {'Land%':>7} {'YrBlt':>6}")
    print("-" * 85)
    for _, row in high_val_sorted.iterrows():
        print(f"{row['Address']:<35} ${row['RMV']:>11,.0f} ${row['SalePrice']:>11,.0f} {row['SaleToRMV']:>7.3f} {row['LandPct']:>6.1f}% {int(row['YearBuilt']):>6}")

# Regression: what does our model predict for Viewcrest?
print()
print(f"=== Viewcrest Sale Price Estimates ===")
print()

# Method 1: Overall regression (from our earlier analysis)
slope, intercept, r_val, p_val, std_err = stats.linregress(db['RMV'], db['SalePrice'])
pred_overall = slope * 1_181_560 + intercept
print(f"Method 1 - Overall regression (all sales):")
print(f"  Predicted sale price: ${pred_overall:,.0f}")
print(f"  (slope={slope:.3f}, intercept=${intercept:,.0f}, r={r_val:.3f})")

# Method 2: High-value only regression
if len(high_val) >= 5:
    slope2, intercept2, r2, p2, se2 = stats.linregress(high_val['RMV'], high_val['SalePrice'])
    pred_high = slope2 * 1_181_560 + intercept2
    print(f"\nMethod 2 - High-value regression (RMV >= $800K):")
    print(f"  Predicted sale price: ${pred_high:,.0f}")
    print(f"  (slope={slope2:.3f}, intercept=${intercept2:,.0f}, r={r2:.3f})")

# Method 3: Median ratio of high-value properties
if len(high_val) > 0:
    med_ratio = high_val['SaleToRMV'].median()
    pred_ratio = med_ratio * 1_181_560
    print(f"\nMethod 3 - Median high-value Sale/RMV ratio ({med_ratio:.3f}):")
    print(f"  Predicted sale price: ${pred_ratio:,.0f}")

# Method 4: Land-heavy subset
if len(land_heavy) >= 5:
    med_land_ratio = land_heavy['SaleToRMV'].median()
    pred_land = med_land_ratio * 1_181_560
    print(f"\nMethod 4 - Land-dominant properties median ratio ({med_land_ratio:.3f}):")
    print(f"  Predicted sale price: ${pred_land:,.0f}")

# Confidence interval based on high-value sales
if len(high_val) >= 5:
    ratios = high_val['SaleToRMV']
    low_ci = ratios.quantile(0.25) * 1_181_560
    high_ci = ratios.quantile(0.75) * 1_181_560
    print(f"\nInterquartile range (25th-75th pctile of high-value ratios):")
    print(f"  ${low_ci:,.0f} - ${high_ci:,.0f}")

    extreme_low = ratios.min() * 1_181_560
    extreme_high = ratios.max() * 1_181_560
    print(f"Full range observed:")
    print(f"  ${extreme_low:,.0f} - ${extreme_high:,.0f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Does Land % Affect Sale Price Relative to RMV?', fontsize=13, fontweight='bold')

# 1. Land% vs Sale/RMV scatter
ax = axes[0]
ax.scatter(db['LandPct'], db['SaleToRMV'], alpha=0.4, s=20, color='steelblue')
# Trend line
z = np.polyfit(db['LandPct'], db['SaleToRMV'], 1)
x_line = np.linspace(db['LandPct'].min(), db['LandPct'].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'r-', alpha=0.7, label=f'Trend (r={r:.2f})')
ax.axvline(76.2, color='red', linestyle='--', alpha=0.5, label='Viewcrest (76.2%)')
ax.axhline(1.0, color='green', linestyle='--', alpha=0.3)
ax.set_xlabel('Land % of RMV')
ax.set_ylabel('Sale Price / RMV')
ax.set_title('Land % vs Sale/RMV Ratio')
ax.legend(fontsize=8)

# 2. Box plot comparing groups
ax = axes[1]
data_groups = [land_heavy['SaleToRMV'].values, bldg_heavy['SaleToRMV'].values]
bp = ax.boxplot(data_groups, tick_labels=[f'Land-heavy\n(>=50%, n={len(land_heavy)})',
                                          f'Bldg-heavy\n(<50%, n={len(bldg_heavy)})'],
                patch_artist=True)
bp['boxes'][0].set_facecolor('#ff9999')
bp['boxes'][1].set_facecolor('#99ccff')
ax.axhline(1.0, color='green', linestyle='--', alpha=0.3)
ax.set_ylabel('Sale Price / RMV')
ax.set_title('Sale/RMV by Land Dominance')

# 3. High-value sales: RMV vs Sale Price with Viewcrest prediction
ax = axes[2]
if len(high_val) > 0:
    colors = ['red' if lp >= 50 else 'steelblue' for lp in high_val['LandPct']]
    ax.scatter(high_val['RMV'] / 1000, high_val['SalePrice'] / 1000, c=colors, alpha=0.6, s=40)
    ax.plot([700, 1600], [700, 1600], 'g--', alpha=0.3, label='RMV = Sale')
    ax.scatter([1181.56], [pred_overall / 1000], color='red', marker='*', s=200, zorder=5,
               label=f'Viewcrest pred ${pred_overall/1000:.0f}K')
    # Add legend entries for colors
    ax.scatter([], [], c='red', alpha=0.6, s=40, label='Land-heavy (>=50%)')
    ax.scatter([], [], c='steelblue', alpha=0.6, s=40, label='Bldg-heavy (<50%)')
    ax.set_xlabel('RMV ($K)')
    ax.set_ylabel('Sale Price ($K)')
    ax.set_title('High-Value Sales (RMV >= $800K)')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('results/figures/viewcrest_sale_impact.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to results/figures/viewcrest_sale_impact.png")
