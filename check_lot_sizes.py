"""Check if building-dominant homes (above 50/50 line) are on smaller lots."""

import pandas as pd
import numpy as np

# Load original Redfin CSV for lot sizes
redfin = pd.read_csv('data/raw/redfin_Gresham_For_Sale.csv')
redfin = redfin.dropna(subset=['ADDRESS'])
redfin = redfin[redfin['CITY'].astype(str).str.upper().str.strip() == 'GRESHAM']

# Load enriched data
enriched = pd.read_csv('results/reports/gresham_forsale_enriched.csv')

# Redfin LOT SIZE is in sqft — convert to acres
redfin['LotSqFt'] = pd.to_numeric(redfin['LOT SIZE'], errors='coerce')
redfin['LotAcres'] = redfin['LotSqFt'] / 43560

# Match by normalizing addresses
from gresham_data_collector import normalize_address
redfin['NormAddr'] = redfin['ADDRESS'].apply(normalize_address)
enriched['NormAddr'] = enriched['Address']

merged = enriched.merge(redfin[['NormAddr', 'LotSqFt', 'LotAcres']], on='NormAddr', how='left')
merged['LandPct'] = merged['RMVLAND'] / merged['RMV'] * 100

valid = merged[merged['LotAcres'].notna() & (merged['LotAcres'] > 0)].copy()

print(f"Listings with lot size data: {len(valid)} of {len(enriched)}")
print()

# Split at 50% land
bldg_dominant = valid[valid['LandPct'] < 50]
land_dominant = valid[valid['LandPct'] >= 50]

print(f"Building-dominant (<50% land): {len(bldg_dominant)} listings")
print(f"  Median lot size: {bldg_dominant['LotAcres'].median():.3f} acres ({bldg_dominant['LotSqFt'].median():,.0f} sqft)")
print(f"  Mean lot size:   {bldg_dominant['LotAcres'].mean():.3f} acres")
print(f"  Range:           {bldg_dominant['LotAcres'].min():.3f} - {bldg_dominant['LotAcres'].max():.3f} acres")
print()
print(f"Land-dominant (>=50% land): {len(land_dominant)} listings")
print(f"  Median lot size: {land_dominant['LotAcres'].median():.3f} acres ({land_dominant['LotSqFt'].median():,.0f} sqft)")
print(f"  Mean lot size:   {land_dominant['LotAcres'].mean():.3f} acres")
print(f"  Range:           {land_dominant['LotAcres'].min():.3f} - {land_dominant['LotAcres'].max():.3f} acres")

from scipy import stats
if len(land_dominant) >= 2 and len(bldg_dominant) >= 2:
    u_stat, p_val = stats.mannwhitneyu(land_dominant['LotAcres'], bldg_dominant['LotAcres'], alternative='two-sided')
    print(f"\n  Mann-Whitney U test: p = {p_val:.4f}")

print(f"\n--- All Listings by Land % (descending) ---")
valid_sorted = valid.sort_values('LandPct', ascending=False)
print(f"{'Address':<35} {'Land%':>7} {'LotAcres':>10} {'LotSqFt':>10} {'RMV':>12} {'LandVal':>12} {'BldgVal':>12}")
print("-" * 103)
for _, row in valid_sorted.iterrows():
    print(f"{row['Address']:<35} {row['LandPct']:>6.1f}% {row['LotAcres']:>10.3f} {row['LotSqFt']:>10,.0f} ${row['RMV']:>11,.0f} ${row['RMVLAND']:>11,.0f} ${row['RMVIMPR']:>11,.0f}")

# Correlation
r, p = stats.pearsonr(valid['LotAcres'], valid['LandPct'])
print(f"\nCorrelation (lot acres vs land %): r = {r:.3f}, p = {p:.4f}")
