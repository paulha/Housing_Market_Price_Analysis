"""How does the realtor's proposed price compare to all our data?"""

import pandas as pd
import numpy as np

SUBJECT_RMV = 1_181_560
REALTOR_LOW = 600_000
REALTOR_HIGH = 750_000

print(f"=== Realtor's Proposed Price vs Data ===")
print(f"Property: 555 SW Viewcrest Dr, Gresham")
print(f"County RMV (2025): ${SUBJECT_RMV:,}")
print(f"Realtor range: ${REALTOR_LOW:,} - ${REALTOR_HIGH:,}")
print(f"Realtor as % of RMV: {REALTOR_LOW/SUBJECT_RMV*100:.1f}% - {REALTOR_HIGH/SUBJECT_RMV*100:.1f}%")
print()

# 1. Compare to sold data
print("--- 1. What do Gresham homes actually SELL for relative to RMV? ---")
sold = pd.read_csv('data/gresham/gresham_research_db.csv')
sold['YearBuilt'] = pd.to_numeric(sold['YearBuilt'], errors='coerce')
sold = sold[(sold['SalePrice'] > 0) & (sold['RMV'] > 0) & (sold['YearBuilt'] < 2024)]
sold['Ratio'] = sold['SalePrice'] / sold['RMV']

print(f"  197 existing-home sales in Gresham:")
print(f"  Median Sale/RMV ratio: {sold['Ratio'].median():.3f}")
print(f"  5th percentile ratio:  {sold['Ratio'].quantile(0.05):.3f}")
print(f"  Lowest ratio observed: {sold['Ratio'].min():.3f}")
print()

realtor_ratio = REALTOR_HIGH / SUBJECT_RMV
below_realtor = (sold['Ratio'] <= realtor_ratio).sum()
print(f"  Realtor's $750K = {realtor_ratio:.3f} ratio to RMV")
print(f"  Only {below_realtor} of {len(sold)} sales ({below_realtor/len(sold)*100:.1f}%) sold at or below that ratio")
print()

# What about the $600K number?
realtor_low_ratio = REALTOR_LOW / SUBJECT_RMV
below_600 = (sold['Ratio'] <= realtor_low_ratio).sum()
print(f"  Realtor's $600K = {realtor_low_ratio:.3f} ratio to RMV")
print(f"  Only {below_600} of {len(sold)} sales ({below_600/len(sold)*100:.1f}%) sold at or below that ratio")

# Show the actual worst-performing sales
print(f"\n  Bottom 10 Sale/RMV ratios in our data:")
worst = sold.nsmallest(10, 'Ratio')
for _, row in worst.iterrows():
    yb = int(row['YearBuilt']) if pd.notna(row['YearBuilt']) else 0
    print(f"    {row['Address']:<35} ratio={row['Ratio']:.3f}  "
          f"(${row['SalePrice']:,.0f} vs ${row['RMV']:,.0f} RMV)  built {yb}")

# 2. Compare to asking prices
print(f"\n--- 2. What are Gresham sellers currently ASKING relative to RMV? ---")
asking = pd.read_csv('results/reports/gresham_forsale_enriched.csv')
print(f"  34 current for-sale listings:")
print(f"  Median Ask/RMV ratio: {asking['AskToRMV'].median():.3f}")
print(f"  Lowest Ask/RMV ratio: {asking['AskToRMV'].min():.3f} ({asking.loc[asking['AskToRMV'].idxmin(), 'Address']})")
print()
below_ask = (asking['AskToRMV'] <= realtor_ratio).sum()
print(f"  Listings at or below {realtor_ratio:.3f} ratio: {below_ask} of {len(asking)}")

# 3. High-value comparables
print(f"\n--- 3. High-value comparable sales ($800K+ RMV) ---")
high = sold[sold['RMV'] >= 800_000].sort_values('Ratio')
for _, row in high.iterrows():
    yb = int(row['YearBuilt']) if pd.notna(row['YearBuilt']) else 0
    print(f"  {row['Address']:<35} sold ${row['SalePrice']:>10,.0f}  RMV ${row['RMV']:>10,.0f}  "
          f"ratio {row['Ratio']:.3f}  built {yb}")

print(f"\n  Worst high-value ratio: {high['Ratio'].min():.3f} = ${high['Ratio'].min() * SUBJECT_RMV:,.0f} for Viewcrest's RMV")
print(f"  Median high-value ratio: {high['Ratio'].median():.3f} = ${high['Ratio'].median() * SUBJECT_RMV:,.0f} for Viewcrest's RMV")

# 4. The comparable land-heavy listings currently for sale
print(f"\n--- 4. Current land-heavy listings (closest comparables) ---")
asking['LandPct'] = asking['RMVLAND'] / asking['RMV'] * 100
land_heavy_ask = asking[asking['LandPct'] >= 50].sort_values('LandPct', ascending=False)
for _, row in land_heavy_ask.iterrows():
    print(f"  {row['Address']:<35} RMV ${row['RMV']:>10,.0f}  asking ${row['AskingPrice']:>10,.0f}  "
          f"ratio {row['AskToRMV']:.3f}  land {row['LandPct']:.0f}%")

# 5. Summary
print(f"\n{'='*60}")
print(f"SUMMARY: Data-supported price range for 555 SW Viewcrest Dr")
print(f"{'='*60}")
print(f"  County RMV (2025):              ${SUBJECT_RMV:>12,}")
print(f"  Overall regression prediction:  ${1_102_993:>12,}")
print(f"  High-value regression:          ${1_234_779:>12,}")
print(f"  Median high-value ratio:        ${high['Ratio'].median() * SUBJECT_RMV:>12,.0f}")
print(f"  Land-dominant median ratio:     ${1_189_763:>12,}")
print(f"  Listing median markup (6.5%):   ${SUBJECT_RMV * 1.065:>12,.0f}")
print(f"  Worst-case (lowest high-val):   ${high['Ratio'].min() * SUBJECT_RMV:>12,.0f}")
print(f"  {'':>36} {'----------':>12}")
print(f"  Data-supported range:           ${'~860K - ~1.26M':>12}")
print(f"")
print(f"  Realtor's proposal:             ${REALTOR_LOW:>12,} - ${REALTOR_HIGH:>12,}")
print(f"  Realtor as % of RMV:            {REALTOR_LOW/SUBJECT_RMV*100:>11.1f}% - {REALTOR_HIGH/SUBJECT_RMV*100:>11.1f}%")
print(f"  Gap below data floor ($860K):   ${860_000 - REALTOR_HIGH:>12,} - ${860_000 - REALTOR_LOW:>12,}")
