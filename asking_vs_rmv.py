"""
Compare Redfin asking prices (current for-sale listings) to county RMV.
Reuses ArcGIS lookup from gresham_data_collector.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from gresham_data_collector import (
    normalize_address, lookup_property, REQUEST_DELAY
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_forsale_csv(filepath):
    """Load a Redfin for-sale CSV. PRICE = asking price, no SOLD DATE."""
    df = pd.read_csv(filepath)

    # Skip Redfin disclaimer rows (no address)
    df = df.dropna(subset=['ADDRESS'])

    # Filter to Gresham
    df = df[df['CITY'].astype(str).str.upper().str.strip() == 'GRESHAM'].copy()

    # Parse asking price
    df['AskingPrice'] = pd.to_numeric(
        df['PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    df = df[df['AskingPrice'] > 0]

    print(f"Loaded {len(df)} Gresham for-sale listings from {os.path.basename(filepath)}")
    return df


def enrich_listings(df):
    """Look up county RMV for each listing via ArcGIS."""
    records = []
    total = len(df)
    matched = 0

    print(f"\nEnriching {total} listings with county RMV (est. ~{total * REQUEST_DELAY:.0f}s)...\n")

    for i, (_, row) in enumerate(df.iterrows()):
        address = row.get('ADDRESS', '')

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{total}] {address}")

        attrs, method, confidence = lookup_property(address)
        time.sleep(REQUEST_DELAY)

        record = {
            'Address': normalize_address(address),
            'AskingPrice': row['AskingPrice'],
            'Beds': row.get('BEDS'),
            'Baths': row.get('BATHS'),
            'SqFt_Redfin': row.get('SQUARE FEET'),
            'YearBuilt_Redfin': row.get('YEAR BUILT'),
            'DaysOnMarket': row.get('DAYS ON MARKET'),
            'MatchMethod': method,
        }

        if attrs:
            # Use 2025 assessment (TOTALVAL3) for current listings
            record.update({
                'PropertyID': attrs.get('PROPERTYID'),
                'RMV': attrs.get('TOTALVAL3'),
                'RMVLAND': attrs.get('LANDVAL3'),
                'RMVIMPR': attrs.get('BLDGVAL3'),
                'YearBuilt': attrs.get('YEARBUILT'),
                'SqFt': attrs.get('BLDGSQFT'),
            })
            matched += 1
        else:
            record.update({
                'PropertyID': None, 'RMV': None, 'RMVLAND': None,
                'RMVIMPR': None, 'YearBuilt': None, 'SqFt': None,
            })
            print(f"    UNMATCHED: {address}")

        records.append(record)

    result = pd.DataFrame(records)
    print(f"\nMatched {matched}/{total} listings")
    return result


def analyze_asking_vs_rmv(df):
    """Analyze and visualize asking price vs county RMV."""
    # Filter to records with both values
    valid = df[(df['AskingPrice'] > 0) & (df['RMV'] > 0)].copy()

    # Exclude new construction (same logic as sold analysis)
    valid['YearBuilt'] = pd.to_numeric(valid['YearBuilt'], errors='coerce')
    new_construction = valid[valid['YearBuilt'] >= 2024]
    existing = valid[valid['YearBuilt'] < 2024]

    print(f"\n{'='*60}")
    print(f"ASKING PRICE vs COUNTY RMV — Gresham For-Sale Listings")
    print(f"{'='*60}")
    print(f"Total matched listings: {len(valid)}")
    print(f"New construction (2024+): {len(new_construction)} (excluded)")
    print(f"Existing homes: {len(existing)}")

    if len(existing) < 5:
        print("Too few existing homes for analysis.")
        return

    existing['AskToRMV'] = existing['AskingPrice'] / existing['RMV']
    existing['PctDiff'] = (existing['AskingPrice'] - existing['RMV']) / existing['RMV'] * 100

    # Summary stats
    print(f"\n--- Asking Price vs RMV (existing homes) ---")
    print(f"  Median asking price:  ${existing['AskingPrice'].median():,.0f}")
    print(f"  Median RMV:           ${existing['RMV'].median():,.0f}")
    print(f"  Median Ask/RMV ratio: {existing['AskToRMV'].median():.3f}")
    print(f"  Mean Ask/RMV ratio:   {existing['AskToRMV'].mean():.3f}")
    print(f"  Median % above RMV:   {existing['PctDiff'].median():+.1f}%")
    print(f"  Std dev of % diff:    {existing['PctDiff'].std():.1f}%")

    # Correlation
    r, p = stats.pearsonr(existing['RMV'], existing['AskingPrice'])
    print(f"\n  Correlation (r):      {r:.3f}")
    print(f"  R-squared:            {r**2:.3f}")

    # Regression
    slope, intercept, r_val, p_val, std_err = stats.linregress(
        existing['RMV'], existing['AskingPrice']
    )
    print(f"  Regression slope:     {slope:.3f}")
    print(f"  Regression intercept: ${intercept:,.0f}")

    # Bias test
    t_stat, t_p = stats.ttest_1samp(existing['AskToRMV'], 1.0)
    print(f"\n  One-sample t-test (ratio vs 1.0): t={t_stat:.2f}, p={t_p:.4f}")
    if t_p < 0.05:
        direction = "above" if existing['AskToRMV'].mean() > 1 else "below"
        print(f"  => Systematic bias: asking prices are significantly {direction} RMV")
    else:
        print(f"  => No significant bias detected")

    # Load sold data for comparison
    sold_db_path = os.path.join(BASE_DIR, 'data', 'gresham', 'gresham_research_db.csv')
    sold_df = None
    if os.path.exists(sold_db_path):
        sold_raw = pd.read_csv(sold_db_path)
        sold_raw['YearBuilt'] = pd.to_numeric(sold_raw['YearBuilt'], errors='coerce')
        sold_df = sold_raw[
            (sold_raw['SalePrice'] > 0) & (sold_raw['RMV'] > 0) & (sold_raw['YearBuilt'] < 2024)
        ].copy()
        sold_df['SaleToRMV'] = sold_df['SalePrice'] / sold_df['RMV']
        sold_df['PctDiff'] = (sold_df['SalePrice'] - sold_df['RMV']) / sold_df['RMV'] * 100

        print(f"\n--- Comparison: Asking vs Sold ---")
        print(f"  {'Metric':<30} {'Asking':>12} {'Sold':>12}")
        print(f"  {'-'*54}")
        print(f"  {'Sample size':<30} {len(existing):>12} {len(sold_df):>12}")
        print(f"  {'Median price':<30} ${existing['AskingPrice'].median():>11,.0f} ${sold_df['SalePrice'].median():>11,.0f}")
        print(f"  {'Median RMV':<30} ${existing['RMV'].median():>11,.0f} ${sold_df['RMV'].median():>11,.0f}")
        print(f"  {'Median ratio to RMV':<30} {existing['AskToRMV'].median():>12.3f} {sold_df['SaleToRMV'].median():>12.3f}")
        print(f"  {'Mean ratio to RMV':<30} {existing['AskToRMV'].mean():>12.3f} {sold_df['SaleToRMV'].mean():>12.3f}")
        print(f"  {'Median % diff from RMV':<30} {existing['PctDiff'].median():>+11.1f}% {sold_df['PctDiff'].median():>+11.1f}%")
        print(f"  {'Correlation (r)':<30} {r:>12.3f} {stats.pearsonr(sold_df['RMV'], sold_df['SalePrice'])[0]:>12.3f}")

        sold_slope = stats.linregress(sold_df['RMV'], sold_df['SalePrice']).slope
        print(f"  {'Regression slope':<30} {slope:>12.3f} {sold_slope:>12.3f}")

    # --- Detailed listing ---
    print(f"\n--- All Listings (sorted by Ask/RMV ratio) ---")
    existing_sorted = existing.sort_values('AskToRMV', ascending=False)
    print(f"{'Address':<35} {'Asking':>12} {'RMV':>12} {'Ratio':>7} {'%Diff':>7} {'YrBlt':>6}")
    print("-" * 85)
    for _, row in existing_sorted.iterrows():
        yb = int(row['YearBuilt']) if pd.notna(row['YearBuilt']) else 0
        print(f"{row['Address']:<35} ${row['AskingPrice']:>11,.0f} ${row['RMV']:>11,.0f} "
              f"{row['AskToRMV']:>7.3f} {row['PctDiff']:>+6.1f}% {yb:>6}")

    # --- Visualization ---
    if sold_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.reshape(1, 2)

    fig.suptitle('Gresham: Asking Price vs County RMV (Current For-Sale Listings)',
                 fontsize=14, fontweight='bold')

    # 1. Scatter: Asking vs RMV
    ax = axes[0, 0]
    max_val = max(existing['AskingPrice'].max(), existing['RMV'].max()) * 1.05
    min_val = min(existing['AskingPrice'].min(), existing['RMV'].min()) * 0.95
    ax.scatter(existing['RMV'] / 1000, existing['AskingPrice'] / 1000,
               alpha=0.6, s=30, color='darkorange', edgecolors='white', linewidth=0.3)
    ax.plot([min_val / 1000, max_val / 1000], [min_val / 1000, max_val / 1000],
            'g--', alpha=0.5, label='Ask = RMV')

    # Regression line
    x_line = np.linspace(existing['RMV'].min(), existing['RMV'].max(), 100)
    ax.plot(x_line / 1000, (slope * x_line + intercept) / 1000,
            'r-', alpha=0.7, label=f'Regression (slope={slope:.2f})')
    ax.set_xlabel('County RMV ($K)')
    ax.set_ylabel('Asking Price ($K)')
    ax.set_title(f'Asking Price vs RMV (n={len(existing)}, r={r:.3f})')
    ax.legend(fontsize=9)

    # 2. Distribution of Ask/RMV ratio
    ax = axes[0, 1]
    ax.hist(existing['PctDiff'], bins=20, color='darkorange', edgecolor='white', alpha=0.7,
            label='Asking')
    ax.axvline(0, color='green', linestyle='--', alpha=0.5, label='RMV = Ask')
    ax.axvline(existing['PctDiff'].median(), color='red', linestyle='--', alpha=0.7,
               label=f'Median: {existing["PctDiff"].median():+.1f}%')
    ax.set_xlabel('% Difference from RMV')
    ax.set_ylabel('Count')
    ax.set_title('How Far Are Asking Prices from RMV?')
    ax.legend(fontsize=9)

    if sold_df is not None:
        # 3. Overlaid distributions: asking vs sold
        ax = axes[1, 0]
        bins = np.linspace(-40, 40, 30)
        ax.hist(sold_df['PctDiff'], bins=bins, alpha=0.5, color='steelblue',
                edgecolor='white', label=f'Sold (n={len(sold_df)})', density=True)
        ax.hist(existing['PctDiff'], bins=bins, alpha=0.5, color='darkorange',
                edgecolor='white', label=f'Asking (n={len(existing)})', density=True)
        ax.axvline(0, color='green', linestyle='--', alpha=0.5)
        ax.axvline(sold_df['PctDiff'].median(), color='steelblue', linestyle='--', alpha=0.7,
                   label=f'Sold median: {sold_df["PctDiff"].median():+.1f}%')
        ax.axvline(existing['PctDiff'].median(), color='darkorange', linestyle='--', alpha=0.7,
                   label=f'Ask median: {existing["PctDiff"].median():+.1f}%')
        ax.set_xlabel('% Difference from RMV')
        ax.set_ylabel('Density')
        ax.set_title('Asking vs Sold: Distance from RMV')
        ax.legend(fontsize=8)

        # 4. Box plot comparison
        ax = axes[1, 1]
        bp = ax.boxplot(
            [existing['AskToRMV'].values, sold_df['SaleToRMV'].values],
            tick_labels=['Asking\n(for-sale)', 'Sold\n(recent sales)'],
            patch_artist=True
        )
        bp['boxes'][0].set_facecolor('#ffcc80')
        bp['boxes'][1].set_facecolor('#90caf9')
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.3, label='RMV')
        ax.set_ylabel('Price / RMV Ratio')
        ax.set_title('Asking vs Sold: Price/RMV Distribution')
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, 'results', 'figures', 'asking_vs_rmv.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {out_path}")

    # Save enriched data
    csv_path = os.path.join(BASE_DIR, 'results', 'reports', 'gresham_forsale_enriched.csv')
    existing.to_csv(csv_path, index=False)
    print(f"Enriched data saved to {csv_path}")


def main():
    if len(sys.argv) < 2:
        csv_path = os.path.join(BASE_DIR, 'data', 'raw', 'redfin_Gresham_For_Sale.csv')
    else:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = load_forsale_csv(csv_path)
    enriched = enrich_listings(df)
    analyze_asking_vs_rmv(enriched)


if __name__ == '__main__':
    main()
