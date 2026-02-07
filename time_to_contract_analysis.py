"""
Analyze "Time to Contract" for Pending listings in Gresham and Salem.
Reuses logic from gresham_data_collector and time_to_sell_both.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import re

# Import enrichment logic
from gresham_data_collector import (
    load_redfin_csv, enrich_with_rmv
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

def ensure_dirs():
    for d in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)

def enrich_gresham_pending(csv_path):
    """Load Gresham pending CSV and enrich with RMV efficiently."""
    print(f"\n--- Loading Gresham Pending data from {os.path.basename(csv_path)} ---")
    
    # 1. Load Redfin CSV (already filtered to Gresham inside load_redfin_csv)
    df = load_redfin_csv(csv_path)
    
    # Identify DOM column
    dom_col = next((c for c in df.columns if 'DAYS ON MARKET' in c.upper()), None)
    if not dom_col:
        print("WARNING: Could not find 'DAYS ON MARKET' column. Assuming NaN for DOM.")
        df['DOM'] = np.nan
    else:
        # Convert explicitly to numeric
        df['DOM'] = pd.to_numeric(df[dom_col], errors='coerce')
        
    print(f"Loaded {len(df)} records. Found DOM column: {dom_col}")

    # 2. Enrich with RMV (one pass!)
    # enrich_with_rmv returns a NEW dataframe with standard columns, same row count/order
    enriched = enrich_with_rmv(df)
    
    # 3. Merge DOM back into enriched dataframe
    # Since row order is preserved, we can just assign the column
    enriched['DOM'] = df['DOM'].values
    enriched['ListingPrice'] = enriched['SalePrice'] # Pending price is listing
    
    # Filter to valid matches
    valid = enriched[enriched['RMV'] > 0].copy()
    print(f"Matched {len(valid)} of {len(df)} pending listings with RMV.")
    
    return valid

def load_salem_pending(csv_path):
    """Load Salem pending CSV and enrich with ORCATS RMV."""
    print(f"\n--- Loading Salem Pending data from {os.path.basename(csv_path)} ---")
    
    # Load Redfin
    redfin = pd.read_csv(csv_path)
    if 'ADDRESS' in redfin.columns:
        redfin = redfin.dropna(subset=['ADDRESS'])
    else:
        # Try finding address col
        addr_col = next((c for c in redfin.columns if 'ADDRESS' in c.upper()), None)
        if addr_col:
            redfin = redfin.dropna(subset=[addr_col])
        else:
            print("Error: No ADDRESS column found in Salem CSV")
            return pd.DataFrame()

    # Parse Price and DOM
    price_col = next((c for c in redfin.columns if 'PRICE' in c.upper()), 'PRICE')
    redfin['ListingPrice'] = pd.to_numeric(
        redfin[price_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce'
    )
    
    dom_col = next((c for c in redfin.columns if 'DAYS ON MARKET' in c.upper()), None)
    if dom_col:
        redfin['DOM'] = pd.to_numeric(redfin[dom_col], errors='coerce')
    else:
        redfin['DOM'] = np.nan
        
    print(f"Loaded {len(redfin)} Salem pending listings.")

    # Load ORCATS
    orcats_path = os.path.join(BASE_DIR, 'data/raw/comprehensive/ORCATS999_(NEW).csv')
    print("Loading ORCATS assessment data...")
    orcats = pd.read_csv(orcats_path, low_memory=False)
    
    # Find address col (SITUSADDR usually)
    addr_col = next((c for c in orcats.columns if 'SITUS' in c.upper() and 'ADDR' in c.upper()), None)
    if not addr_col:
        print("Error: Could not find SITUS address column in ORCATS.")
        return pd.DataFrame()
        
    orcats['RMVLAND'] = pd.to_numeric(orcats.get('RMVLAND', 0), errors='coerce').fillna(0)
    orcats['RMVIMPR'] = pd.to_numeric(orcats.get('RMVIMPR', 0), errors='coerce').fillna(0)
    orcats['TotalRMV'] = orcats['RMVLAND'] + orcats['RMVIMPR']
    orcats['NormAddr'] = orcats[addr_col].astype(str).str.upper().str.strip()
    
    records = []
    matched = 0
    
    print("Matching Salem addresses...")
    
    # Pre-filter ORCATS to just needed columns for speed
    orcats_slim = orcats[['NormAddr', 'TotalRMV', 'RMVLAND', 'RMVIMPR']].copy()
    
    for i, row in redfin.iterrows():
        addr = str(row['ADDRESS']).upper().strip() if 'ADDRESS' in row else ''
        house_match = re.match(r'^(\d+)\s+(.+)', addr)
        
        matches = pd.DataFrame()
        if house_match:
            house_num = house_match.group(1)
            street = house_match.group(2).strip()
            # Try 1: Exact start
            search_str = f"{house_num} {street}"
            matches = orcats_slim[orcats_slim['NormAddr'].str.startswith(search_str, na=False)]
            
            # Try 2: Without suffix if no match
            if len(matches) == 0:
                 street_core = re.sub(r'\s+[NSEW]$', '', street) # Remove trailing directional
                 search_str2 = f"{house_num} {street_core}"
                 matches = orcats_slim[orcats_slim['NormAddr'].str.contains(search_str2, na=False)]
        
        rmv = None
        if not matches.empty:
            valid_matches = matches[matches['TotalRMV'] > 0]
            if not valid_matches.empty:
                best = valid_matches.sort_values('TotalRMV', ascending=False).iloc[0]
                rmv = best['TotalRMV']
                matched += 1
            elif not matches.empty:
                # Matches exist but RMV=0
                best = matches.iloc[0]
                rmv = best['TotalRMV'] # 0
        
        records.append({
            'Address': addr,
            'City': 'SALEM',
            'ListingPrice': row['ListingPrice'],
            'DOM': row['DOM'],
            'RMV': rmv,
            'Source': 'redfin_pending'
        })
        
        if (i+1) % 50 == 0:
            print(f"  Matched {matched}/{i+1}...")

    print(f"Final: Matched {matched} of {len(redfin)} Salem pending listings.")
    return pd.DataFrame(records)

def analyze_and_plot(gresham_df, salem_df):
    """Analyze Time to Contract vs Price/RMV ratio."""
    print("\n" + "="*60)
    print("PENDING SALES ANALYSIS (Time to Contract)")
    print("="*60)
    
    # Combine
    gresham_df['Dataset'] = 'Gresham'
    salem_df['Dataset'] = 'Salem'
    
    # Filter valid
    combined = pd.concat([gresham_df, salem_df], ignore_index=True)
    valid = combined[
        (combined['ListingPrice'] > 0) & 
        (combined['RMV'] > 0) & 
        (combined['DOM'].notna())
    ].copy()
    
    valid['Ratio'] = valid['ListingPrice'] / valid['RMV']
    valid['PctDiff'] = (valid['ListingPrice'] - valid['RMV']) / valid['RMV'] * 100
    
    print(f"Total valid pending records for analysis: {len(valid)}")
    print(f"  Gresham: {len(valid[valid['Dataset'] == 'Gresham'])}")
    print(f"  Salem:   {len(valid[valid['Dataset'] == 'Salem'])}")
    
    # --- Statistics ---
    for city in ['Gresham', 'Salem']:
        subset = valid[valid['Dataset'] == city]
        if len(subset) < 5: 
            continue
            
        print(f"\n{city} Statistics (Pending Sales):")
        print(f"  Median DOM (Time to Contract): {subset['DOM'].median():.0f} days")
        print(f"  Mean DOM:   {subset['DOM'].mean():.1f} days")
        print(f"  Median Ratio (List/RMV): {subset['Ratio'].median():.3f}")
        
        # Correlation
        r, p = stats.pearsonr(subset['Ratio'], subset['DOM'])
        rho, rho_p = stats.spearmanr(subset['Ratio'], subset['DOM'])
        
        print(f"  Correlation (Ratio vs DOM): Pearson r={r:.3f}, Spearman rho={rho:.3f}")
        
        # Breakdown by pricing strategy
        under = subset[subset['Ratio'] < 1.0]
        at = subset[(subset['Ratio'] >= 1.0) & (subset['Ratio'] <= 1.1)]
        over = subset[subset['Ratio'] > 1.1]
        
        print(f"  Time to Contract by Strategy:")
        if len(under) > 0: print(f"    Under RMV (<1.0): {under['DOM'].median():.0f} days (n={len(under)})")
        if len(at) > 0:    print(f"    At RMV (1.0-1.1): {at['DOM'].median():.0f} days (n={len(at)})")
        if len(over) > 0:  print(f"    Over RMV (>1.1):  {over['DOM'].median():.0f} days (n={len(over)})")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Time to Contract: How Pricing Affects Speed of Sale (Pending Listings)', fontsize=14, fontweight='bold')
    
    # Scatter Plot
    ax = axes[0]
    colors = {'Gresham': 'darkorange', 'Salem': 'steelblue'}
    
    for city in ['Salem', 'Gresham']: 
        subset = valid[valid['Dataset'] == city]
        if len(subset) == 0: continue
        
        ax.scatter(subset['Ratio'], subset['DOM'], label=city, color=colors[city], alpha=0.6, edgecolors='white', s=50)
        
        # Trend Line
        if len(subset) > 5:
            z = np.polyfit(subset['Ratio'], subset['DOM'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(subset['Ratio'].min(), subset['Ratio'].max(), 100)
            ax.plot(x_range, p(x_range), color=colors[city], linestyle='--', linewidth=2, alpha=0.8)

    ax.set_xlabel('List Price / RMV Ratio')
    ax.set_ylabel('Days to Pending (DOM)')
    ax.set_title('Days to Contract vs. Listing Premium')
    ax.axvline(1.0, color='gray', linestyle=':', label='RMV Match')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box Plot
    ax2 = axes[1]
    
    strategies = [
        ('Priced Below RMV', valid[valid['Ratio'] < 1.0]['DOM']),
        ('Priced Near RMV\n(1.0 - 1.1x)', valid[(valid['Ratio'] >= 1.0) & (valid['Ratio'] <= 1.1)]['DOM']),
        ('Priced High\n(>1.1x)', valid[valid['Ratio'] > 1.1]['DOM'])
    ]
    # Filter empty
    strategies = [s for s in strategies if len(s[1]) > 0]
    
    bp = ax2.boxplot([s[1] for s in strategies], labels=[s[0] for s in strategies], patch_artist=True)
    
    box_colors = ['lightblue', 'lightgreen', 'salmon']
    for i, patch in enumerate(bp['boxes']):
        if i < len(box_colors):
            patch.set_facecolor(box_colors[i])
        
    ax2.set_ylabel('Days to Pending')
    ax2.set_title('Speed of Sale by Pricing Strategy (Combined)')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(FIGURES_DIR, 'time_to_contract_pending.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {out_path}")
    
    csv_path = os.path.join(REPORTS_DIR, 'pending_analysis_data.csv')
    valid.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def main():
    ensure_dirs()
    
    gresham_csv = os.path.join(BASE_DIR, 'data/raw/redfin_Gresham_pending.csv')
    salem_csv = os.path.join(BASE_DIR, 'data/raw/redfin_Salem_pending.csv')
    
    gresham_data = pd.DataFrame()
    salem_data = pd.DataFrame()
    
    if os.path.exists(gresham_csv):
        gresham_data = enrich_gresham_pending(gresham_csv)
    else:
        print(f"File not found: {gresham_csv}")
        
    if os.path.exists(salem_csv):
        salem_data = load_salem_pending(salem_csv)
    else:
        print(f"File not found: {salem_csv}")
        
    if not gresham_data.empty or not salem_data.empty:
        analyze_and_plot(gresham_data, salem_data)
    else:
        print("No data collected.")

if __name__ == '__main__':
    main()
