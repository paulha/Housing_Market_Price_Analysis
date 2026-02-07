"""Compare price vs time-to-sell for both Gresham and Salem for-sale listings."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_gresham_forsale():
    """Load already-enriched Gresham for-sale data."""
    df = pd.read_csv(os.path.join(BASE_DIR, 'results/reports/gresham_forsale_enriched.csv'))
    df['DOM'] = pd.to_numeric(df.get('DaysOnMarket'), errors='coerce')
    valid = df[df['DOM'].notna() & (df['RMV'] > 0)].copy()
    valid['AskToRMV'] = valid['AskingPrice'] / valid['RMV']
    valid['PctAboveRMV'] = (valid['AskingPrice'] - valid['RMV']) / valid['RMV'] * 100
    valid['City'] = 'Gresham'
    print(f"Gresham: {len(valid)} listings with DOM + RMV")
    return valid[['Address', 'AskingPrice', 'RMV', 'RMVLAND', 'RMVIMPR',
                  'AskToRMV', 'PctAboveRMV', 'DOM', 'YearBuilt', 'SqFt', 'City']]


def normalize_salem_address(addr):
    """Normalize a Salem address for ORCATS matching."""
    if not addr or not isinstance(addr, str):
        return ''
    addr = addr.upper().strip()
    # Redfin uses "S" at end: "3030 Argyle Dr S" -> need to match ORCATS situs
    return addr


def load_salem_forsale():
    """Load Salem for-sale CSV and enrich with ORCATS999 RMV."""
    # Load Redfin
    redfin = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/redfin_Salem_For_Sale.csv'))
    redfin = redfin.dropna(subset=['ADDRESS'])
    redfin = redfin[redfin['CITY'].astype(str).str.upper().str.strip() == 'SALEM']

    redfin['AskingPrice'] = pd.to_numeric(
        redfin['PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    redfin['DOM'] = pd.to_numeric(redfin['DAYS ON MARKET'], errors='coerce')
    redfin = redfin[redfin['AskingPrice'] > 0].copy()
    print(f"Salem Redfin: {len(redfin)} listings, {redfin['DOM'].notna().sum()} with DOM")

    # Load ORCATS999 for RMV matching
    orcats_path = os.path.join(BASE_DIR, 'data/raw/comprehensive/ORCATS999_(NEW).csv')
    orcats = pd.read_csv(orcats_path, low_memory=False)

    # Parse ORCATS address: SITUSADDR field
    # Find the right column name
    addr_col = None
    for col in orcats.columns:
        if 'SITUS' in col.upper() and 'ADDR' in col.upper():
            addr_col = col
            break
    if not addr_col:
        # Try broader search
        for col in orcats.columns:
            if 'SITUS' in col.upper():
                addr_col = col
                break

    if addr_col:
        print(f"  ORCATS address column: {addr_col}")
    else:
        print(f"  ORCATS columns: {list(orcats.columns)[:20]}")
        print("  Cannot find address column!")
        return pd.DataFrame()

    orcats['RMVLAND'] = pd.to_numeric(orcats.get('RMVLAND', pd.Series()), errors='coerce')
    orcats['RMVIMPR'] = pd.to_numeric(orcats.get('RMVIMPR', pd.Series()), errors='coerce')
    orcats['TotalRMV'] = orcats['RMVLAND'].fillna(0) + orcats['RMVIMPR'].fillna(0)
    orcats['NormAddr'] = orcats[addr_col].astype(str).str.upper().str.strip()

    # Match Redfin addresses to ORCATS
    # Redfin: "3030 Argyle Dr S, Salem, OR" -> normalize
    # ORCATS: "3030 ARGYLE DR S SALEM OR 97302"
    records = []
    matched = 0

    for _, row in redfin.iterrows():
        addr = str(row['ADDRESS']).upper().strip()
        # Try to match: extract house number and street name
        # Look for ORCATS entries containing the address
        house_match = re.match(r'^(\d+)\s+(.+)', addr)
        if not house_match:
            continue

        house_num = house_match.group(1)
        street = house_match.group(2).strip()
        # Remove trailing directional that Redfin puts at end
        # "Argyle Dr S" -> search for "3030 ARGYLE DR S"
        search_str = f"{house_num} {street}"

        matches = orcats[orcats['NormAddr'].str.startswith(search_str)]

        if len(matches) == 0:
            # Try without trailing directional
            street_core = re.sub(r'\s+[NSEW]$', '', street)
            search_str2 = f"{house_num} {street_core}"
            matches = orcats[orcats['NormAddr'].str.contains(search_str2, na=False)]

        rmv = None
        rmvland = None
        rmvimpr = None
        if len(matches) >= 1:
            best = matches.iloc[0]
            rmv = best['TotalRMV']
            rmvland = best['RMVLAND']
            rmvimpr = best['RMVIMPR']
            if rmv > 0:
                matched += 1

        records.append({
            'Address': addr,
            'AskingPrice': row['AskingPrice'],
            'RMV': rmv if rmv and rmv > 0 else None,
            'RMVLAND': rmvland,
            'RMVIMPR': rmvimpr,
            'DOM': row['DOM'],
            'YearBuilt': row.get('YEAR BUILT'),
            'SqFt': row.get('SQUARE FEET'),
            'City': 'Salem',
        })

    result = pd.DataFrame(records)
    valid = result[(result['RMV'] > 0) & (result['DOM'].notna())].copy()
    valid['AskToRMV'] = valid['AskingPrice'] / valid['RMV']
    valid['PctAboveRMV'] = (valid['AskingPrice'] - valid['RMV']) / valid['RMV'] * 100

    print(f"  Matched {matched}/{len(redfin)} with RMV")
    print(f"  {len(valid)} with both DOM + RMV")
    return valid[['Address', 'AskingPrice', 'RMV', 'RMVLAND', 'RMVIMPR',
                  'AskToRMV', 'PctAboveRMV', 'DOM', 'YearBuilt', 'SqFt', 'City']]


def analyze_city(df, city_name):
    """Print DOM analysis for one city."""
    print(f"\n{'='*60}")
    print(f"{city_name}: {len(df)} for-sale listings")
    print(f"{'='*60}")

    print(f"  DOM: median {df['DOM'].median():.0f}, mean {df['DOM'].mean():.0f}, "
          f"range {df['DOM'].min():.0f}-{df['DOM'].max():.0f} days")
    print(f"  Asking price: median ${df['AskingPrice'].median():,.0f}")
    print(f"  Ask/RMV ratio: median {df['AskToRMV'].median():.3f}")

    r_price, p_price = stats.pearsonr(df['AskingPrice'], df['DOM'])
    r_ratio, p_ratio = stats.pearsonr(df['AskToRMV'], df['DOM'])
    rho, p_rho = stats.spearmanr(df['AskingPrice'], df['DOM'])
    rho_r, p_rho_r = stats.spearmanr(df['AskToRMV'], df['DOM'])

    print(f"\n  Correlations with DOM:")
    print(f"    Price vs DOM:     Pearson r={r_price:.3f} (p={p_price:.3f}), Spearman rho={rho:.3f} (p={p_rho:.3f})")
    print(f"    Ask/RMV vs DOM:   Pearson r={r_ratio:.3f} (p={p_ratio:.3f}), Spearman rho={rho_r:.3f} (p={p_rho_r:.3f})")

    # By pricing strategy
    below = df[df['AskToRMV'] < 1.0]
    near = df[(df['AskToRMV'] >= 1.0) & (df['AskToRMV'] < 1.10)]
    above = df[df['AskToRMV'] >= 1.10]

    print(f"\n  DOM by pricing strategy:")
    print(f"  {'Strategy':<25} {'n':>4} {'Med DOM':>9} {'Mean DOM':>10}")
    print(f"  {'-'*50}")
    for label, subset in [('Below RMV', below), ('Near RMV (1.0-1.10)', near), ('Above RMV (>=1.10)', above)]:
        if len(subset) > 0:
            print(f"  {label:<25} {len(subset):>4} {subset['DOM'].median():>9.0f} {subset['DOM'].mean():>10.0f}")

    return r_price, r_ratio, rho, rho_r


def main():
    print("=== Price vs Time to Sell: Salem vs Gresham ===\n")

    gresham = load_gresham_forsale()

    print()
    salem = load_salem_forsale()

    if len(salem) == 0:
        print("Could not match Salem data.")
        return

    g_stats = analyze_city(gresham, 'GRESHAM')
    s_stats = analyze_city(salem, 'SALEM')

    # Cross-city comparison
    combined = pd.concat([gresham, salem], ignore_index=True)

    print(f"\n{'='*60}")
    print(f"COMPARISON: Salem vs Gresham")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<30} {'Gresham':>12} {'Salem':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Listings':<30} {len(gresham):>12} {len(salem):>12}")
    print(f"  {'Median asking price':<30} ${gresham['AskingPrice'].median():>11,.0f} ${salem['AskingPrice'].median():>11,.0f}")
    print(f"  {'Median RMV':<30} ${gresham['RMV'].median():>11,.0f} ${salem['RMV'].median():>11,.0f}")
    print(f"  {'Median Ask/RMV':<30} {gresham['AskToRMV'].median():>12.3f} {salem['AskToRMV'].median():>12.3f}")
    print(f"  {'Median DOM':<30} {gresham['DOM'].median():>12.0f} {salem['DOM'].median():>12.0f}")
    print(f"  {'Mean DOM':<30} {gresham['DOM'].mean():>12.0f} {salem['DOM'].mean():>12.0f}")
    print(f"  {'Price-DOM Spearman rho':<30} {g_stats[2]:>12.3f} {s_stats[2]:>12.3f}")
    print(f"  {'Ask/RMV-DOM Spearman rho':<30} {g_stats[3]:>12.3f} {s_stats[3]:>12.3f}")

    # DOM by price tier — both cities
    print(f"\n  DOM by price tier (both cities):")
    tiers = [
        ('Under $500K', 0, 500_000),
        ('$500K - $700K', 500_000, 700_000),
        ('$700K - $1M', 700_000, 1_000_000),
        ('$1M+', 1_000_000, 99_000_000),
    ]
    print(f"  {'Tier':<16} {'Gresham n':>10} {'Gresham DOM':>12} {'Salem n':>9} {'Salem DOM':>11}")
    print(f"  {'-'*60}")
    for label, lo, hi in tiers:
        g_sub = gresham[(gresham['AskingPrice'] >= lo) & (gresham['AskingPrice'] < hi)]
        s_sub = salem[(salem['AskingPrice'] >= lo) & (salem['AskingPrice'] < hi)]
        g_dom = f"{g_sub['DOM'].median():.0f}" if len(g_sub) > 0 else "-"
        s_dom = f"{s_sub['DOM'].median():.0f}" if len(s_sub) > 0 else "-"
        print(f"  {label:<16} {len(g_sub):>10} {g_dom:>12} {len(s_sub):>9} {s_dom:>11}")

    # Mann-Whitney: do the two cities have different DOM?
    u_stat, u_p = stats.mannwhitneyu(gresham['DOM'], salem['DOM'], alternative='two-sided')
    print(f"\n  DOM comparison (Mann-Whitney): p = {u_p:.4f}")
    if u_p < 0.05:
        faster = 'Salem' if salem['DOM'].median() < gresham['DOM'].median() else 'Gresham'
        print(f"  => {faster} homes sell significantly faster")
    else:
        print(f"  => No significant difference in time to sell")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Price vs Time on Market — Salem vs Gresham For-Sale Listings',
                 fontsize=14, fontweight='bold')

    colors = {'Gresham': 'darkorange', 'Salem': 'steelblue'}

    # 1. Asking price vs DOM — both cities
    ax = axes[0, 0]
    for city, color in colors.items():
        subset = combined[combined['City'] == city]
        ax.scatter(subset['AskingPrice'] / 1000, subset['DOM'], alpha=0.6, s=35,
                   color=color, edgecolors='white', linewidth=0.3, label=city)
    ax.set_xlabel('Asking Price ($K)')
    ax.set_ylabel('Days on Market')
    ax.set_title('Asking Price vs DOM')
    ax.legend(fontsize=9)

    # 2. Ask/RMV ratio vs DOM — both cities
    ax = axes[0, 1]
    for city, color in colors.items():
        subset = combined[combined['City'] == city]
        ax.scatter(subset['AskToRMV'], subset['DOM'], alpha=0.6, s=35,
                   color=color, edgecolors='white', linewidth=0.3, label=city)
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.3)
    ax.set_xlabel('Asking Price / RMV Ratio')
    ax.set_ylabel('Days on Market')
    ax.set_title('Overpricing vs DOM')
    ax.legend(fontsize=9)

    # 3. DOM distributions overlaid
    ax = axes[0, 2]
    bins = np.linspace(0, max(combined['DOM'].max(), 350), 20)
    ax.hist(gresham['DOM'], bins=bins, alpha=0.5, color='darkorange',
            edgecolor='white', label=f'Gresham (med={gresham["DOM"].median():.0f}d)', density=True)
    ax.hist(salem['DOM'], bins=bins, alpha=0.5, color='steelblue',
            edgecolor='white', label=f'Salem (med={salem["DOM"].median():.0f}d)', density=True)
    ax.set_xlabel('Days on Market')
    ax.set_ylabel('Density')
    ax.set_title('DOM Distribution')
    ax.legend(fontsize=9)

    # 4. Box plot: DOM by city
    ax = axes[1, 0]
    bp = ax.boxplot(
        [gresham['DOM'].values, salem['DOM'].values],
        tick_labels=[f'Gresham\n(n={len(gresham)})', f'Salem\n(n={len(salem)})'],
        patch_artist=True
    )
    bp['boxes'][0].set_facecolor('#ffcc80')
    bp['boxes'][1].set_facecolor('#90caf9')
    ax.set_ylabel('Days on Market')
    ax.set_title('DOM by City')

    # 5. Box plot: DOM by pricing strategy (both cities combined)
    ax = axes[1, 1]
    below = combined[combined['AskToRMV'] < 1.0]
    near = combined[(combined['AskToRMV'] >= 1.0) & (combined['AskToRMV'] < 1.10)]
    above = combined[combined['AskToRMV'] >= 1.10]
    group_data = []
    group_labels = []
    for label, subset in [('Below RMV', below), ('Near RMV\n(1.0-1.10)', near), ('Above RMV\n(>=1.10)', above)]:
        if len(subset) >= 2:
            group_data.append(subset['DOM'].values)
            group_labels.append(f'{label}\n(n={len(subset)})')
    if group_data:
        bp2 = ax.boxplot(group_data, tick_labels=group_labels, patch_artist=True)
        bcolors = ['#90caf9', '#ffcc80', '#ef9a9a']
        for box, c in zip(bp2['boxes'], bcolors[:len(bp2['boxes'])]):
            box.set_facecolor(c)
    ax.set_ylabel('Days on Market')
    ax.set_title('DOM by Pricing Strategy (Both Cities)')

    # 6. Box plot: DOM by price tier (both cities combined)
    ax = axes[1, 2]
    tier_data = []
    tier_labels = []
    for label, lo, hi in tiers:
        subset = combined[(combined['AskingPrice'] >= lo) & (combined['AskingPrice'] < hi)]
        if len(subset) >= 2:
            tier_data.append(subset['DOM'].values)
            tier_labels.append(f'{label}\n(n={len(subset)})')
    if tier_data:
        bp3 = ax.boxplot(tier_data, tick_labels=tier_labels, patch_artist=True)
        tcolors = ['#a5d6a7', '#66bb6a', '#388e3c', '#1b5e20']
        for box, c in zip(bp3['boxes'], tcolors[:len(bp3['boxes'])]):
            box.set_facecolor(c)
    ax.set_ylabel('Days on Market')
    ax.set_title('DOM by Price Tier (Both Cities)')

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'results/figures/time_to_sell_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {out}")


if __name__ == '__main__':
    main()
