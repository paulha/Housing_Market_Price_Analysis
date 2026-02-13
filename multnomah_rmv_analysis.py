"""
Multnomah County RMV vs. Sale Price Analysis

Validates the Gresham-only findings from rmv_sale_price_analysis.py and
viewcrest_sale_impact.py using county-wide data (Portland, Gresham, Troutdale,
Fairview, Wood Village, Lake Oswego).

Key questions:
  1. Does county RMV predict sale prices across the whole county? (r, slope)
  2. Does the pattern hold at the high end ($800K+ RMV)?
  3. Does land-dominant vs building-dominant matter for Sale/RMV ratio?
  4. What does the expanded dataset predict for 555 SW Viewcrest Dr?

Usage:
    python multnomah_rmv_analysis.py                      # full county analysis
    python multnomah_rmv_analysis.py --high-value          # focus on $800K+ RMV
    python multnomah_rmv_analysis.py --city PORTLAND        # single city
    python multnomah_rmv_analysis.py --region east          # east county (Gresham area)
    python multnomah_rmv_analysis.py --compare-gresham      # side-by-side with Gresham DB
    python multnomah_rmv_analysis.py --viewcrest            # Viewcrest sale price estimates
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTNOMAH_DB = os.path.join(BASE_DIR, 'data', 'multnomah', 'multnomah_research_db.csv')
GRESHAM_DB = os.path.join(BASE_DIR, 'data', 'gresham', 'gresham_research_db.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'results', 'figures')
REPORTS_DIR = os.path.join(BASE_DIR, 'results', 'reports')

# Viewcrest subject property
VIEWCREST_RMV = 1_181_560
VIEWCREST_LAND_PCT = 76.2

# Region groupings
REGIONS = {
    'east': {'GRESHAM', 'TROUTDALE', 'FAIRVIEW', 'WOOD VILLAGE'},
    'portland': {'PORTLAND'},
    'lake_oswego': {'LAKE OSWEGO'},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_multnomah_db(exclude_new_construction=True):
    """
    Load the Multnomah County research database.
    Filters to analysis-ready records (valid RMV and SalePrice, $50K-$5M).
    Optionally excludes new construction (YearBuilt >= 2024).
    """
    if not os.path.exists(MULTNOMAH_DB):
        print(f"ERROR: Database not found: {MULTNOMAH_DB}")
        print("Run 'python multnomah_data_collector.py download && "
              "python multnomah_data_collector.py enrich' first.")
        sys.exit(1)

    df = pd.read_csv(MULTNOMAH_DB)
    df['SaleDate'] = pd.to_datetime(df['SaleDate'], errors='coerce')
    for col in ['SalePrice', 'RMV', 'RMVLAND', 'RMVIMPR', 'YearBuilt', 'SqFt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    total = len(df)
    df = df[(df['SalePrice'] > 0) & (df['RMV'] > 0)].copy()
    df = df[(df['SalePrice'] >= 50_000) & (df['SalePrice'] <= 5_000_000)]
    df = df[(df['RMV'] >= 50_000) & (df['RMV'] <= 5_000_000)]

    if exclude_new_construction:
        before = len(df)
        df = df[df['YearBuilt'] < 2024].copy()
        excluded = before - len(df)
        if excluded > 0:
            print(f"  Excluded {excluded} new construction records (built 2024+)")

    # Derived columns
    df['SaleToRMV'] = df['SalePrice'] / df['RMV']
    df['LandPct'] = df['RMVLAND'] / df['RMV'] * 100
    df['Difference'] = df['SalePrice'] - df['RMV']
    df['PctDiff'] = df['Difference'] / df['RMV'] * 100

    print(f"  Loaded {total} total records, {len(df)} analysis-ready")
    return df


def load_gresham_db(exclude_new_construction=True):
    """Load the original Gresham-only research database for comparison."""
    if not os.path.exists(GRESHAM_DB):
        print(f"  Gresham DB not found: {GRESHAM_DB}")
        return None

    df = pd.read_csv(GRESHAM_DB)
    df['SaleDate'] = pd.to_datetime(df['SaleDate'], errors='coerce')
    for col in ['SalePrice', 'RMV', 'RMVLAND', 'RMVIMPR', 'YearBuilt', 'SqFt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[(df['SalePrice'] > 0) & (df['RMV'] > 0)].copy()
    df = df[(df['SalePrice'] >= 50_000) & (df['SalePrice'] <= 5_000_000)]
    df = df[(df['RMV'] >= 50_000) & (df['RMV'] <= 5_000_000)]

    if exclude_new_construction:
        df = df[df['YearBuilt'] < 2024].copy()

    df['SaleToRMV'] = df['SalePrice'] / df['RMV']
    df['LandPct'] = df['RMVLAND'] / df['RMV'] * 100
    df['Difference'] = df['SalePrice'] - df['RMV']
    df['PctDiff'] = df['Difference'] / df['RMV'] * 100

    return df


# ---------------------------------------------------------------------------
# Statistical analysis functions
# ---------------------------------------------------------------------------

def compute_stats(df, label=''):
    """Compute core RMV vs SalePrice statistics for a dataset."""
    n = len(df)
    if n < 5:
        print(f"  WARNING: {label} has only {n} records — insufficient for analysis")
        return None

    r, p_corr = sp_stats.pearsonr(df['RMV'], df['SalePrice'])
    slope, intercept, r_val, p_reg, std_err = sp_stats.linregress(
        df['RMV'], df['SalePrice'])

    return {
        'label': label,
        'n': n,
        'median_sale': df['SalePrice'].median(),
        'median_rmv': df['RMV'].median(),
        'median_ratio': df['SaleToRMV'].median(),
        'mean_ratio': df['SaleToRMV'].mean(),
        'median_pct_diff': df['PctDiff'].median(),
        'std_pct_diff': df['PctDiff'].std(),
        'correlation': r,
        'r_squared': r ** 2,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_reg,
    }


def print_stats_table(stats_list):
    """Print a formatted comparison table of statistics."""
    if not stats_list:
        return

    print(f"\n  {'Dataset':<22} {'N':>6} {'r':>6} {'R²':>6} {'Slope':>7} "
          f"{'Med.Ratio':>10} {'Med.%Diff':>10} {'Med.RMV':>12}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*7} "
          f"{'-'*10} {'-'*10} {'-'*12}")

    for s in stats_list:
        if s is None:
            continue
        print(f"  {s['label']:<22} {s['n']:>6} {s['correlation']:>6.3f} "
              f"{s['r_squared']:>6.3f} {s['slope']:>7.3f} "
              f"{s['median_ratio']:>10.3f} {s['median_pct_diff']:>9.1f}% "
              f"${s['median_rmv']:>11,.0f}")


# ---------------------------------------------------------------------------
# City breakdown analysis
# ---------------------------------------------------------------------------

def city_breakdown(df):
    """Analyze RMV vs SalePrice by city."""
    print("\n" + "=" * 80)
    print("PER-CITY BREAKDOWN")
    print("=" * 80)

    stats_list = []
    cities = df['City'].value_counts()

    for city, count in cities.items():
        if count < 10:
            continue
        city_df = df[df['City'] == city]
        s = compute_stats(city_df, label=city)
        if s:
            stats_list.append(s)

    # Also compute "All Cities" aggregate
    all_stats = compute_stats(df, label='ALL CITIES')
    if all_stats:
        stats_list.insert(0, all_stats)

    print_stats_table(stats_list)
    return stats_list


# ---------------------------------------------------------------------------
# High-value analysis
# ---------------------------------------------------------------------------

def high_value_analysis(df, threshold=800_000):
    """
    Focus analysis on high-value properties (RMV >= threshold).
    Tests whether land-dominant properties sell at different ratios.
    """
    print("\n" + "=" * 80)
    print(f"HIGH-VALUE ANALYSIS (RMV >= ${threshold:,.0f})")
    print("=" * 80)

    hv = df[df['RMV'] >= threshold].copy()
    print(f"\n  Total high-value properties: {len(hv)}")

    if len(hv) < 10:
        print("  Insufficient data for high-value analysis.")
        return None

    # Overall stats
    hv_stats = compute_stats(hv, label=f'All >= ${threshold/1000:.0f}K')
    print_stats_table([hv_stats])

    # Land-dominant vs building-dominant split
    land_heavy = hv[hv['LandPct'] >= 50]
    bldg_heavy = hv[hv['LandPct'] < 50]

    print(f"\n  Land-dominant (>=50% land): {len(land_heavy)} properties")
    if len(land_heavy) > 0:
        print(f"    Median Sale/RMV: {land_heavy['SaleToRMV'].median():.3f}")
        print(f"    Mean Sale/RMV:   {land_heavy['SaleToRMV'].mean():.3f}")
        print(f"    Median RMV:      ${land_heavy['RMV'].median():,.0f}")

    print(f"\n  Building-dominant (<50% land): {len(bldg_heavy)} properties")
    if len(bldg_heavy) > 0:
        print(f"    Median Sale/RMV: {bldg_heavy['SaleToRMV'].median():.3f}")
        print(f"    Mean Sale/RMV:   {bldg_heavy['SaleToRMV'].mean():.3f}")
        print(f"    Median RMV:      ${bldg_heavy['RMV'].median():,.0f}")

    # Mann-Whitney U test
    if len(land_heavy) >= 5 and len(bldg_heavy) >= 5:
        u_stat, p_val = sp_stats.mannwhitneyu(
            land_heavy['SaleToRMV'], bldg_heavy['SaleToRMV'],
            alternative='two-sided')
        print(f"\n  Mann-Whitney U test (land vs building dominant):")
        print(f"    U = {u_stat:.0f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print(f"    SIGNIFICANT: Land dominance DOES affect sale/RMV ratio")
        else:
            print(f"    NOT significant: Land dominance does NOT affect sale/RMV ratio")
            print(f"    (Both types sell at similar ratios relative to RMV)")
    else:
        p_val = None

    # Correlation: Land% vs Sale/RMV
    if len(hv) >= 10:
        r_land, p_land = sp_stats.pearsonr(hv['LandPct'], hv['SaleToRMV'])
        print(f"\n  Correlation (Land% vs Sale/RMV): r = {r_land:.3f}, p = {p_land:.4f}")

    # Per-city high-value breakdown
    print(f"\n  High-value by city:")
    for city, group in hv.groupby('City'):
        if len(group) >= 3:
            lh = (group['LandPct'] >= 50).sum()
            print(f"    {city:<18} n={len(group):>4}  "
                  f"med.ratio={group['SaleToRMV'].median():.3f}  "
                  f"land-dom={lh}")

    return {
        'hv_stats': hv_stats,
        'n_land_heavy': len(land_heavy),
        'n_bldg_heavy': len(bldg_heavy),
        'mann_whitney_p': p_val,
        'hv_df': hv,
    }


# ---------------------------------------------------------------------------
# Viewcrest predictions
# ---------------------------------------------------------------------------

def viewcrest_estimates(df, threshold=800_000):
    """Estimate Viewcrest sale price using multiple methods."""
    print("\n" + "=" * 80)
    print(f"VIEWCREST SALE PRICE ESTIMATES (RMV = ${VIEWCREST_RMV:,.0f})")
    print("=" * 80)

    # Method 1: Overall regression (all sales)
    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(
        df['RMV'], df['SalePrice'])
    pred_overall = slope * VIEWCREST_RMV + intercept
    print(f"\n  Method 1 — Overall regression (n={len(df)}, slope={slope:.3f}):")
    print(f"    Predicted: ${pred_overall:,.0f}")

    # Method 2: High-value regression ($800K+)
    hv = df[df['RMV'] >= threshold]
    pred_high = None
    if len(hv) >= 10:
        s2, i2, r2, p2, se2 = sp_stats.linregress(hv['RMV'], hv['SalePrice'])
        pred_high = s2 * VIEWCREST_RMV + i2
        print(f"\n  Method 2 — High-value regression (n={len(hv)}, slope={s2:.3f}):")
        print(f"    Predicted: ${pred_high:,.0f}")

    # Method 3: Median ratio of $800K+ properties
    if len(hv) > 0:
        med_ratio = hv['SaleToRMV'].median()
        pred_ratio = med_ratio * VIEWCREST_RMV
        print(f"\n  Method 3 — Median high-value ratio ({med_ratio:.3f}):")
        print(f"    Predicted: ${pred_ratio:,.0f}")

    # Method 4: Land-dominant subset median ratio (all prices)
    land_dom = df[df['LandPct'] >= 50]
    if len(land_dom) >= 5:
        med_land = land_dom['SaleToRMV'].median()
        pred_land = med_land * VIEWCREST_RMV
        print(f"\n  Method 4 — Land-dominant median ratio (n={len(land_dom)}, "
              f"ratio={med_land:.3f}):")
        print(f"    Predicted: ${pred_land:,.0f}")

    # Method 5: Land-dominant AND high-value
    land_hv = hv[hv['LandPct'] >= 50]
    if len(land_hv) >= 5:
        med_land_hv = land_hv['SaleToRMV'].median()
        pred_land_hv = med_land_hv * VIEWCREST_RMV
        print(f"\n  Method 5 — Land-dominant + high-value (n={len(land_hv)}, "
              f"ratio={med_land_hv:.3f}):")
        print(f"    Predicted: ${pred_land_hv:,.0f}")

    # Method 6: East county only (Gresham+Troutdale+Fairview+WV) high-value
    east_cities = {'GRESHAM', 'TROUTDALE', 'FAIRVIEW', 'WOOD VILLAGE'}
    east_hv = hv[hv['City'].isin(east_cities)]
    if len(east_hv) >= 5:
        med_east = east_hv['SaleToRMV'].median()
        pred_east = med_east * VIEWCREST_RMV
        print(f"\n  Method 6 — East county high-value (n={len(east_hv)}, "
              f"ratio={med_east:.3f}):")
        print(f"    Predicted: ${pred_east:,.0f}")

    # Confidence interval based on $800K+ sales
    if len(hv) >= 10:
        ratios = hv['SaleToRMV']
        q25 = ratios.quantile(0.25) * VIEWCREST_RMV
        q75 = ratios.quantile(0.75) * VIEWCREST_RMV
        print(f"\n  Interquartile range (25th-75th pctile of $800K+ ratios):")
        print(f"    ${q25:,.0f} — ${q75:,.0f}")
        print(f"\n  Full range observed:")
        print(f"    ${ratios.min() * VIEWCREST_RMV:,.0f} — "
              f"${ratios.max() * VIEWCREST_RMV:,.0f}")


# ---------------------------------------------------------------------------
# Comparison with Gresham-only database
# ---------------------------------------------------------------------------

def compare_to_gresham(mdf, gdf):
    """Side-by-side comparison of Multnomah County vs Gresham-only findings."""
    print("\n" + "=" * 80)
    print("COMPARISON: MULTNOMAH COUNTY vs GRESHAM-ONLY")
    print("=" * 80)

    m_stats = compute_stats(mdf, label='Multnomah County')
    g_stats = compute_stats(gdf, label='Gresham (original)')

    print_stats_table([m_stats, g_stats])

    # High-value comparison
    print(f"\n  --- High-Value ($800K+ RMV) ---")
    m_hv = mdf[mdf['RMV'] >= 800_000]
    g_hv = gdf[gdf['RMV'] >= 800_000]

    m_hv_stats = compute_stats(m_hv, label='Multnomah $800K+')
    g_hv_stats = compute_stats(g_hv, label='Gresham $800K+')
    print_stats_table([m_hv_stats, g_hv_stats])

    # Land-dominant comparison at high values
    print(f"\n  --- Land-Dominant High-Value ---")
    m_land_hv = m_hv[m_hv['LandPct'] >= 50]
    g_land_hv = g_hv[g_hv['LandPct'] >= 50]

    print(f"  Multnomah: {len(m_land_hv)} land-dominant $800K+ properties")
    if len(m_land_hv) > 0:
        print(f"    Median Sale/RMV: {m_land_hv['SaleToRMV'].median():.3f}")
    print(f"  Gresham:   {len(g_land_hv)} land-dominant $800K+ properties")
    if len(g_land_hv) > 0:
        print(f"    Median Sale/RMV: {g_land_hv['SaleToRMV'].median():.3f}")

    # Mann-Whitney test at high values
    m_land = m_hv[m_hv['LandPct'] >= 50]['SaleToRMV']
    m_bldg = m_hv[m_hv['LandPct'] < 50]['SaleToRMV']
    if len(m_land) >= 5 and len(m_bldg) >= 5:
        u, p = sp_stats.mannwhitneyu(m_land, m_bldg, alternative='two-sided')
        print(f"\n  Mann-Whitney (land vs bldg, Multnomah $800K+): p = {p:.4f}")
        if p < 0.05:
            print(f"    Land dominance SIGNIFICANTLY affects sale/RMV ratio at high values")
        else:
            print(f"    Land dominance does NOT significantly affect ratio (confirms Gresham finding)")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def create_county_overview_figure(df, stats_list):
    """Create county overview visualization with scatter + city comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multnomah County: RMV vs Sale Price Analysis',
                 fontsize=15, fontweight='bold')

    # 1. Overall scatter with regression
    ax = axes[0, 0]
    ax.scatter(df['RMV'] / 1000, df['SalePrice'] / 1000, alpha=0.3, s=15,
               color='steelblue')
    slope, intercept, _, _, _ = sp_stats.linregress(df['RMV'], df['SalePrice'])
    x_line = np.linspace(df['RMV'].min(), df['RMV'].max(), 100)
    ax.plot(x_line / 1000, (slope * x_line + intercept) / 1000, 'r-',
            linewidth=2, label=f'Regression (slope={slope:.3f})')
    lim = max(df['RMV'].max(), df['SalePrice'].max()) / 1000
    ax.plot([0, lim], [0, lim], 'g--', alpha=0.4, label='RMV = Sale')
    ax.set_xlabel('County RMV ($K)')
    ax.set_ylabel('Sale Price ($K)')
    r = df['RMV'].corr(df['SalePrice'])
    ax.set_title(f'RMV vs Sale Price (r={r:.3f}, n={len(df):,})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 2. City comparison bar chart (median ratio)
    ax = axes[0, 1]
    city_data = []
    for city, group in df.groupby('City'):
        if len(group) >= 10:
            city_data.append({
                'city': city,
                'n': len(group),
                'median_ratio': group['SaleToRMV'].median(),
            })
    city_data.sort(key=lambda x: x['median_ratio'])
    cities = [d['city'] for d in city_data]
    ratios = [d['median_ratio'] for d in city_data]
    ns = [d['n'] for d in city_data]
    colors = ['#e74c3c' if c == 'GRESHAM' else 'steelblue' for c in cities]
    bars = ax.barh(range(len(cities)), ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels([f'{c} (n={n})' for c, n in zip(cities, ns)], fontsize=9)
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.4)
    ax.set_xlabel('Median Sale/RMV Ratio')
    ax.set_title('Sale/RMV by City')

    # 3. Distribution of Sale/RMV ratios
    ax = axes[1, 0]
    ax.hist(df['SaleToRMV'].clip(0.3, 2.0), bins=60, color='steelblue',
            edgecolor='white', alpha=0.7)
    ax.axvline(df['SaleToRMV'].median(), color='red', linewidth=2, linestyle='--',
               label=f'Median: {df["SaleToRMV"].median():.3f}')
    ax.axvline(1.0, color='green', linewidth=1.5, linestyle='--',
               label='RMV = Sale', alpha=0.5)
    ax.set_xlabel('Sale Price / RMV')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Sale/RMV Ratios')
    ax.legend(fontsize=9)

    # 4. Stats summary table
    ax = axes[1, 1]
    ax.axis('off')
    if stats_list:
        headers = ['City', 'N', 'r', 'Slope', 'Med.Ratio']
        rows = []
        for s in stats_list:
            if s is None:
                continue
            rows.append([
                s['label'],
                f"{s['n']:,}",
                f"{s['correlation']:.3f}",
                f"{s['slope']:.3f}",
                f"{s['median_ratio']:.3f}",
            ])
        table = ax.table(cellText=rows, colLabels=headers, loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        ax.set_title('Summary by City', fontsize=12, pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'multnomah_county_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {path}")
    plt.close(fig)


def create_high_value_figure(df, hv_result):
    """Create high-value analysis visualization."""
    hv = hv_result['hv_df']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multnomah County: High-Value Properties ($800K+ RMV)',
                 fontsize=15, fontweight='bold')

    # 1. Land% vs Sale/RMV scatter
    ax = axes[0, 0]
    ax.scatter(hv['LandPct'], hv['SaleToRMV'], alpha=0.4, s=25, color='steelblue')
    z = np.polyfit(hv['LandPct'], hv['SaleToRMV'], 1)
    x_line = np.linspace(hv['LandPct'].min(), hv['LandPct'].max(), 100)
    r_land, _ = sp_stats.pearsonr(hv['LandPct'], hv['SaleToRMV'])
    ax.plot(x_line, np.polyval(z, x_line), 'r-', alpha=0.7,
            label=f'Trend (r={r_land:.3f})')
    ax.axvline(VIEWCREST_LAND_PCT, color='red', linestyle='--', alpha=0.5,
               label=f'Viewcrest ({VIEWCREST_LAND_PCT}%)')
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.3)
    ax.set_xlabel('Land % of RMV')
    ax.set_ylabel('Sale Price / RMV')
    ax.set_title(f'Land % vs Sale/RMV (n={len(hv)})')
    ax.legend(fontsize=9)

    # 2. Box plot: land-dom vs bldg-dom
    ax = axes[0, 1]
    land_heavy = hv[hv['LandPct'] >= 50]['SaleToRMV'].values
    bldg_heavy = hv[hv['LandPct'] < 50]['SaleToRMV'].values
    n_lh = len(land_heavy)
    n_bh = len(bldg_heavy)
    bp = ax.boxplot(
        [land_heavy, bldg_heavy],
        tick_labels=[f'Land-heavy\n(>=50%, n={n_lh})',
                     f'Bldg-heavy\n(<50%, n={n_bh})'],
        patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff9999')
    bp['boxes'][1].set_facecolor('#99ccff')
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.3)
    ax.set_ylabel('Sale Price / RMV')
    p_val = hv_result.get('mann_whitney_p')
    p_str = f'p={p_val:.4f}' if p_val is not None else 'N/A'
    ax.set_title(f'Sale/RMV by Land Dominance ({p_str})')

    # 3. RMV vs Sale Price scatter for high-value, colored by land%
    ax = axes[1, 0]
    colors = ['red' if lp >= 50 else 'steelblue' for lp in hv['LandPct']]
    ax.scatter(hv['RMV'] / 1000, hv['SalePrice'] / 1000, c=colors, alpha=0.5, s=30)
    ax.plot([400, 3000], [400, 3000], 'g--', alpha=0.3, label='RMV = Sale')

    # Viewcrest prediction star
    slope, intercept, _, _, _ = sp_stats.linregress(hv['RMV'], hv['SalePrice'])
    pred = slope * VIEWCREST_RMV + intercept
    ax.scatter([VIEWCREST_RMV / 1000], [pred / 1000], color='red', marker='*',
               s=200, zorder=5, label=f'Viewcrest est. ${pred/1000:.0f}K')
    ax.scatter([], [], c='red', alpha=0.5, s=30, label=f'Land-dom (n={n_lh})')
    ax.scatter([], [], c='steelblue', alpha=0.5, s=30, label=f'Bldg-dom (n={n_bh})')
    ax.set_xlabel('RMV ($K)')
    ax.set_ylabel('Sale Price ($K)')
    ax.set_title(f'High-Value Sales by Land Dominance')
    ax.legend(fontsize=8)

    # 4. City breakdown for high-value
    ax = axes[1, 1]
    city_hv = []
    for city, group in hv.groupby('City'):
        if len(group) >= 3:
            city_hv.append({
                'city': city,
                'n': len(group),
                'median_ratio': group['SaleToRMV'].median(),
                'n_land': (group['LandPct'] >= 50).sum(),
            })
    city_hv.sort(key=lambda x: x['n'], reverse=True)

    if city_hv:
        x_pos = range(len(city_hv))
        bars = ax.bar(x_pos, [d['median_ratio'] for d in city_hv],
                       color=['#e74c3c' if d['city'] == 'GRESHAM' else 'steelblue'
                              for d in city_hv], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{d['city']}\nn={d['n']}\n({d['n_land']} land)" for d in city_hv],
            fontsize=8)
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.4)
        ax.set_ylabel('Median Sale/RMV')
        ax.set_title('High-Value Median Ratio by City')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'multnomah_high_value.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def create_comparison_figure(mdf, gdf):
    """Create Multnomah vs Gresham comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Multnomah County vs Gresham-Only: Validation',
                 fontsize=14, fontweight='bold')

    # 1. Overlaid scatter plots
    ax = axes[0]
    ax.scatter(gdf['RMV'] / 1000, gdf['SalePrice'] / 1000, alpha=0.5, s=20,
               color='tab:orange', label=f'Gresham (n={len(gdf)})')
    ax.scatter(mdf['RMV'] / 1000, mdf['SalePrice'] / 1000, alpha=0.15, s=10,
               color='tab:blue', label=f'Multnomah (n={len(mdf)})')
    lim = max(mdf['RMV'].max(), mdf['SalePrice'].max()) / 1000
    ax.plot([0, lim], [0, lim], 'g--', alpha=0.4, label='RMV = Sale')
    ax.set_xlabel('RMV ($K)')
    ax.set_ylabel('Sale Price ($K)')
    ax.set_title('RMV vs Sale Price')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 2. Overlaid ratio distributions
    ax = axes[1]
    bins = np.linspace(0.4, 1.8, 50)
    ax.hist(gdf['SaleToRMV'].clip(0.4, 1.8), bins=bins, alpha=0.6,
            color='tab:orange', label='Gresham', density=True)
    ax.hist(mdf['SaleToRMV'].clip(0.4, 1.8), bins=bins, alpha=0.4,
            color='tab:blue', label='Multnomah', density=True)
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.4)
    ax.set_xlabel('Sale / RMV Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Sale/RMV Ratio Distributions')
    ax.legend(fontsize=9)

    # 3. High-value comparison ($800K+)
    ax = axes[2]
    m_hv = mdf[mdf['RMV'] >= 800_000]['SaleToRMV']
    g_hv = gdf[gdf['RMV'] >= 800_000]['SaleToRMV']

    data_to_plot = []
    labels = []
    if len(g_hv) >= 3:
        data_to_plot.append(g_hv.values)
        labels.append(f'Gresham\n$800K+\n(n={len(g_hv)})')
    if len(m_hv) >= 3:
        data_to_plot.append(m_hv.values)
        labels.append(f'Multnomah\n$800K+\n(n={len(m_hv)})')

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        colors_bp = ['#ff9999', '#99ccff']
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors_bp[i % len(colors_bp)])
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.4)
        ax.set_ylabel('Sale/RMV Ratio')
        ax.set_title('High-Value Comparison')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'multnomah_vs_gresham.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV report export
# ---------------------------------------------------------------------------

def export_reports(df, hv_result, stats_list):
    """Export analysis results to CSV files."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # City summary
    if stats_list:
        summary_rows = []
        for s in stats_list:
            if s is None:
                continue
            summary_rows.append(s)
        summary_df = pd.DataFrame(summary_rows)
        path = os.path.join(REPORTS_DIR, 'multnomah_city_summary.csv')
        summary_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # High-value detail listing
    if hv_result and 'hv_df' in hv_result:
        hv = hv_result['hv_df'].copy()
        cols = ['Address', 'City', 'SalePrice', 'RMV', 'SaleToRMV',
                'LandPct', 'RMVLAND', 'RMVIMPR', 'YearBuilt', 'SqFt']
        cols = [c for c in cols if c in hv.columns]
        hv_out = hv[cols].sort_values('RMV', ascending=False)
        path = os.path.join(REPORTS_DIR, 'multnomah_high_value_detail.csv')
        hv_out.to_csv(path, index=False)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multnomah County RMV vs Sale Price analysis. Validates '
                    'Gresham findings with expanded county-wide data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--high-value', action='store_true',
                        help='Focus analysis on $800K+ RMV properties')
    parser.add_argument('--city', type=str, default=None,
                        help='Filter to a single city (e.g., PORTLAND, GRESHAM)')
    parser.add_argument('--region', type=str, choices=['east', 'portland', 'lake_oswego'],
                        help='Filter to a region (east = Gresham area)')
    parser.add_argument('--compare-gresham', action='store_true',
                        help='Side-by-side comparison with Gresham-only database')
    parser.add_argument('--viewcrest', action='store_true',
                        help='Estimate sale price for 555 SW Viewcrest Dr')
    parser.add_argument('--include-new', action='store_true',
                        help='Include new construction (built 2024+)')
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    exclude_new = not args.include_new

    print("=" * 80)
    print("MULTNOMAH COUNTY RMV vs SALE PRICE ANALYSIS")
    print("=" * 80)

    df = load_multnomah_db(exclude_new_construction=exclude_new)

    # Apply city/region filter
    filter_label = 'ALL MULTNOMAH COUNTY'
    if args.city:
        city_upper = args.city.upper().strip()
        df = df[df['City'] == city_upper]
        filter_label = city_upper
        print(f"  Filtered to {city_upper}: {len(df)} records")
    elif args.region:
        region_cities = REGIONS.get(args.region, set())
        df = df[df['City'].isin(region_cities)]
        filter_label = f"Region: {args.region} ({', '.join(sorted(region_cities))})"
        print(f"  Filtered to {filter_label}: {len(df)} records")

    if len(df) < 10:
        print(f"\nERROR: Only {len(df)} records after filtering. Need at least 10.")
        sys.exit(1)

    # Overall statistics
    print(f"\n--- Overall Statistics ({filter_label}) ---")
    overall_stats = compute_stats(df, label=filter_label)
    print_stats_table([overall_stats])

    # City breakdown (only if not filtered to single city)
    stats_list = None
    if not args.city:
        stats_list = city_breakdown(df)

    # High-value analysis
    hv_result = None
    if args.high_value or args.viewcrest:
        hv_result = high_value_analysis(df)

    # Viewcrest estimates
    if args.viewcrest:
        viewcrest_estimates(df)

    # Comparison with Gresham
    if args.compare_gresham:
        gdf = load_gresham_db(exclude_new_construction=exclude_new)
        if gdf is not None and len(gdf) > 0:
            compare_to_gresham(df, gdf)
            create_comparison_figure(df, gdf)

    # Visualizations
    print(f"\n--- Generating Visualizations ---")
    create_county_overview_figure(df, stats_list or [overall_stats])

    if hv_result:
        create_high_value_figure(df, hv_result)

    # Export reports
    print(f"\n--- Exporting Reports ---")
    export_reports(df, hv_result, stats_list)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
