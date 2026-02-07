"""
Analysis of Real Market Value (RMV) vs. Actual Sale Price
for Gresham and Salem, Oregon

This script performs statistical analysis to determine if county-assessed
Real Market Values correlate properly with actual sale prices.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

def load_orcats_data(filepath):
    """
    Load Marion County ORCATS999 comprehensive assessment data.

    Key columns:
        RMVLAND, RMVIMPR - Real Market Value components
        ACCOUNT_ID - Account number for joining with sales data
        SITUSCITY - City name
        PCLS - Property class code (101-199 = residential)
    """
    df = pd.read_csv(filepath, dtype={'RMVLAND': str, 'RMVIMPR': str})
    print(f"Loaded {len(df):,} total property records")

    # Convert RMV columns
    for col in ['RMVLAND', 'RMVIMPR', 'AV']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Calculate total RMV
    df['RMV'] = df['RMVLAND'].fillna(0) + df['RMVIMPR'].fillna(0)

    return df


def load_sales_data(*filepaths):
    """
    Load Marion County sales CSV files and combine them.

    The sales CSVs have Condition Code which lets us filter to
    confirmed arms-length transactions (code 33).
    """
    frames = []
    for fp in filepaths:
        df = pd.read_csv(fp)
        print(f"  Loaded {len(df):,} rows from {os.path.basename(fp)}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined):,} total sale records")
    return combined


def prepare_salem_data(orcats_df, sales_df):
    """
    Join sales data with assessment data to get RMV + validated sale prices.

    Strategy:
        1. Filter sales to Condition Code 33 (confirmed arms-length sales)
        2. Filter sales to Salem residential addresses
        3. Join with ORCATS999 on Account Number to get RMV values
        4. Apply price range filters
    """
    print("\n--- Preparing Salem data ---")

    # --- Clean up sales data ---
    # Parse Sale Price (has commas)
    sales_df['SalePrice'] = pd.to_numeric(
        sales_df['Sale Price'].astype(str).str.replace(',', ''), errors='coerce'
    )
    sales_df['SaleDate'] = pd.to_datetime(sales_df['Sale Date'], errors='coerce')

    # Condition Code 33 = Confirmed Sale (arms-length, open market)
    sales_df['Condition Code'] = pd.to_numeric(sales_df['Condition Code'], errors='coerce')
    confirmed = sales_df[sales_df['Condition Code'] == 33].copy()
    print(f"  Confirmed sales (code 33): {len(confirmed):,} of {len(sales_df):,}")

    # Filter to residential (Property Class starts with 1)
    confirmed['PropClassNum'] = pd.to_numeric(confirmed['Property Class'], errors='coerce')
    confirmed = confirmed[(confirmed['PropClassNum'] >= 100) & (confirmed['PropClassNum'] < 200)]
    print(f"  Residential confirmed sales: {len(confirmed):,}")

    # Filter to Salem via Situs Address
    confirmed['SitusUpper'] = confirmed['Situs Address'].astype(str).str.upper()
    salem_sales = confirmed[confirmed['SitusUpper'].str.contains('SALEM', na=False)].copy()
    print(f"  Salem residential sales: {len(salem_sales):,}")

    # Deduplicate - keep one row per Sale ID (multi-account sales repeat)
    salem_sales = salem_sales.drop_duplicates(subset='Sale ID', keep='first')
    print(f"  After dedup (unique sales): {len(salem_sales):,}")

    # --- Join with ORCATS for RMV ---
    # Clean account numbers for joining
    salem_sales['Account Number'] = pd.to_numeric(
        salem_sales['Account Number'], errors='coerce'
    )
    orcats_df['ACCOUNT_ID_NUM'] = pd.to_numeric(
        orcats_df['ACCOUNT_ID'], errors='coerce'
    )

    merged = salem_sales.merge(
        orcats_df[['ACCOUNT_ID_NUM', 'RMVLAND', 'RMVIMPR', 'RMV', 'AV']].drop_duplicates('ACCOUNT_ID_NUM'),
        left_on='Account Number',
        right_on='ACCOUNT_ID_NUM',
        how='inner'
    )
    print(f"  Matched with assessment data: {len(merged):,}")

    # --- Final cleaning ---
    df = merged[(merged['SalePrice'] > 0) & (merged['RMV'] > 0)].copy()
    df = df[(df['SalePrice'] >= 50000) & (df['SalePrice'] <= 5000000)]
    df = df[(df['RMV'] >= 50000) & (df['RMV'] <= 5000000)]
    print(f"  After price range filter ($50k-$5M): {len(df):,}")

    return df

def load_gresham_data(filepath, exclude_new_construction=False):
    """
    Load Gresham research database CSV (produced by gresham_data_collector.py).
    Filters to records with valid SalePrice and RMV in the $50k-$5M range.
    Optionally excludes new construction (YearBuilt >= 2024) whose RMV
    reflects land-only assessments before the home was completed.
    """
    df = pd.read_csv(filepath)
    df['SaleDate'] = pd.to_datetime(df['SaleDate'], errors='coerce')
    for col in ['SalePrice', 'RMV', 'RMVLAND', 'RMVIMPR']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce')

    print(f"\n--- Preparing Gresham data ---")
    print(f"  Total records in research DB: {len(df)}")

    df = df[(df['SalePrice'] > 0) & (df['RMV'] > 0)].copy()
    df = df[(df['SalePrice'] >= 50000) & (df['SalePrice'] <= 5000000)]
    df = df[(df['RMV'] >= 50000) & (df['RMV'] <= 5000000)]
    print(f"  Valid prices ($50k-$5M): {len(df)}")

    if exclude_new_construction:
        before = len(df)
        df = df[df['YearBuilt'] < 2024].copy()
        excluded = before - len(df)
        print(f"  Excluded new construction (built 2024+): {excluded} removed, {len(df)} remain")

    print(f"  Analysis-ready: {len(df)}")
    return df


def calculate_statistics(df, sale_price_col='SalePrice', rmv_col='RMV'):
    """
    Calculate key statistics comparing RMV to Sale Price
    """
    df['Difference'] = df[sale_price_col] - df[rmv_col]
    df['Percent_Difference'] = ((df[sale_price_col] - df[rmv_col]) / df[rmv_col] * 100)
    df['Ratio'] = df[sale_price_col] / df[rmv_col]
    
    stats_dict = {
        'Count': len(df),
        'Mean_Sale_Price': df[sale_price_col].mean(),
        'Mean_RMV': df[rmv_col].mean(),
        'Median_Sale_Price': df[sale_price_col].median(),
        'Median_RMV': df[rmv_col].median(),
        'Mean_Difference': df['Difference'].mean(),
        'Median_Difference': df['Difference'].median(),
        'Mean_Percent_Diff': df['Percent_Difference'].mean(),
        'Median_Percent_Diff': df['Percent_Difference'].median(),
        'Std_Percent_Diff': df['Percent_Difference'].std(),
        'Mean_Ratio': df['Ratio'].mean(),
        'Median_Ratio': df['Ratio'].median()
    }
    
    # Calculate correlation
    correlation = df[sale_price_col].corr(df[rmv_col])
    r_squared = correlation ** 2
    
    stats_dict['Correlation'] = correlation
    stats_dict['R_Squared'] = r_squared
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[rmv_col], df[sale_price_col])
    stats_dict['Regression_Slope'] = slope
    stats_dict['Regression_Intercept'] = intercept
    stats_dict['Regression_P_Value'] = p_value
    
    # Test if RMV systematically over/under predicts
    # H0: median percent difference = 0
    t_stat, p_value = stats.ttest_1samp(df['Percent_Difference'].dropna(), 0)
    stats_dict['Bias_T_Stat'] = t_stat
    stats_dict['Bias_P_Value'] = p_value
    
    return stats_dict, df

def create_visualizations(df, city_name, sale_price_col='SalePrice', rmv_col='RMV'):
    """
    Create visualization comparing RMV to Sale Price
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{city_name}: RMV vs. Actual Sale Price Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    ax1.scatter(df[rmv_col], df[sale_price_col], alpha=0.5, s=20)
    
    # Add regression line
    z = np.polyfit(df[rmv_col], df[sale_price_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[rmv_col].min(), df[rmv_col].max(), 100)
    ax1.plot(x_line, p(x_line), "r-", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.0f}')
    
    # Add perfect correlation line
    ax1.plot([df[rmv_col].min(), df[rmv_col].max()], 
             [df[rmv_col].min(), df[rmv_col].max()], 
             'g--', linewidth=2, label='Perfect Correlation (y=x)')
    
    ax1.set_xlabel('County RMV ($)', fontsize=12)
    ax1.set_ylabel('Actual Sale Price ($)', fontsize=12)
    ax1.set_title('RMV vs. Sale Price Scatter Plot', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of percent differences
    ax2 = axes[0, 1]
    ax2.hist(df['Percent_Difference'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Difference')
    ax2.axvline(x=df['Percent_Difference'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {df["Percent_Difference"].median():.1f}%')
    ax2.set_xlabel('Percent Difference ((Sale - RMV) / RMV * 100)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of RMV Prediction Errors', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of ratios
    ax3 = axes[1, 0]
    ax3.boxplot([df['Ratio']], tick_labels=[city_name])
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Match')
    ax3.set_ylabel('Sale Price / RMV Ratio', fontsize=12)
    ax3.set_title('Distribution of Sale Price to RMV Ratios', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residual plot
    ax4 = axes[1, 1]
    predicted = df[rmv_col]  # RMV is the "prediction"
    residuals = df[sale_price_col] - predicted
    ax4.scatter(predicted, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('County RMV ($)', fontsize=12)
    ax4.set_ylabel('Residuals (Sale Price - RMV)', fontsize=12)
    ax4.set_title('Residual Plot', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_analysis_report(city_name, stats_dict):
    """
    Print a comprehensive analysis report
    """
    print("\n" + "="*70)
    print(f"STATISTICAL ANALYSIS REPORT: {city_name}")
    print("="*70)
    
    print(f"\nSAMPLE SIZE: {stats_dict['Count']:,} home sales\n")
    
    print("PRICE STATISTICS:")
    print(f"  Mean Sale Price:     ${stats_dict['Mean_Sale_Price']:,.0f}")
    print(f"  Mean RMV:            ${stats_dict['Mean_RMV']:,.0f}")
    print(f"  Median Sale Price:   ${stats_dict['Median_Sale_Price']:,.0f}")
    print(f"  Median RMV:          ${stats_dict['Median_RMV']:,.0f}")
    
    print("\nPREDICTION ACCURACY:")
    print(f"  Mean Difference:     ${stats_dict['Mean_Difference']:,.0f}")
    print(f"  Median Difference:   ${stats_dict['Median_Difference']:,.0f}")
    print(f"  Mean % Difference:   {stats_dict['Mean_Percent_Diff']:.2f}%")
    print(f"  Median % Difference: {stats_dict['Median_Percent_Diff']:.2f}%")
    print(f"  Std Dev % Diff:      {stats_dict['Std_Percent_Diff']:.2f}%")
    
    print("\nRATIO STATISTICS:")
    print(f"  Mean Ratio (Sale/RMV):   {stats_dict['Mean_Ratio']:.3f}")
    print(f"  Median Ratio (Sale/RMV): {stats_dict['Median_Ratio']:.3f}")
    
    print("\nCORRELATION ANALYSIS:")
    print(f"  Pearson Correlation: {stats_dict['Correlation']:.4f}")
    print(f"  R-Squared:           {stats_dict['R_Squared']:.4f}")
    
    print("\nREGRESSION ANALYSIS:")
    print(f"  Slope:               {stats_dict['Regression_Slope']:.4f}")
    print(f"  Intercept:           ${stats_dict['Regression_Intercept']:,.0f}")
    print(f"  P-Value:             {stats_dict['Regression_P_Value']:.6f}")
    
    print("\nBIAS TEST (Is RMV systematically over/under valued?):")
    print(f"  T-Statistic:         {stats_dict['Bias_T_Stat']:.4f}")
    print(f"  P-Value:             {stats_dict['Bias_P_Value']:.6f}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    
    if stats_dict['Correlation'] >= 0.9:
        corr_quality = "EXCELLENT"
    elif stats_dict['Correlation'] >= 0.8:
        corr_quality = "GOOD"
    elif stats_dict['Correlation'] >= 0.7:
        corr_quality = "MODERATE"
    else:
        corr_quality = "POOR"
    
    print(f"\n1. CORRELATION QUALITY: {corr_quality}")
    print(f"   The correlation of {stats_dict['Correlation']:.4f} indicates {corr_quality.lower()} agreement")
    print(f"   between RMV and actual sale prices.")
    
    bias_threshold = 0.05  # 5% significance level
    if stats_dict['Bias_P_Value'] < bias_threshold:
        if stats_dict['Median_Percent_Diff'] > 0:
            bias_direction = "OVERPRICED"
            print(f"\n2. SYSTEMATIC BIAS DETECTED: Properties sell for MORE than RMV")
        else:
            bias_direction = "UNDERPRICED"
            print(f"\n2. SYSTEMATIC BIAS DETECTED: Properties sell for LESS than RMV")
        print(f"   Median difference: {stats_dict['Median_Percent_Diff']:.2f}%")
        print(f"   (Statistically significant, p={stats_dict['Bias_P_Value']:.6f})")
    else:
        print(f"\n2. NO SYSTEMATIC BIAS: RMV appears unbiased overall")
        print(f"   (p={stats_dict['Bias_P_Value']:.4f} > 0.05)")
    
    if abs(stats_dict['Regression_Slope'] - 1.0) > 0.1:
        print(f"\n3. SCALING ISSUE: Regression slope of {stats_dict['Regression_Slope']:.4f}")
        if stats_dict['Regression_Slope'] > 1.0:
            print(f"   Higher-valued homes sell for proportionally MORE than RMV suggests")
        else:
            print(f"   Higher-valued homes sell for proportionally LESS than RMV suggests")
    else:
        print(f"\n3. PROPORTIONAL ACCURACY: Slope near 1.0 indicates consistent scaling")
    
    print("\n" + "="*70)

def compare_cities(gresham_stats, salem_stats):
    """
    Compare the two cities' RMV accuracy
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: GRESHAM VS. SALEM")
    print("="*70)

    metrics = [
        ('Sample Size', 'Count', '{:,}', '{:,}'),
        ('Correlation (r)', 'Correlation', '{:.4f}', '{:.4f}'),
        ('R-Squared', 'R_Squared', '{:.4f}', '{:.4f}'),
        ('Median % Diff', 'Median_Percent_Diff', '{:.2f}%', '{:.2f}%'),
        ('Mean Ratio', 'Mean_Ratio', '{:.3f}', '{:.3f}'),
        ('Regression Slope', 'Regression_Slope', '{:.4f}', '{:.4f}'),
        ('Median Sale Price', 'Median_Sale_Price', '${:,.0f}', '${:,.0f}'),
        ('Median RMV', 'Median_RMV', '${:,.0f}', '${:,.0f}'),
    ]

    print(f"\n  {'Metric':<25} {'Gresham':>15} {'Salem':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    for label, key, gfmt, sfmt in metrics:
        gval = gfmt.format(gresham_stats[key])
        sval = sfmt.format(salem_stats[key])
        print(f"  {label:<25} {gval:>15} {sval:>15}")

    print("\n" + "="*70)


def create_comparison_visualization(salem_df, gresham_df, output_dir):
    """
    Create side-by-side comparison chart for Salem vs Gresham.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Salem vs. Gresham: RMV Assessment Accuracy Comparison',
                 fontsize=16, fontweight='bold')

    # 1. Overlaid scatter plots
    ax1 = axes[0, 0]
    ax1.scatter(salem_df['RMV'], salem_df['SalePrice'], alpha=0.3, s=15,
                color='tab:blue', label='Salem')
    ax1.scatter(gresham_df['RMV'], gresham_df['SalePrice'], alpha=0.4, s=15,
                color='tab:orange', label='Gresham')
    lims = [
        min(salem_df['RMV'].min(), gresham_df['RMV'].min()),
        max(salem_df['RMV'].max(), gresham_df['RMV'].max()),
    ]
    ax1.plot(lims, lims, 'g--', linewidth=2, label='Perfect (y=x)')
    ax1.set_xlabel('County RMV ($)')
    ax1.set_ylabel('Sale Price ($)')
    ax1.set_title('RMV vs. Sale Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Overlaid percent difference distributions
    ax2 = axes[0, 1]
    bins = np.linspace(-50, 100, 60)
    ax2.hist(salem_df['Percent_Difference'].clip(-50, 100), bins=bins, alpha=0.5,
             color='tab:blue', label='Salem', density=True)
    ax2.hist(gresham_df['Percent_Difference'].clip(-50, 100), bins=bins, alpha=0.5,
             color='tab:orange', label='Gresham', density=True)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('% Difference (Sale - RMV) / RMV')
    ax2.set_ylabel('Density')
    ax2.set_title('RMV Prediction Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Side-by-side box plots
    ax3 = axes[1, 0]
    ax3.boxplot([salem_df['Ratio'], gresham_df['Ratio']],
                tick_labels=['Salem', 'Gresham'])
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect Match')
    ax3.set_ylabel('Sale Price / RMV Ratio')
    ax3.set_title('Ratio Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Summary statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    s_stats, _ = calculate_statistics(salem_df.copy())
    g_stats, _ = calculate_statistics(gresham_df.copy())
    rows = [
        ['Metric', 'Salem', 'Gresham'],
        ['N sales', f"{s_stats['Count']:,}", f"{g_stats['Count']:,}"],
        ['Correlation (r)', f"{s_stats['Correlation']:.3f}", f"{g_stats['Correlation']:.3f}"],
        ['R-squared', f"{s_stats['R_Squared']:.3f}", f"{g_stats['R_Squared']:.3f}"],
        ['Median % Diff', f"{s_stats['Median_Percent_Diff']:.1f}%", f"{g_stats['Median_Percent_Diff']:.1f}%"],
        ['Mean Ratio', f"{s_stats['Mean_Ratio']:.3f}", f"{g_stats['Mean_Ratio']:.3f}"],
        ['Reg. Slope', f"{s_stats['Regression_Slope']:.3f}", f"{g_stats['Regression_Slope']:.3f}"],
    ]
    table = ax4.table(cellText=rows[1:], colLabels=rows[0], loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax4.set_title('Summary Comparison', fontsize=14)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'gresham_vs_salem_comparison.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved: {fig_path}")
    return fig

def apply_property_filters(df, min_price=None, max_price=None,
                           min_acres=None, min_baths=None, label=''):
    """
    Apply property filters to match comparable criteria across cities.
    Works with both Salem (sales CSV columns) and Gresham (research DB columns).
    """
    before = len(df)
    filters_desc = []

    if min_price is not None or max_price is not None:
        lo = min_price or 0
        hi = max_price or float('inf')
        df = df[(df['SalePrice'] >= lo) & (df['SalePrice'] <= hi)]
        filters_desc.append(f"price ${lo:,.0f}-${hi:,.0f}")

    if min_acres is not None:
        # Salem: 'Total Fragment (Land) Acres', Gresham: not available (skip)
        if 'Total Fragment (Land) Acres' in df.columns:
            acres = pd.to_numeric(df['Total Fragment (Land) Acres'], errors='coerce')
            df = df[acres >= min_acres]
            filters_desc.append(f"lot >= {min_acres} acres")

    if min_baths is not None:
        # Salem has separate Bathrooms + Half Bathrooms columns
        if 'Bathrooms' in df.columns:
            full = pd.to_numeric(df['Bathrooms'], errors='coerce').fillna(0)
            half = pd.to_numeric(df.get('Half Bathrooms', 0), errors='coerce').fillna(0)
            total_baths = full + half * 0.5
            df = df[total_baths >= min_baths]
            filters_desc.append(f"baths >= {min_baths}")

    if filters_desc:
        desc = ', '.join(filters_desc)
        print(f"  Property filter ({desc}): {before} -> {len(df)}")

    return df


def run_salem_analysis(base_dir, figures_dir, reports_dir, property_filters=None):
    """Load Salem data, run analysis, return (stats, analyzed_df)."""
    orcats_file = os.path.join(base_dir, 'data', 'raw', 'comprehensive', 'ORCATS999_(NEW).csv')
    sales_2024 = os.path.join(base_dir, 'data', 'raw', '2024SalesData.csv')
    sales_2025 = os.path.join(base_dir, 'data', 'raw', '2025SalesData.csv')

    print("=" * 70)
    print("RMV vs. SALE PRICE ANALYSIS - SALEM, OREGON")
    print("Data: Marion County sales (2024-2025) + ORCATS999 assessments")
    print("Filter: Confirmed arms-length sales only (Condition Code 33)")
    print("=" * 70)

    print("\n--- Loading assessment data ---")
    orcats_df = load_orcats_data(orcats_file)

    print("\n--- Loading sales data ---")
    sales_df = load_sales_data(sales_2024, sales_2025)

    salem_df = prepare_salem_data(orcats_df, sales_df)

    if property_filters:
        salem_df = apply_property_filters(salem_df, **property_filters)

    if len(salem_df) < 30:
        print(f"\nWARNING: Only {len(salem_df)} records. Minimum 30 recommended.")
        if len(salem_df) == 0:
            return None, None

    salem_stats, salem_analyzed = calculate_statistics(salem_df)
    print_analysis_report('SALEM', salem_stats)

    fig = create_visualizations(salem_analyzed, 'Salem')
    fig_path = os.path.join(figures_dir, 'salem_rmv_vs_sale_price.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {fig_path}")

    output_cols = ['Account Number', 'Situs Address', 'SaleDate', 'SalePrice',
                   'RMVLAND', 'RMVIMPR', 'RMV', 'AV',
                   'Difference', 'Percent_Difference', 'Ratio']
    output_cols = [c for c in output_cols if c in salem_analyzed.columns]
    results_path = os.path.join(reports_dir, 'salem_analysis_results.csv')
    salem_analyzed[output_cols].to_csv(results_path, index=False)
    print(f"Detailed results saved: {results_path}")

    return salem_stats, salem_analyzed


def run_gresham_analysis(base_dir, figures_dir, reports_dir,
                         exclude_new_construction=False):
    """Load Gresham data, run analysis, return (stats, analyzed_df)."""
    gresham_db = os.path.join(base_dir, 'data', 'gresham', 'gresham_research_db.csv')

    if not os.path.exists(gresham_db):
        print("\nGresham research database not found.")
        print("Run 'python gresham_data_collector.py ingest <redfin.csv>' first.")
        return None, None

    print("\n" + "=" * 70)
    print("RMV vs. SALE PRICE ANALYSIS - GRESHAM, OREGON")
    print("Data: Redfin sales + Multnomah County ArcGIS assessments")
    if exclude_new_construction:
        print("Filter: Excluding new construction (built 2024+)")
    print("=" * 70)

    gresham_df = load_gresham_data(gresham_db,
                                   exclude_new_construction=exclude_new_construction)

    if len(gresham_df) < 30:
        print(f"\nWARNING: Only {len(gresham_df)} records. Minimum 30 recommended.")
        if len(gresham_df) == 0:
            return None, None

    gresham_stats, gresham_analyzed = calculate_statistics(gresham_df)
    print_analysis_report('GRESHAM', gresham_stats)

    fig = create_visualizations(gresham_analyzed, 'Gresham')
    fig_path = os.path.join(figures_dir, 'gresham_rmv_vs_sale_price.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {fig_path}")

    output_cols = ['PropertyID', 'Address', 'SaleDate', 'SalePrice',
                   'RMVLAND', 'RMVIMPR', 'RMV',
                   'Difference', 'Percent_Difference', 'Ratio']
    output_cols = [c for c in output_cols if c in gresham_analyzed.columns]
    results_path = os.path.join(reports_dir, 'gresham_analysis_results.csv')
    gresham_analyzed[output_cols].to_csv(results_path, index=False)
    print(f"Detailed results saved: {results_path}")

    return gresham_stats, gresham_analyzed


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Analyze how well county-assessed Real Market Values (RMV) '
                    'predict actual home sale prices in Salem and Gresham, Oregon.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s                                  Salem analysis (default)
  %(prog)s --city gresham                   Gresham analysis
  %(prog)s --city both                      Compare both cities
  %(prog)s --city both --exclude-new        Exclude new construction from Gresham
  %(prog)s --city both --match-gresham      Match Salem filters to Gresham criteria
  %(prog)s --min-price 400000 --max-price 800000   Custom price range

data setup:
  Salem:   Automatic (uses data/raw/ CSVs + ORCATS999 assessment data)
  Gresham: Run 'python gresham_data_collector.py ingest <redfin.csv>' first
""")

    parser.add_argument('--city', choices=['salem', 'gresham', 'both'],
                        default='salem',
                        help='which city to analyze (default: salem)')
    parser.add_argument('--exclude-new', action='store_true',
                        help='exclude new construction (built 2024+) from Gresham; '
                             'these have land-only RMV before the home was completed')
    parser.add_argument('--match-gresham', action='store_true',
                        help='filter Salem to match Gresham Redfin search criteria '
                             '($500K-$2M, 0.25 acre+, 2.5 bath+)')
    parser.add_argument('--min-price', type=float, default=None, metavar='$',
                        help='minimum sale price filter')
    parser.add_argument('--max-price', type=float, default=None, metavar='$',
                        help='maximum sale price filter')
    parser.add_argument('--min-acres', type=float, default=None, metavar='N',
                        help='minimum lot size in acres (Salem only)')
    parser.add_argument('--min-baths', type=float, default=None, metavar='N',
                        help='minimum total bathrooms, e.g. 2.5 (Salem only)')
    args = parser.parse_args()

    # Build property filters
    property_filters = {}
    if args.match_gresham:
        property_filters = {
            'min_price': 500000, 'max_price': 2000000,
            'min_acres': 0.25, 'min_baths': 2.5,
        }
        print("** Applying Gresham-matching filters: $500K-$2M, 0.25ac+, 2.5ba+ **\n")
    else:
        if args.min_price: property_filters['min_price'] = args.min_price
        if args.max_price: property_filters['max_price'] = args.max_price
        if args.min_acres: property_filters['min_acres'] = args.min_acres
        if args.min_baths: property_filters['min_baths'] = args.min_baths

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    reports_dir = os.path.join(results_dir, 'reports')
    for d in [results_dir, figures_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    salem_stats = salem_analyzed = None
    gresham_stats = gresham_analyzed = None

    if args.city in ('salem', 'both'):
        salem_stats, salem_analyzed = run_salem_analysis(
            base_dir, figures_dir, reports_dir,
            property_filters=property_filters or None
        )

    if args.city in ('gresham', 'both'):
        gresham_stats, gresham_analyzed = run_gresham_analysis(
            base_dir, figures_dir, reports_dir,
            exclude_new_construction=args.exclude_new
        )

    if args.city == 'both' and salem_stats and gresham_stats:
        compare_cities(gresham_stats, salem_stats)
        create_comparison_visualization(salem_analyzed, gresham_analyzed, figures_dir)

    if os.environ.get('MPLBACKEND') != 'Agg':
        plt.show()

if __name__ == "__main__":
    main()
