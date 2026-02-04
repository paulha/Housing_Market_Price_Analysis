"""
Analysis of Real Market Value (RMV) vs. Actual Sale Price
for Gresham and Salem, Oregon

This script performs statistical analysis to determine if county-assessed
Real Market Values correlate properly with actual sale prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

def load_marion_county_data(filepath):
    """
    Load Marion County (Salem) sales data from CSV
    
    Expected columns will vary - inspect the CSV to determine exact column names
    Common columns: Account, SaleDate, SalePrice, RMV, etc.
    """
    df = pd.read_csv(filepath)
    
    # Print column names to help identify the right fields
    print("Marion County Columns:", df.columns.tolist())
    
    return df

def filter_recent_sales(df, months=12, date_column='SaleDate'):
    """
    Filter to sales within the last N months
    """
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=months*30)
    
    # Filter
    recent = df[df[date_column] >= cutoff_date].copy()
    
    print(f"Found {len(recent)} sales in the last {months} months")
    
    return recent

def clean_sales_data(df, sale_price_col='SalePrice', rmv_col='RMV', city_filter=None):
    """
    Clean the sales data:
    - Remove non-arms-length transactions
    - Remove outliers
    - Filter to specific city if needed
    """
    # Remove zeros and nulls
    df = df[(df[sale_price_col] > 0) & (df[rmv_col] > 0)].copy()
    
    # Remove extreme outliers (likely data errors)
    # Keep sales between $50k and $5M
    df = df[(df[sale_price_col] >= 50000) & (df[sale_price_col] <= 5000000)]
    df = df[(df[rmv_col] >= 50000) & (df[rmv_col] <= 5000000)]
    
    # Filter to specific city if provided
    if city_filter:
        if 'City' in df.columns:
            df = df[df['City'].str.upper() == city_filter.upper()]
        elif 'MailingCity' in df.columns:
            df = df[df['MailingCity'].str.upper() == city_filter.upper()]
    
    print(f"After cleaning: {len(df)} valid sales remain")
    
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
    ax3.boxplot([df['Ratio']], labels=[city_name])
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
    
    print("\nCORRELATION COMPARISON:")
    print(f"  Gresham R²: {gresham_stats['R_Squared']:.4f}")
    print(f"  Salem R²:   {salem_stats['R_Squared']:.4f}")
    
    if gresham_stats['R_Squared'] > salem_stats['R_Squared']:
        print(f"  → Gresham has BETTER correlation (difference: {gresham_stats['R_Squared'] - salem_stats['R_Squared']:.4f})")
    else:
        print(f"  → Salem has BETTER correlation (difference: {salem_stats['R_Squared'] - gresham_stats['R_Squared']:.4f})")
    
    print("\nPREDICTION ACCURACY COMPARISON:")
    print(f"  Gresham Median % Diff: {gresham_stats['Median_Percent_Diff']:.2f}%")
    print(f"  Salem Median % Diff:   {salem_stats['Median_Percent_Diff']:.2f}%")
    
    print("\n" + "="*70)

def main():
    """
    Main analysis function
    """
    print("="*70)
    print("RMV vs. SALE PRICE ANALYSIS - GRESHAM AND SALEM, OREGON")
    print("="*70)
    
    # Instructions for data access
    print("\nDATA SOURCES:")
    print("1. Salem (Marion County): Download from")
    print("   https://apps.co.marion.or.us/AO/PropertySalesData/2025SalesData.csv")
    print("   https://apps.co.marion.or.us/AO/PropertySalesData/2024SalesData.csv")
    print("\n2. Gresham (Multnomah County): Individual property searches at")
    print("   https://multcoproptax.com")
    print("   (No bulk download available - would need web scraping)")
    
    print("\n" + "="*70)
    print("\nTo run this analysis:")
    print("1. Download the Marion County data files")
    print("2. Update the file paths below")
    print("3. Inspect the CSV columns and update column names as needed")
    print("4. Run the script")
    print("\n" + "="*70)
    
    # Example usage (uncomment and modify when data is available):
    """
    # Load Salem data
    salem_df = load_marion_county_data('marion_2024_sales.csv')
    salem_recent = filter_recent_sales(salem_df, months=12)
    salem_clean = clean_sales_data(salem_recent, city_filter='SALEM')
    
    salem_stats, salem_df_analyzed = calculate_statistics(salem_clean)
    print_analysis_report('SALEM', salem_stats)
    
    fig1 = create_visualizations(salem_df_analyzed, 'Salem')
    plt.savefig('salem_analysis.png', dpi=300, bbox_inches='tight')
    
    # Save detailed results
    salem_df_analyzed.to_csv('salem_analysis_results.csv', index=False)
    
    # Similar process for Gresham once data is obtained
    # ...
    
    plt.show()
    """

if __name__ == "__main__":
    main()
