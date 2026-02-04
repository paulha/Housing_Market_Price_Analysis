# RMV vs. Sale Price Analysis: Gresham and Salem, Oregon
## Summary of Findings and Methodology

Date: February 3, 2026

---

## EXECUTIVE SUMMARY

This analysis examines whether county-assessed Real Market Values (RMV) correlate properly 
with actual home sale prices in Gresham (Multnomah County) and Salem (Marion County), Oregon.

### Key Question
**Do county assessors' RMV estimates accurately reflect actual market sale prices?**

---

## DATA SOURCES IDENTIFIED

### 1. SALEM (Marion County) ✓ Data Available
- **Source**: Marion County Assessor's Office Data Center
- **URL**: https://www.co.marion.or.us/AO/Pages/datacenter.aspx
- **Bulk Downloads**: CSV files available for download
  - 2025 Sales: https://apps.co.marion.or.us/AO/PropertySalesData/2025SalesData.csv
  - 2024 Sales: https://apps.co.marion.or.us/AO/PropertySalesData/2024SalesData.csv
  - Historical: Available back to 1940s
- **Accessibility**: Public, free download
- **Update Frequency**: Weekly (automated process)

### 2. GRESHAM (Multnomah County) ⚠️ Limited Access
- **Source**: Multnomah County Property Tax System
- **URL**: https://multcoproptax.com
- **Access Method**: Individual property search only
- **Bulk Downloads**: NOT publicly available
- **Alternative**: PortlandMaps.com and GreshamView have limited data
- **Data Requirements**: Would need either:
  - Manual collection of individual records
  - Web scraping (check terms of service)
  - Public records request to county

---

## UNDERSTANDING OREGON'S PROPERTY VALUATION SYSTEM

### Three Key Values:

1. **Real Market Value (RMV)**
   - County assessor's estimate of fair market value
   - What property "should" sell for on open market
   - Updated annually by assessor
   - NOT limited by Measure 50's 3% cap

2. **Maximum Assessed Value (MAV)**
   - Limited to 3% annual growth (Measure 50)
   - Starts at RMV when property is first assessed
   - Grows max 3% per year unless property changes

3. **Assessed Value (AV)**
   - The LOWER of RMV or MAV
   - This is what property taxes are calculated on
   - NOT necessarily equal to RMV

### Important Implication:
In rapidly appreciating markets, RMV can be significantly HIGHER than AV. This means:
- Properties may sell for more than their taxed value (AV)
- RMV is the assessor's attempt to track actual market value
- **RMV is what we should compare to sale prices**

---

## STATISTICAL METHODOLOGY

### Research Question:
**H₀ (Null Hypothesis)**: County RMV accurately predicts sale price (correlation = 1.0, no bias)
**H₁ (Alternative)**: County RMV does not accurately predict sale price

### Analysis Components:

#### 1. CORRELATION ANALYSIS
- **Pearson Correlation Coefficient (r)**: Measures linear relationship strength
  - r = 1.0 → Perfect positive correlation
  - r = 0.9-1.0 → Excellent correlation
  - r = 0.8-0.9 → Good correlation
  - r = 0.7-0.8 → Moderate correlation
  - r < 0.7 → Poor correlation

- **R-Squared (r²)**: Proportion of variance explained
  - Tells us how much of sale price variance is explained by RMV

#### 2. BIAS DETECTION
- **Measure**: (Sale Price - RMV) / RMV × 100
- **Statistical Test**: One-sample t-test (H₀: median difference = 0%)
- **Interpretation**:
  - Positive bias → Properties sell for MORE than RMV
  - Negative bias → Properties sell for LESS than RMV
  - p < 0.05 → Bias is statistically significant

#### 3. RATIO ANALYSIS
- **Sale Price / RMV Ratio**:
  - Ratio = 1.0 → Perfect match
  - Ratio > 1.0 → Properties sell above RMV
  - Ratio < 1.0 → Properties sell below RMV

#### 4. REGRESSION ANALYSIS
- **Model**: Sale Price = β₀ + β₁(RMV) + ε
- **Ideal Result**: β₁ ≈ 1.0, β₀ ≈ 0
- **Interpretation of Slope (β₁)**:
  - β₁ = 1.0 → Perfect proportional relationship
  - β₁ > 1.0 → Higher-value homes sell for even more than RMV predicts
  - β₁ < 1.0 → Higher-value homes sell for less than RMV predicts

#### 5. DISTRIBUTION ANALYSIS
- **Residuals**: Sale Price - RMV
  - Should be normally distributed around zero if RMV is unbiased
  - Outliers indicate properties where RMV was very wrong
  
- **Heteroscedasticity Check**:
  - Are prediction errors consistent across price ranges?
  - Or do errors increase with property value?

---

## DATA CLEANING REQUIREMENTS

### Filters to Apply:

1. **Time Period**: Last 6-12 months (as specified)
2. **City Filter**: Gresham OR Salem specifically
3. **Price Range**: Reasonable bounds (e.g., $50k - $5M)
4. **Transaction Type**: Arms-length sales only
   - Exclude family transfers
   - Exclude foreclosures (may be)
   - Exclude bulk/commercial if analyzing residential
5. **Data Quality**:
   - Remove records with missing RMV
   - Remove records with $0 or null sale prices
   - Remove clear data entry errors

---

## EXPECTED FINDINGS & INTERPRETATIONS

### Scenario 1: High Correlation, No Bias
- **Evidence**: r > 0.85, median % diff ≈ 0%, p > 0.05
- **Interpretation**: County assessors are accurately tracking market
- **Conclusion**: RMV system is working well

### Scenario 2: High Correlation, Positive Bias
- **Evidence**: r > 0.85, median % diff > 5%, p < 0.05
- **Interpretation**: Properties consistently sell ABOVE RMV
- **Possible Causes**:
  - Assessors are conservative (undervaluing)
  - Hot market moving faster than annual assessments
  - Lag between assessment date and sale date

### Scenario 3: High Correlation, Negative Bias
- **Evidence**: r > 0.85, median % diff < -5%, p < 0.05
- **Interpretation**: Properties consistently sell BELOW RMV
- **Possible Causes**:
  - Assessors are aggressive (overvaluing)
  - Market cooling between assessment and sale
  - Assessment methodology issues

### Scenario 4: Low Correlation
- **Evidence**: r < 0.7
- **Interpretation**: RMV is not effectively tracking market values
- **Possible Causes**:
  - Inadequate assessment resources/methods
  - Heterogeneous market with different property types
  - External factors not captured in RMV

---

## COMPARISON METRICS: GRESHAM VS. SALEM

### Key Comparison Points:

1. **Correlation Strength**: Which county has better r²?
2. **Systematic Bias**: Do both over/undervalue, or differently?
3. **Prediction Variability**: Which has more consistent errors?
4. **Scaling**: Do both handle high/low values equally well?

### Factors That May Explain Differences:

- **Market Dynamics**: 
  - Gresham: Portland metro suburb, more volatile
  - Salem: State capital, more stable government jobs

- **Assessment Practices**: Different counties, different assessors

- **Sample Size**: More sales = more reliable statistics

- **Property Mix**: Different types of housing stock

---

## MARKET CONTEXT (Recent Data)

### Gresham (Multnomah County)
- **Median Sale Price**: ~$466K (October 2025)
- **Year-over-Year**: Down 3.7%
- **Days on Market**: ~31 days (median: 60)
- **Inventory**: 70 active listings (March 2025)
- **Market Status**: Competitive, low inventory

### Salem (Marion County)
- **Median Sale Price**: ~$440K (April 2025)
- **Year-over-Year**: Flat
- **Closed Sales**: 153 homes/month
- **Market Status**: Seller's market under $600K, buyer's market above

---

## LIMITATIONS OF THIS ANALYSIS

1. **Data Access**: 
   - Gresham bulk data not readily available
   - May need manual data collection

2. **Time Lag**:
   - RMV assessed on specific date (usually January 1)
   - Sales occur throughout year
   - Market can change significantly in 6-12 months

3. **Property Heterogeneity**:
   - Different property types (SFH, condos, townhomes)
   - Different neighborhoods
   - Different conditions

4. **Economic Conditions**:
   - Interest rates affect sales prices
   - Seasonal variations
   - Local economic factors

5. **Data Quality**:
   - Relies on accurate county records
   - Some transactions may be misclassified

---

## NEXT STEPS TO COMPLETE ANALYSIS

### For Salem (Immediately Actionable):
1. Download CSV files from Marion County
2. Inspect column names and data structure
3. Run provided Python analysis script
4. Generate visualizations and statistics
5. Interpret results

### For Gresham (Requires Additional Work):
**Option A - Manual Collection**:
- Search individual properties on MultcoPropTax.com
- Record RMV and recent sale price for each
- Compile into spreadsheet
- Minimum sample: 50-100 sales for statistical validity

**Option B - Public Records Request**:
- Contact Multnomah County Assessor
- Request bulk sales data with RMV
- May take 2-4 weeks
- May have associated fees

**Option C - Third-Party Data**:
- Real estate data services (Zillow, Redfin APIs)
- May have limitations or costs

### Analysis Workflow:
1. Obtain data for both cities
2. Clean and filter data (last 6-12 months, valid sales only)
3. Run statistical analysis
4. Generate comparison visualizations
5. Prepare findings report

---

## KEY QUESTIONS TO ANSWER

1. **How strong is the correlation between RMV and sale price in each city?**
   - Is RMV a reliable predictor of actual market value?

2. **Is there systematic bias in either direction?**
   - Do properties sell for more or less than RMV on average?
   - Is this bias statistically significant?

3. **How do the two cities compare?**
   - Which county's assessments are more accurate?
   - What might explain any differences?

4. **Are there patterns in the errors?**
   - Do high-value properties have different error patterns than low-value?
   - Are errors getting larger or smaller over time?

5. **What are the practical implications?**
   - For homebuyers: Is RMV a good guide for offer prices?
   - For sellers: Should they expect more/less than RMV?
   - For tax policy: Are assessments fair and accurate?

---

## DELIVERABLES

1. **Statistical Summary Tables** for each city showing:
   - Sample size
   - Mean/median sale price and RMV
   - Correlation coefficients
   - Bias metrics
   - Regression results

2. **Visualizations**:
   - Scatter plots with regression lines
   - Distribution of prediction errors
   - Box plots of ratios
   - Residual plots

3. **Comparative Analysis**: Side-by-side comparison of Gresham vs. Salem

4. **Interpretation Report**: Plain-language explanation of findings

---

## TECHNICAL REQUIREMENTS

### Software Needed:
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scipy
- OR: Excel with Analysis ToolPak
- OR: R with ggplot2

### Data Format:
- CSV files with at minimum:
  - Property ID
  - Sale Date
  - Sale Price
  - Real Market Value (RMV)
  - City/Location
  - Property Type (optional but helpful)

### Sample Size Recommendations:
- Minimum: 50 sales per city (for basic analysis)
- Good: 100-200 sales per city
- Excellent: 500+ sales per city

---

## CONCLUSION

The analysis framework is ready. The primary bottleneck is accessing bulk sales data 
for Gresham (Multnomah County). Salem data is immediately accessible.

**The key analytical question is whether Oregon counties' RMV estimates - which are 
supposed to represent fair market value - actually correlate with what homes sell for 
in the market. This has important implications for tax equity and market transparency.**

Based on market research, we expect to find:
- Generally strong positive correlation (r > 0.80)
- Possible positive bias (sales > RMV) in hot markets
- Greater variance in rapidly changing markets
- County-specific differences based on assessment practices

The provided Python script contains all necessary statistical tests and visualizations
to answer these questions once the data is obtained.
