# Housing Market Price Analysis: Gresham and Salem, Oregon

Statistical analysis comparing county-assessed Real Market Values (RMV) to actual home sale prices in Gresham and Salem, Oregon.

## Project Overview

This project examines whether county tax assessors' RMV estimates accurately reflect actual market sale prices. The analysis has important implications for:
- **Tax equity**: Are properties being assessed fairly?
- **Market transparency**: Can buyers/sellers trust RMV as a price indicator?
- **Assessment accuracy**: How well do counties track rapidly changing markets?

## Research Questions

1. How strong is the correlation between RMV and actual sale prices?
2. Is there systematic bias (over/under-valuation)?
3. How do the two cities compare in assessment accuracy?
4. Are there patterns in prediction errors across price ranges?

## Project Structure

```
.
├── README.md                           # This file
├── analysis_methodology_summary.md     # Detailed methodology and background
├── data_collection_guide.md            # Step-by-step guide to obtaining data
├── rmv_sale_price_analysis.py         # Main analysis script
├── data/                               # Data directory (gitignored)
│   ├── raw/                           # Original downloaded data
│   └── processed/                     # Cleaned data ready for analysis
├── results/                            # Analysis outputs (gitignored)
│   ├── figures/                       # Generated visualizations
│   └── reports/                       # Statistical summaries
└── docs/                               # Additional documentation
```

## Quick Start

### 1. Get the Data

**For Salem (Easy - Available Now):**
```bash
# Download Marion County sales data
curl -L "https://apps.co.marion.or.us/AO/PropertySalesData/2025SalesData.csv" -o data/raw/marion_2025_sales.csv
curl -L "https://apps.co.marion.or.us/AO/PropertySalesData/2024SalesData.csv" -o data/raw/marion_2024_sales.csv
```

**For Gresham (Requires Manual Collection):**
- See `data_collection_guide.md` for detailed instructions
- Options: Manual collection, public records request, or third-party data

### 2. Set Up Environment

```bash
# Install required Python packages
pip install pandas numpy matplotlib seaborn scipy
```

### 3. Run the Analysis

```bash
# Edit the script to point to your data files
python rmv_sale_price_analysis.py
```

## Data Sources

### Salem (Marion County)
- **Source**: Marion County Assessor's Office
- **URL**: https://www.co.marion.or.us/AO/Pages/datacenter.aspx
- **Format**: CSV (bulk download available)
- **Update Frequency**: Weekly

### Gresham (Multnomah County)
- **Source**: Multnomah County Assessment & Taxation
- **URL**: https://multcoproptax.com
- **Format**: Individual property searches only
- **Access**: Requires manual collection or public records request

## Key Findings (To Be Updated After Analysis)

*This section will be updated once the analysis is complete.*

### Salem Results
- Sample size: TBD
- Correlation (r): TBD
- Median % difference: TBD
- Systematic bias: TBD

### Gresham Results
- Sample size: TBD
- Correlation (r): TBD
- Median % difference: TBD
- Systematic bias: TBD

## Methodology

The analysis employs several statistical techniques:

1. **Correlation Analysis**: Pearson correlation coefficient and R²
2. **Bias Detection**: One-sample t-test on (Sale Price - RMV) / RMV
3. **Regression Analysis**: Linear regression of Sale Price on RMV
4. **Distribution Analysis**: Residual plots and error distribution
5. **Comparative Analysis**: Side-by-side comparison of both cities

See `analysis_methodology_summary.md` for detailed methodology.

## Understanding Oregon's Property Tax System

Oregon uses three different property values:

- **RMV (Real Market Value)**: County's estimate of market value - *this is what we're testing*
- **MAV (Maximum Assessed Value)**: Limited to 3% annual growth (Measure 50)
- **AV (Assessed Value)**: The lower of RMV or MAV - *this is what taxes are based on*

In appreciating markets, RMV > AV, so homes may sell for more than their taxed value.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Data Privacy

This analysis uses publicly available property records. No private or confidential information is included. All sales data is from official county records.

## Contributing

This is a research project. If you'd like to contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Suggested contributions:
- Additional cities/counties
- Alternative visualization approaches
- Time-series analysis
- Geographic/neighborhood-level analysis

## License

[To be determined]

## Contact

For questions or collaboration:
- GitHub: [@paulha](https://github.com/paulha)

## Acknowledgments

- Marion County Assessor's Office for publicly accessible data
- Multnomah County Assessment & Taxation
- Oregon Department of Revenue for property tax documentation

## References

- Oregon Department of Revenue Property Tax Statistics
- Oregon Revised Statutes (ORS) on Property Assessment
- Measure 5 and Measure 50 documentation

---

**Last Updated**: February 3, 2026
**Status**: Data collection phase
