# Data Collection Guide: Gresham and Salem Property Records

## SALEM (Marion County) - Easy ✓

### Method: Direct Download from County Website

**Step 1: Download 2025 Sales Data**
```
URL: https://apps.co.marion.or.us/AO/PropertySalesData/2025SalesData.csv
Method: Direct download - click link or use curl/wget
```

**Step 2: Download 2024 Sales Data** (for larger sample)
```
URL: https://apps.co.marion.or.us/AO/PropertySalesData/2024SalesData.csv
Method: Direct download
```

**Step 3: Review Data Structure**
The CSV should contain columns like:
- Account/Property ID
- SaleDate
- SalePrice
- RMV (Real Market Value)
- City/Location
- Property type/class

**Expected Column Names** (may vary):
- Check the actual column headers when you open the file
- Refer to the key documents provided on the website:
  - https://www.co.marion.or.us/AO/Pages/datacenter.aspx
  - Look for "Excel help" and "Keys" for column definitions

**Step 4: Filter for Salem**
- Filter to City = "SALEM" (or similar field)
- Filter to residential properties if analyzing homes

---

## GRESHAM (Multnomah County) - Challenging ⚠️

### Option 1: Manual Collection (Immediate but Labor-Intensive)

**Time Required**: ~2-4 hours for 50-100 properties
**Cost**: Free
**Accuracy**: High (official county data)

#### Process:
1. Go to https://multcoproptax.com
2. Search for properties in Gresham by address or property ID
3. For each property, record:
   - Property ID (Account Number)
   - Address
   - Real Market Value (RMV) - current year
   - Recent sale price (from "Sales History" section)
   - Sale date

4. Create a spreadsheet with columns:
   ```
   Property_ID | Address | Sale_Date | Sale_Price | RMV | City
   ```

5. Target sample:
   - Minimum: 50 recent sales
   - Better: 100+ sales
   - Focus on sales within last 6-12 months

**How to Find Recent Sales**:
- Use Redfin/Zillow to identify recently sold homes in Gresham
- Look up each address in MultcoPropTax.com
- Record both the sale price AND the county's RMV

---

### Option 2: Public Records Request (Best for Bulk Data)

**Time Required**: 2-4 weeks processing
**Cost**: May have associated fees
**Accuracy**: Official county data

#### Process:

1. **Contact Information**:
   ```
   Multnomah County Assessment, Recording & Taxation
   501 SE Hawthorne Blvd
   Portland, OR 97214
   Phone: (503) 988-3326
   Email: assessment@multco.us
   Website: https://multco.us/assessment-taxation
   ```

2. **Public Records Request Template**:
   ```
   Subject: Public Records Request - Recent Property Sales Data

   To Whom It May Concern:

   Pursuant to Oregon Public Records Law (ORS 192.410 et seq.), I am requesting 
   access to the following public records:

   Property sales data for the City of Gresham for the period [Date] to [Date], 
   including the following fields for each transaction:
   
   - Property Account Number
   - Site Address
   - Sale Date
   - Sale Price
   - Real Market Value (RMV) at time of sale
   - Property Type/Class
   - Square Footage (if available)
   
   Please provide this data in electronic format (CSV or Excel preferred).

   If there are any fees associated with this request, please notify me before 
   proceeding if fees will exceed $[your limit].

   Thank you for your assistance.

   Sincerely,
   [Your Name]
   [Your Contact Information]
   ```

3. **Expected Response Time**: 
   - Initial response: Within 5-10 business days
   - Data delivery: 2-4 weeks (varies by complexity)

4. **Potential Fees**:
   - May charge for staff time
   - May charge per record
   - Electronic records often cheaper than printed

---

### Option 3: Third-Party Data Sources (Fastest but May Have Gaps)

**Time Required**: Immediate
**Cost**: Varies ($0-$$$)
**Accuracy**: Generally good but may differ from official county records

#### Free Options:

**A. Redfin Data**
- Website: https://www.redfin.com/city/7995/OR/Gresham/recently-sold
- Provides: Recent sales, prices, dates
- Limitation: Doesn't include official county RMV
- Process:
  1. Find recently sold homes on Redfin
  2. Look up each address on MultcoPropTax.com for RMV
  3. Manual data entry required

**B. Zillow**
- Similar to Redfin
- Provides "Zestimate" but not official RMV
- Must cross-reference with county records

**C. PropertyShark/PropertyChecker**
- Free limited searches
- May have some bulk data options
- Check: https://oregon.propertychecker.com/multnomah-county

#### Paid Options:

**A. ATTOM Data Solutions**
- Comprehensive property data
- API access available
- Pricing: Contact for quote
- Website: https://www.attomdata.com

**B. CoreLogic**
- Industry-standard property data
- Expensive but comprehensive
- Typically for commercial/research use

**C. Zillow Data API**
- Some data available through API
- May have usage limits
- Check: https://www.zillow.com/howto/api/APIOverview.htm

---

## RECOMMENDED APPROACH

### For Quick Analysis (This Week):
**Salem Only**:
1. Download Marion County CSV files
2. Run analysis on Salem data
3. Document methodology
4. Generate preliminary findings

**Result**: You'll have complete analysis for Salem, methodology documented

### For Comprehensive Analysis (2-4 Weeks):
**Both Cities**:
1. Download Salem data (immediate)
2. Submit public records request to Multnomah County (Week 1)
3. Analyze Salem while waiting for Gresham data
4. Complete Gresham analysis when data arrives (Week 3-4)
5. Run comparative analysis

**Result**: Full analysis for both cities with official data

### For Modest Analysis (This Weekend):
**Both Cities with Manual Collection**:
1. Download Salem data
2. Manually collect 50-100 Gresham sales from county website
3. Run analysis on both datasets

**Result**: Statistically valid analysis for both cities, smaller Gresham sample

---

## DATA QUALITY CHECKLIST

When you obtain data, verify:

✓ **Completeness**:
- All required fields present (Sale Date, Price, RMV)
- Reasonable sample size (50+ minimum)

✓ **Validity**:
- Sale prices in reasonable range ($50k-$5M for homes)
- Dates within target timeframe
- RMV values present and non-zero

✓ **Accuracy**:
- Cross-check a few records manually against county website
- Look for obvious data entry errors (extra zeros, etc.)

✓ **Appropriate Transactions**:
- Arms-length sales (not family transfers)
- Residential properties (if that's your focus)
- Exclude foreclosures if desired

---

## SAMPLE SIZE REQUIREMENTS

### Statistical Validity:

**Minimum Acceptable**: 30 sales
- Can detect large effects
- Wide confidence intervals
- Limited statistical power

**Good Sample**: 50-100 sales
- Reasonable statistical power
- Can detect moderate effects
- Narrower confidence intervals

**Excellent Sample**: 100-500 sales
- High statistical power
- Can detect small effects
- Very precise estimates

**Ideal**: 500+ sales
- Maximum statistical rigor
- Subgroup analysis possible
- Seasonal pattern detection

### Current Market Context:

**Salem**: 
- ~153 homes sold per month
- 6 months = ~900 sales
- 12 months = ~1,800 sales
- **Excellent data availability**

**Gresham**:
- ~103 homes sold per month  
- 6 months = ~600 sales
- 12 months = ~1,200 sales
- **Good data availability IF accessible**

---

## TIMELINE ESTIMATES

### Scenario A: Salem Only
- Day 1: Download data (30 minutes)
- Day 1-2: Data cleaning and analysis (2-3 hours)
- Day 2: Generate visualizations (1 hour)
- Day 2-3: Write report (2 hours)
- **Total: 2-3 days**

### Scenario B: Manual Collection for Gresham + Salem
- Day 1: Download Salem data (30 minutes)
- Day 1-2: Manually collect 50-100 Gresham properties (3-4 hours)
- Day 2-3: Data cleaning both cities (3 hours)
- Day 3-4: Analysis and visualizations (3 hours)
- Day 4-5: Comparative report (3 hours)
- **Total: 4-5 days**

### Scenario C: Public Records Request
- Week 1: Submit request
- Week 2-3: Wait for county response
- Week 3-4: Receive data, perform analysis
- Week 4: Complete report
- **Total: 3-4 weeks**

---

## CONTACT INFORMATION

### Marion County (Salem):
**Assessor's Office**
- Website: https://www.co.marion.or.us/AO
- Phone: (503) 588-5215
- Email: propertytax@co.marion.or.us
- Address: 555 Court St NE, Salem, OR 97301

### Multnomah County (Gresham):
**Assessment, Recording & Taxation**
- Website: https://multco.us/assessment-taxation
- Phone: (503) 988-3326
- Address: 501 SE Hawthorne Blvd, Portland, OR 97214

**For Gresham-specific questions**:
- City of Gresham: (503) 618-2760
- GreshamView: https://www.greshamoregon.gov/services/maps-and-gis/

---

## TROUBLESHOOTING

**Problem**: CSV file won't download
- **Solution**: Try different browser, check popup blockers
- **Alternative**: Right-click link, "Save Link As..."

**Problem**: Can't find RMV column in data
- **Solution**: Look for "Real Market Value", "RMV", "Market Value", "Appraised Value"
- **Check**: Marion County keys/documentation on their website

**Problem**: Too many records to process
- **Solution**: Filter by date first (last 6-12 months only)
- **Solution**: Filter by city
- **Solution**: Sample randomly if needed

**Problem**: Public records request denied
- **Solution**: Oregon has strong public records laws - escalate if needed
- **Resource**: Oregon DOJ Public Records Advocate: (503) 934-4100

**Problem**: County website down
- **Solution**: Try off-peak hours (early morning)
- **Solution**: Contact county IT department
- **Solution**: Use cached version or Internet Archive

---

## NEXT STEPS

1. **Decide on your approach** based on time/resources:
   - Quick: Salem only
   - Modest: Salem + manual Gresham collection
   - Comprehensive: Salem + public records request

2. **Download Salem data** (can do this immediately)

3. **For Gresham**, choose:
   - Submit public records request today (if going comprehensive route)
   - Start manual collection (if going modest route)
   - Skip for now (if doing Salem-only pilot)

4. **Run the Python analysis script** once you have the data

5. **Review outputs** and iterate as needed

The analysis framework is ready to go - you just need the raw data!
