"""
Multnomah County Property Data Collector

Downloads Redfin recently-sold data via the Stingray API, enriches with
Multnomah County RMV data from PortlandMaps ArcGIS REST service, and
maintains a persistent research database for countywide analysis.

Usage:
    python multnomah_data_collector.py download                  # all cities, last 365 days
    python multnomah_data_collector.py download --city portland   # single city
    python multnomah_data_collector.py download --days 180        # custom time window
    python multnomah_data_collector.py ingest <csv1> [csv2 ...]  # ingest existing Redfin CSVs
    python multnomah_data_collector.py ingest-dir <directory>    # ingest all CSVs in a directory
    python multnomah_data_collector.py enrich                    # ArcGIS enrich un-enriched records
    python multnomah_data_collector.py status                    # database statistics
    python multnomah_data_collector.py status --city PORTLAND    # filter to one city
"""

import argparse
import io
import os
import re
import sys
import time

import pandas as pd
import requests
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'multnomah')
DOWNLOADS_DIR = os.path.join(DATA_DIR, 'redfin_downloads')
RESEARCH_DB_PATH = os.path.join(DATA_DIR, 'multnomah_research_db.csv')

# Redfin Stingray API for CSV downloads
REDFIN_CSV_URL = "https://www.redfin.com/stingray/api/gis-csv"
REDFIN_DELAY = 2.0  # seconds between Redfin API requests (be polite)
REDFIN_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/122.0.0.0 Safari/537.36'),
    'Accept': 'text/csv',
}

# Redfin region IDs for Multnomah County cities (verified against redfin.com URLs)
REDFIN_REGIONS = {
    'portland':     {'region_id': 30772, 'label': 'Portland'},
    'gresham':      {'region_id': 7995,  'label': 'Gresham'},
    'troutdale':    {'region_id': 18977, 'label': 'Troutdale'},
    'fairview':     {'region_id': 6239,  'label': 'Fairview'},
    'wood village': {'region_id': 20727, 'label': 'Wood Village'},
    'lake oswego':  {'region_id': 30777, 'label': 'Lake Oswego'},
}

# PortlandMaps ArcGIS REST service
ARCGIS_URL = (
    "https://www.portlandmaps.com/arcgis/rest/services/"
    "Public/Taxlots/MapServer/0/query"
)
ARCGIS_FIELDS = (
    "PROPERTYID,SITEADDR,SITECITY,SITEZIP,YEARBUILT,BLDGSQFT,BEDROOMS,"
    "TOTALVAL1,LANDVAL1,BLDGVAL1,MKTVALYR1,"
    "TOTALVAL2,LANDVAL2,BLDGVAL2,MKTVALYR2,"
    "TOTALVAL3,LANDVAL3,BLDGVAL3,MKTVALYR3,"
    "PROP_CODE,SALEPRICE,SALEDATE"
)
ARCGIS_DELAY = 0.5  # seconds between ArcGIS requests

# Redfin city name -> ArcGIS SITECITY mapping
REDFIN_TO_ARCGIS_CITY = {
    'PORTLAND': 'PORTLAND',
    'GRESHAM': 'GRESHAM',
    'TROUTDALE': 'TROUTDALE',
    'FAIRVIEW': 'FAIRVIEW',
    'WOOD VILLAGE': 'WOOD VILLAGE',
    'LAKE OSWEGO': 'LAKE OSWEGO',
    'BORING': 'BORING',
    'CORBETT': 'CORBETT',
    'MAYWOOD PARK': 'MAYWOOD PARK',
    'DAMASCUS': 'DAMASCUS',
    'HAPPY VALLEY': 'HAPPY VALLEY',
    'MILWAUKIE': 'MILWAUKIE',
    'ORIENT': 'ORIENT',
    'BRIDAL VEIL': 'BRIDAL VEIL',
}

# Region groupings for analysis
REGIONS = {
    'east': {'GRESHAM', 'TROUTDALE', 'FAIRVIEW', 'WOOD VILLAGE'},
    'west': {'PORTLAND', 'LAKE OSWEGO', 'MAYWOOD PARK'},
}

# Address normalization maps
DIRECTIONAL_MAP = {
    'NORTH': 'N', 'SOUTH': 'S', 'EAST': 'E', 'WEST': 'W',
    'NORTHEAST': 'NE', 'NORTHWEST': 'NW',
    'SOUTHEAST': 'SE', 'SOUTHWEST': 'SW',
}
STREET_TYPE_MAP = {
    'STREET': 'ST', 'AVENUE': 'AVE', 'BOULEVARD': 'BLVD', 'DRIVE': 'DR',
    'ROAD': 'RD', 'LANE': 'LN', 'COURT': 'CT', 'CIRCLE': 'CIR',
    'PLACE': 'PL', 'WAY': 'WAY', 'TERRACE': 'TER', 'TRAIL': 'TRL',
    'PARKWAY': 'PKWY', 'LOOP': 'LOOP', 'PATH': 'PATH',
}


# ---------------------------------------------------------------------------
# Address normalization (from gresham_data_collector.py)
# ---------------------------------------------------------------------------

def normalize_address(address):
    """Normalize an address for ArcGIS SITEADDR matching."""
    if not address or not isinstance(address, str):
        return ''
    addr = address.upper().strip()
    addr = re.sub(r'\s*#\s*\S+', '', addr)
    addr = re.sub(r'\s+(UNIT|APT|STE|SUITE|BLDG)\s*\S*', '', addr)
    for full, abbr in DIRECTIONAL_MAP.items():
        addr = re.sub(rf'\b{full}\b', abbr, addr)
    for full, abbr in STREET_TYPE_MAP.items():
        addr = re.sub(rf'\b{full}\b', abbr, addr)
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr


def normalize_city(redfin_city):
    """Normalize a Redfin city name to the ArcGIS SITECITY value."""
    if not redfin_city or not isinstance(redfin_city, str):
        return ''
    upper = redfin_city.upper().strip()
    return REDFIN_TO_ARCGIS_CITY.get(upper, upper)


# ---------------------------------------------------------------------------
# ArcGIS lookup (from gresham_data_collector.py)
# ---------------------------------------------------------------------------

def query_arcgis(where_clause):
    """Execute a single ArcGIS REST query. Returns list of attribute dicts."""
    params = {
        'where': where_clause,
        'outFields': ARCGIS_FIELDS,
        'returnGeometry': 'false',
        'f': 'json',
    }
    try:
        resp = requests.get(ARCGIS_URL, params=params, timeout=10, verify=False)
        resp.raise_for_status()
        data = resp.json()
        if 'error' in data:
            return []
        return [f['attributes'] for f in data.get('features', [])]
    except (requests.RequestException, ValueError):
        return []


def lookup_property(address, city='GRESHAM'):
    """
    Look up a property by address. Tries exact match first, then fuzzy.
    Returns (attributes_dict, match_method, confidence) or (None, None, 0).
    """
    normalized = normalize_address(address)
    if not normalized:
        return None, None, 0

    # Tier 1: Exact prefix match
    escaped_addr = normalized.replace("'", "''")
    escaped_city = city.replace("'", "''")
    where = f"SITEADDR LIKE '{escaped_addr}%' AND SITECITY='{escaped_city}'"
    results = query_arcgis(where)

    if len(results) == 1:
        return results[0], 'exact', 1.0
    elif len(results) > 1:
        best = min(results, key=lambda r: abs(
            len(r.get('SITEADDR', '').strip()) - len(normalized)))
        return best, 'exact_multi', 0.9

    # Tier 2: Fuzzy — extract house number + street name core
    match = re.match(r'^(\d+)\s+(.+)', normalized)
    if not match:
        return None, None, 0

    house_num = match.group(1)
    street_part = match.group(2)
    core = street_part
    core = re.sub(r'^(N|S|E|W|NE|NW|SE|SW)\s+', '', core)
    core = re.sub(
        r'\s+(ST|AVE|BLVD|DR|RD|LN|CT|CIR|PL|WAY|TER|TRL|PKWY|LOOP|PATH)$',
        '', core)
    if not core:
        return None, None, 0

    where = f"SITEADDR LIKE '{house_num}%{core}%' AND SITECITY='{escaped_city}'"
    results = query_arcgis(where)

    if len(results) == 1:
        return results[0], 'fuzzy', 0.7
    elif len(results) > 1:
        best = min(results, key=lambda r: abs(
            len(r.get('SITEADDR', '').strip()) - len(normalized)))
        return best, 'fuzzy_multi', 0.5

    return None, None, 0


def select_rmv_for_sale_year(attrs, sale_year):
    """Pick the assessment year closest to the sale year."""
    years = []
    for i in (1, 2, 3):
        yr_raw = attrs.get(f'MKTVALYR{i}')
        total = attrs.get(f'TOTALVAL{i}')
        if yr_raw is not None and total is not None:
            yr_str = str(yr_raw)
            yr_match = re.search(r'(\d{4})', yr_str)
            if yr_match:
                yr = int(yr_match.group(1))
                years.append((yr, i))
    if not years:
        return None, None, None, None
    best_yr, best_idx = min(years, key=lambda t: abs(t[0] - sale_year))
    return (
        attrs.get(f'LANDVAL{best_idx}'),
        attrs.get(f'BLDGVAL{best_idx}'),
        attrs.get(f'TOTALVAL{best_idx}'),
        best_yr,
    )


# ---------------------------------------------------------------------------
# Redfin API download
# ---------------------------------------------------------------------------

def _fetch_redfin_page(region_id, label, sold_within_days, page=1,
                       min_price=None, max_price=None):
    """
    Fetch a single page from Redfin's Stingray CSV API.
    Returns a cleaned DataFrame (Oregon-only, valid prices), or empty DataFrame.
    """
    params = {
        'al': 1,
        'num_homes': 350,
        'ord': 'redfin-recommended-asc',
        'page_number': page,
        'region_id': region_id,
        'region_type': 6,
        'sold_within_days': sold_within_days,
        'status': 9,
        'uipt': '1,2,3,4,5,6,7,8',
        'v': 8,
    }
    if min_price is not None:
        params['min_price'] = min_price
    if max_price is not None:
        params['max_price'] = max_price

    price_label = ''
    if min_price or max_price:
        lo = f'${min_price/1000:.0f}K' if min_price else '$0'
        hi = f'${max_price/1000:.0f}K' if max_price else 'max'
        price_label = f' [{lo}-{hi}]'

    print(f"  [{label}] Fetching page {page}{price_label}...", end=' ', flush=True)
    try:
        resp = requests.get(REDFIN_CSV_URL, params=params,
                            headers=REDFIN_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"FAILED: {e}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"PARSE ERROR: {e}")
        return pd.DataFrame()

    if len(df) > 0:
        df['_price_check'] = pd.to_numeric(
            df['PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce')
        df = df[df['_price_check'].notna()].drop(columns=['_price_check'])
        if 'STATE OR PROVINCE' in df.columns:
            df = df[df['STATE OR PROVINCE'].astype(str).str.strip() == 'OR']

    if len(df) == 0:
        print("0 records")
    else:
        print(f"{len(df)} records")
    return df


# Price bands for large-city downloads (Redfin caps at ~350 per query).
# Using bands ensures we pull comprehensive data for cities like Portland.
PRICE_BANDS = [
    (None, 400_000),
    (400_000, 600_000),
    (600_000, 800_000),
    (800_000, 1_200_000),
    (1_200_000, None),
]


def download_redfin_city(city_key, sold_within_days=365):
    """
    Download recently-sold CSV from Redfin's Stingray API for a single city.
    For cities where a single query returns 350 rows (the cap), uses price-band
    windowing to get more complete coverage. Returns combined, deduped DataFrame.
    """
    city_info = REDFIN_REGIONS[city_key]
    region_id = city_info['region_id']
    label = city_info['label']

    # First attempt: single query, no price filter
    df = _fetch_redfin_page(region_id, label, sold_within_days)
    time.sleep(REDFIN_DELAY)

    if len(df) < 300:
        # Small enough that we likely got everything — return as-is
        return df

    # Likely hit Redfin's ~350-row cap (may be <350 after OR filter).
    # Use price bands to get wider coverage.
    print(f"  [{label}] Hit 350-row cap; using price bands for wider coverage...")
    all_frames = [df]  # keep the initial batch

    for min_p, max_p in PRICE_BANDS:
        band_df = _fetch_redfin_page(region_id, label, sold_within_days,
                                     min_price=min_p, max_price=max_p)
        if not band_df.empty:
            all_frames.append(band_df)
        time.sleep(REDFIN_DELAY)

    combined = pd.concat(all_frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=['ADDRESS', 'SOLD DATE', 'PRICE'], keep='first')
    print(f"  [{label}] Combined: {before} -> {len(combined)} unique records")
    return combined


def cmd_download(args):
    """Download Redfin recently-sold data for Multnomah County cities."""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    days = args.days
    cities = list(REDFIN_REGIONS.keys())
    if args.city:
        key = args.city.lower().strip()
        if key not in REDFIN_REGIONS:
            print(f"Unknown city: {args.city}")
            print(f"Available: {', '.join(REDFIN_REGIONS.keys())}")
            sys.exit(1)
        cities = [key]

    print("=" * 60)
    print("MULTNOMAH COUNTY DATA COLLECTOR — Redfin Download")
    print(f"Cities: {', '.join(c.title() for c in cities)}")
    print(f"Window: last {days} days")
    print("=" * 60)

    all_frames = []
    for city_key in cities:
        print(f"\n--- {REDFIN_REGIONS[city_key]['label']} ---")
        df = download_redfin_city(city_key, sold_within_days=days)
        if not df.empty:
            all_frames.append(df)
            # Save individual city CSV for auditability
            timestamp = datetime.now().strftime('%Y-%m-%d')
            filename = f"redfin_{city_key.replace(' ', '_')}_{timestamp}.csv"
            filepath = os.path.join(DOWNLOADS_DIR, filename)
            df.to_csv(filepath, index=False)
            print(f"  Saved: {filename} ({len(df)} records)")
        time.sleep(REDFIN_DELAY)

    if not all_frames:
        print("\nNo data downloaded.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Save combined file
    combined_path = os.path.join(DOWNLOADS_DIR, 'combined_recent.csv')
    combined.to_csv(combined_path, index=False)

    print(f"\n--- Download Summary ---")
    if 'CITY' in combined.columns:
        for city, count in combined['CITY'].value_counts().items():
            print(f"  {city}: {count} records")
    print(f"  Total: {len(combined)} records")
    print(f"  Combined CSV: {combined_path}")

    # Now ingest into the research DB (without ArcGIS enrichment yet)
    print(f"\n--- Ingesting into research database ---")
    parsed = parse_redfin_df(combined)
    merged, new_count = merge_into_research_db(parsed)
    print(f"  Added {new_count} new records (database total: {len(merged)})")
    print(f"\nRun 'enrich' to add county RMV data via ArcGIS.")


# ---------------------------------------------------------------------------
# Redfin CSV parsing (adapted from gresham_data_collector.py)
# ---------------------------------------------------------------------------

def parse_redfin_df(df):
    """
    Parse a Redfin DataFrame into our standard record format.
    No city filter — keeps all rows. City comes from Redfin CITY column.
    Returns DataFrame with our standard columns (RMV columns will be NaN
    until enrichment).
    """
    records = []

    for _, row in df.iterrows():
        # Extract Redfin columns (case-insensitive)
        address = ''
        city = ''
        zip_code = ''
        price = None
        sold_date = None
        beds = None
        sqft = None
        year_built = None

        for col in df.columns:
            upper = col.upper().strip()
            if upper == 'ADDRESS':
                address = str(row[col]) if pd.notna(row[col]) else ''
            elif upper == 'CITY':
                city = str(row[col]) if pd.notna(row[col]) else ''
            elif upper in ('ZIP OR POSTAL CODE', 'ZIP'):
                zip_code = row[col]
            elif upper == 'PRICE':
                price = pd.to_numeric(
                    str(row[col]).replace('$', '').replace(',', ''),
                    errors='coerce') if pd.notna(row[col]) else None
            elif upper == 'SOLD DATE':
                sold_date = pd.to_datetime(row[col], errors='coerce')
            elif upper == 'BEDS':
                beds = row[col]
            elif upper == 'SQUARE FEET':
                sqft = row[col]
            elif upper == 'YEAR BUILT':
                year_built = row[col]

        arcgis_city = normalize_city(city)

        record = {
            'Address': normalize_address(address),
            'City': arcgis_city,
            'SalePrice': price,
            'SaleDate': sold_date,
            'Source': 'redfin',
            'DateCollected': datetime.now().isoformat(),
            'MatchMethod': None,
            'MatchConfidence': None,
            'PropertyID': None,
            'Zip': zip_code,
            'RMVLAND': None,
            'RMVIMPR': None,
            'RMV': None,
            'RMV_Year': None,
            'YearBuilt': year_built,
            'SqFt': sqft,
            'Bedrooms': beds,
        }
        records.append(record)

    return pd.DataFrame(records)


def load_redfin_csv(filepath):
    """Load a Redfin CSV file. No city filter — all rows kept."""
    df = pd.read_csv(filepath)

    # Drop MLS disclaimer rows
    if 'PRICE' in df.columns:
        df['_price_check'] = pd.to_numeric(
            df['PRICE'].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce')
        df = df[df['_price_check'].notna()].drop(columns=['_price_check'])
    elif 'Price' in df.columns:
        df['_price_check'] = pd.to_numeric(
            df['Price'].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce')
        df = df[df['_price_check'].notna()].drop(columns=['_price_check'])

    print(f"  Loaded {len(df)} records from {os.path.basename(filepath)}")
    return df


# ---------------------------------------------------------------------------
# Research database management
# ---------------------------------------------------------------------------

def load_research_db():
    """Load the persistent Multnomah research database."""
    if os.path.exists(RESEARCH_DB_PATH):
        df = pd.read_csv(RESEARCH_DB_PATH)
        df['SaleDate'] = pd.to_datetime(df['SaleDate'], errors='coerce')
        return df
    return pd.DataFrame()


def save_research_db(df):
    """Save the Multnomah research database."""
    os.makedirs(os.path.dirname(RESEARCH_DB_PATH), exist_ok=True)
    df.to_csv(RESEARCH_DB_PATH, index=False)


def merge_into_research_db(new_data):
    """
    Merge new records into the persistent research database.
    Deduplicates on (Address, City, SaleDate).
    Returns (merged_df, new_count).
    """
    db = load_research_db()

    if db.empty:
        save_research_db(new_data)
        return new_data, len(new_data)

    combined = pd.concat([db, new_data], ignore_index=True)
    before_db = len(db)

    combined['_dedup_date'] = combined['SaleDate'].astype(str).str[:10]
    combined = combined.sort_values('DateCollected', ascending=False)
    combined = combined.drop_duplicates(
        subset=['Address', 'City', '_dedup_date'], keep='first')
    combined = combined.drop(columns=['_dedup_date'])

    new_count = len(combined) - before_db
    save_research_db(combined)
    return combined, new_count


# ---------------------------------------------------------------------------
# ArcGIS enrichment
# ---------------------------------------------------------------------------

def cmd_enrich(args):
    """Enrich un-enriched records in the research DB with ArcGIS RMV data."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    db = load_research_db()
    if db.empty:
        print("Research database is empty. Run 'download' or 'ingest' first.")
        return

    # Find records that need enrichment (RMV is NaN or 0)
    needs_enrichment = db['RMV'].isna() | (db['RMV'] == 0)
    to_enrich = db[needs_enrichment].copy()
    already_done = db[~needs_enrichment].copy()

    if len(to_enrich) == 0:
        print(f"All {len(db)} records are already enriched. Nothing to do.")
        return

    print("=" * 60)
    print("MULTNOMAH COUNTY DATA COLLECTOR — ArcGIS Enrichment")
    print("=" * 60)
    print(f"  Total records: {len(db)}")
    print(f"  Already enriched: {len(already_done)}")
    print(f"  Need enrichment: {len(to_enrich)}")
    est_minutes = len(to_enrich) * ARCGIS_DELAY / 60
    print(f"  Estimated time: ~{est_minutes:.0f} minutes")
    print()

    matched = 0
    failed = 0
    enriched_records = []

    for i, (idx, row) in enumerate(to_enrich.iterrows()):
        address = row.get('Address', '')
        city = row.get('City', '')
        sale_date = row.get('SaleDate')
        sale_year = sale_date.year if pd.notna(sale_date) else 2025

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(to_enrich)}] {address}, {city}")

        attrs, method, confidence = lookup_property(address, city=city)
        time.sleep(ARCGIS_DELAY)

        row_dict = row.to_dict()
        row_dict['MatchMethod'] = method
        row_dict['MatchConfidence'] = confidence

        if attrs:
            land, bldg, total_rmv, rmv_year = select_rmv_for_sale_year(
                attrs, int(sale_year))
            row_dict.update({
                'PropertyID': attrs.get('PROPERTYID'),
                'Zip': attrs.get('SITEZIP', row.get('Zip')),
                'RMVLAND': land,
                'RMVIMPR': bldg,
                'RMV': total_rmv,
                'RMV_Year': rmv_year,
                'YearBuilt': attrs.get('YEARBUILT', row.get('YearBuilt')),
                'SqFt': attrs.get('BLDGSQFT', row.get('SqFt')),
                'Bedrooms': attrs.get('BEDROOMS', row.get('Bedrooms')),
            })
            matched += 1
        else:
            failed += 1
            if (i + 1) <= 50 or failed <= 20:
                print(f"    UNMATCHED: {address} ({city})")

        enriched_records.append(row_dict)

    enriched_df = pd.DataFrame(enriched_records)

    # Combine with already-enriched records
    final = pd.concat([already_done, enriched_df], ignore_index=True)
    save_research_db(final)

    print(f"\n--- Enrichment Complete ---")
    print(f"  Matched: {matched}")
    print(f"  Unmatched: {failed}")
    if matched + failed > 0:
        print(f"  Match rate: {matched / (matched + failed) * 100:.1f}%")
    print(f"  Database saved: {RESEARCH_DB_PATH}")


# ---------------------------------------------------------------------------
# Ingest existing Redfin CSVs
# ---------------------------------------------------------------------------

def cmd_ingest(args):
    """Ingest one or more existing Redfin CSV files."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    files = args.files
    if not files:
        print("Error: provide one or more CSV file paths")
        sys.exit(1)

    print("=" * 60)
    print("MULTNOMAH COUNTY DATA COLLECTOR — CSV Ingest")
    print("=" * 60)

    all_frames = []
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"  WARNING: file not found: {filepath}")
            continue
        print(f"\n--- Loading {os.path.basename(filepath)} ---")
        df = load_redfin_csv(filepath)
        if not df.empty:
            parsed = parse_redfin_df(df)
            all_frames.append(parsed)

    if not all_frames:
        print("No valid records found.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n--- Merging {len(combined)} records into research database ---")
    merged, new_count = merge_into_research_db(combined)
    print(f"  Added {new_count} new records (database total: {len(merged)})")
    print(f"\nRun 'enrich' to add county RMV data via ArcGIS.")


def cmd_ingest_dir(args):
    """Ingest all CSV files in a directory."""
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"Error: not a directory: {directory}")
        sys.exit(1)

    csv_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith('.csv')
    ])

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    print(f"Found {len(csv_files)} CSV files in {directory}")
    args.files = csv_files
    cmd_ingest(args)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def cmd_status(args):
    """Print research database statistics."""
    db = load_research_db()
    if db.empty:
        print("Research database is empty. Run 'download' or 'ingest' first.")
        return

    city_filter = args.city.upper().strip() if args.city else None
    if city_filter:
        db = db[db['City'].str.upper() == city_filter]
        if db.empty:
            print(f"No records for city: {city_filter}")
            return

    label = city_filter if city_filter else 'MULTNOMAH COUNTY'

    print("=" * 60)
    print(f"{label} RESEARCH DATABASE STATUS")
    print("=" * 60)
    print(f"  Total records:     {len(db)}")

    # City breakdown
    if not city_filter and 'City' in db.columns:
        print(f"\n  Records by city:")
        for city, count in db['City'].value_counts().items():
            print(f"    {city}: {count}")

    has_rmv = db['RMV'].notna() & (db['RMV'] > 0)
    has_price = db['SalePrice'].notna() & (db['SalePrice'] > 0)
    both = has_rmv & has_price

    print(f"\n  With RMV:          {has_rmv.sum()}")
    print(f"  With Sale Price:   {has_price.sum()}")
    print(f"  Analysis-ready:    {both.sum()} (have both RMV and sale price)")

    not_enriched = (~has_rmv) & has_price
    if not_enriched.sum() > 0:
        print(f"  Awaiting enrichment: {not_enriched.sum()}")

    if 'SaleDate' in db.columns:
        dates = pd.to_datetime(db['SaleDate'], errors='coerce').dropna()
        if not dates.empty:
            print(f"  Date range:        {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")

    if 'MatchMethod' in db.columns:
        methods = db['MatchMethod'].dropna()
        if not methods.empty:
            print(f"\n  Match methods:")
            for method, count in methods.value_counts().items():
                print(f"    {method}: {count}")

    # High-value summary
    if both.sum() > 0:
        analysis_df = db[both]
        high_val = analysis_df[analysis_df['RMV'] >= 800000]
        if len(high_val) > 0:
            print(f"\n  High-value (RMV >= $800K): {len(high_val)} records")
            rmvland = pd.to_numeric(high_val['RMVLAND'], errors='coerce')
            rmv = pd.to_numeric(high_val['RMV'], errors='coerce')
            land_pct = rmvland / rmv * 100
            land_dom = (land_pct >= 50).sum()
            print(f"    Land-dominant (>=50% land): {land_dom}")
            print(f"    Building-dominant (<50% land): {len(high_val) - land_dom}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multnomah County property data collector. Downloads '
                    'Redfin sales data and enriches with county RMV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # download
    dl = subparsers.add_parser('download',
        help='Download recently-sold data from Redfin API')
    dl.add_argument('--city', type=str, default=None,
        help='Download a single city (e.g., portland, gresham)')
    dl.add_argument('--days', type=int, default=365,
        help='Sold within N days (default: 365)')

    # ingest
    ing = subparsers.add_parser('ingest',
        help='Ingest existing Redfin CSV files')
    ing.add_argument('files', nargs='+', help='Redfin CSV file(s)')

    # ingest-dir
    igd = subparsers.add_parser('ingest-dir',
        help='Ingest all CSVs in a directory')
    igd.add_argument('directory', help='Directory containing Redfin CSVs')

    # enrich
    subparsers.add_parser('enrich',
        help='Enrich un-enriched records with ArcGIS RMV data')

    # status
    st = subparsers.add_parser('status',
        help='Show research database statistics')
    st.add_argument('--city', type=str, default=None,
        help='Filter to a single city')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'download':
        cmd_download(args)
    elif args.command == 'ingest':
        cmd_ingest(args)
    elif args.command == 'ingest-dir':
        cmd_ingest_dir(args)
    elif args.command == 'enrich':
        cmd_enrich(args)
    elif args.command == 'status':
        cmd_status(args)


if __name__ == '__main__':
    main()
