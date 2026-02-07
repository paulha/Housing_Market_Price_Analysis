"""
Gresham Property Data Collector

Ingests Redfin recently-sold CSVs, enriches with Multnomah County RMV data
from PortlandMaps ArcGIS REST service, and maintains a persistent research
database for ongoing analysis.

Usage:
    python gresham_data_collector.py ingest <redfin_csv_path>
    python gresham_data_collector.py status
"""

import os
import re
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESEARCH_DB_PATH = os.path.join(BASE_DIR, 'data', 'gresham', 'gresham_research_db.csv')

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
REQUEST_DELAY = 0.5  # seconds between ArcGIS requests

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


def normalize_address(address):
    """
    Normalize an address for ArcGIS SITEADDR matching.

    Handles: case, directionals, street types, unit numbers, whitespace.
    """
    if not address or not isinstance(address, str):
        return ''

    addr = address.upper().strip()

    # Remove unit/apt suffixes: "#101", "UNIT B", "APT 2", "STE 100"
    addr = re.sub(r'\s*#\s*\S+', '', addr)
    addr = re.sub(r'\s+(UNIT|APT|STE|SUITE|BLDG)\s*\S*', '', addr)

    # Standardize directionals
    for full, abbr in DIRECTIONAL_MAP.items():
        addr = re.sub(rf'\b{full}\b', abbr, addr)

    # Standardize street types
    for full, abbr in STREET_TYPE_MAP.items():
        addr = re.sub(rf'\b{full}\b', abbr, addr)

    # Collapse whitespace
    addr = re.sub(r'\s+', ' ', addr).strip()

    return addr


def query_arcgis(where_clause):
    """
    Execute a single ArcGIS REST query. Returns list of feature attribute dicts.
    Retries up to 3 times with exponential backoff.
    """
    params = {
        'where': where_clause,
        'outFields': ARCGIS_FIELDS,
        'returnGeometry': 'false',
        'f': 'json',
    }

    for attempt in range(1):
        try:
            resp = requests.get(ARCGIS_URL, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if 'error' in data:
                print(f"    ArcGIS error: {data['error'].get('message', data['error'])}")
                return []
            return [f['attributes'] for f in data.get('features', [])]
        except (requests.RequestException, ValueError) as e:
            print(f"    ArcGIS request failed: {e}")
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
    where = f"SITEADDR LIKE '{normalized}%' AND SITECITY='{city}'"
    results = query_arcgis(where)

    if len(results) == 1:
        return results[0], 'exact', 1.0
    elif len(results) > 1:
        # Multiple matches — pick closest by string length
        best = min(results, key=lambda r: abs(len(r.get('SITEADDR', '').strip()) - len(normalized)))
        return best, 'exact_multi', 0.9

    # Tier 2: Fuzzy — extract house number + street name core
    match = re.match(r'^(\d+)\s+(.+)', normalized)
    if not match:
        return None, None, 0

    house_num = match.group(1)
    street_part = match.group(2)

    # Remove directional prefix and street type suffix for fuzzy core
    core = street_part
    core = re.sub(r'^(N|S|E|W|NE|NW|SE|SW)\s+', '', core)
    core = re.sub(r'\s+(ST|AVE|BLVD|DR|RD|LN|CT|CIR|PL|WAY|TER|TRL|PKWY|LOOP|PATH)$', '', core)

    if not core:
        return None, None, 0

    where = f"SITEADDR LIKE '{house_num}%{core}%' AND SITECITY='{city}'"
    results = query_arcgis(where)

    if len(results) == 1:
        return results[0], 'fuzzy', 0.7
    elif len(results) > 1:
        best = min(results, key=lambda r: abs(len(r.get('SITEADDR', '').strip()) - len(normalized)))
        return best, 'fuzzy_multi', 0.5

    return None, None, 0


def select_rmv_for_sale_year(attrs, sale_year):
    """
    Pick the assessment year closest to the sale year.
    Returns (land_val, bldg_val, total_val, rmv_year).
    """
    years = []
    for i in (1, 2, 3):
        yr_raw = attrs.get(f'MKTVALYR{i}')
        total = attrs.get(f'TOTALVAL{i}')
        if yr_raw is not None and total is not None:
            # MKTVALYR can be "2023" or "9/27/2025" — extract the year
            yr_str = str(yr_raw)
            yr_match = re.search(r'(\d{4})', yr_str)
            if yr_match:
                yr = int(yr_match.group(1))
                years.append((yr, i))

    if not years:
        return None, None, None, None

    # Find closest assessment year to sale year
    best_yr, best_idx = min(years, key=lambda t: abs(t[0] - sale_year))

    return (
        attrs.get(f'LANDVAL{best_idx}'),
        attrs.get(f'BLDGVAL{best_idx}'),
        attrs.get(f'TOTALVAL{best_idx}'),
        best_yr,
    )


def load_redfin_csv(filepath):
    """Load a Redfin recently-sold CSV download."""
    df = pd.read_csv(filepath)

    # Redfin columns use uppercase names
    col_map = {}
    for col in df.columns:
        upper = col.upper().strip()
        if upper == 'ADDRESS':
            col_map[col] = 'RedfimAddress'
        elif upper == 'CITY':
            col_map[col] = 'RedfimCity'
        elif upper in ('ZIP OR POSTAL CODE', 'ZIP'):
            col_map[col] = 'RedfimZip'
        elif upper == 'PRICE':
            col_map[col] = 'RedfimPrice'
        elif upper == 'SOLD DATE':
            col_map[col] = 'RedfimSoldDate'
        elif upper == 'BEDS':
            col_map[col] = 'RedfimBeds'
        elif upper == 'BATHS':
            col_map[col] = 'RedfimBaths'
        elif upper == 'SQUARE FEET':
            col_map[col] = 'RedfimSqFt'
        elif upper == 'YEAR BUILT':
            col_map[col] = 'RedfimYearBuilt'
        elif upper == 'URL (SEE HTTP://WWW.REDFIN.COM/BUY-A-HOME/COMPARATIVE-MARKET-ANALYSIS FOR INFO ON PRICING)':
            col_map[col] = 'RedfimURL'
        elif upper == 'URL':
            col_map[col] = 'RedfimURL'

    df = df.rename(columns=col_map)

    # Filter to Gresham
    if 'RedfimCity' in df.columns:
        before = len(df)
        df = df[df['RedfimCity'].astype(str).str.upper().str.strip() == 'GRESHAM'].copy()
        if len(df) < before:
            print(f"  Filtered to Gresham: {len(df)} of {before} rows")

    # Parse price
    if 'RedfimPrice' in df.columns:
        df['SalePrice'] = pd.to_numeric(
            df['RedfimPrice'].astype(str).str.replace(r'[\$,]', '', regex=True),
            errors='coerce'
        )

    # Parse date
    if 'RedfimSoldDate' in df.columns:
        df['SaleDate'] = pd.to_datetime(df['RedfimSoldDate'], errors='coerce')
        df['SaleYear'] = df['SaleDate'].dt.year

    print(f"  Loaded {len(df)} Redfin records from {os.path.basename(filepath)}")
    return df


def enrich_with_rmv(redfin_df):
    """
    For each Redfin property, look up county RMV from ArcGIS.
    Returns enriched DataFrame.
    """
    records = []
    total = len(redfin_df)
    matched = 0
    fuzzy = 0
    failed = 0

    print(f"\n  Enriching {total} properties with county RMV data...")
    print(f"  (estimated time: ~{total * REQUEST_DELAY:.0f}s with rate limiting)\n")

    for i, (_, row) in enumerate(redfin_df.iterrows()):
        address = row.get('RedfimAddress', '')
        sale_year = row.get('SaleYear', 2025)

        if pd.isna(sale_year):
            sale_year = 2025

        # Progress
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{total}] Looking up: {address}")

        attrs, method, confidence = lookup_property(address)
        time.sleep(REQUEST_DELAY)

        record = {
            'Address': normalize_address(address),
            'City': 'GRESHAM',
            'SalePrice': row.get('SalePrice'),
            'SaleDate': row.get('SaleDate'),
            'Source': 'redfin',
            'DateCollected': datetime.now().isoformat(),
            'MatchMethod': method,
            'MatchConfidence': confidence,
        }

        if attrs:
            land, bldg, total_rmv, rmv_year = select_rmv_for_sale_year(attrs, int(sale_year))
            record.update({
                'PropertyID': attrs.get('PROPERTYID'),
                'Zip': attrs.get('SITEZIP'),
                'RMVLAND': land,
                'RMVIMPR': bldg,
                'RMV': total_rmv,
                'RMV_Year': rmv_year,
                'YearBuilt': attrs.get('YEARBUILT'),
                'SqFt': attrs.get('BLDGSQFT'),
                'Bedrooms': attrs.get('BEDROOMS'),
            })
            if method and 'fuzzy' in method:
                fuzzy += 1
            matched += 1
        else:
            record.update({
                'PropertyID': None, 'Zip': row.get('RedfimZip'),
                'RMVLAND': None, 'RMVIMPR': None, 'RMV': None,
                'RMV_Year': None, 'YearBuilt': row.get('RedfimYearBuilt'),
                'SqFt': row.get('RedfimSqFt'), 'Bedrooms': row.get('RedfimBeds'),
            })
            failed += 1
            print(f"    UNMATCHED: {address}")

        records.append(record)

    result = pd.DataFrame(records)
    print(f"\n  Enrichment complete: {matched} matched ({fuzzy} fuzzy), {failed} unmatched")
    return result


def load_research_db():
    """Load the persistent research database, or return empty DataFrame."""
    if os.path.exists(RESEARCH_DB_PATH):
        df = pd.read_csv(RESEARCH_DB_PATH)
        df['SaleDate'] = pd.to_datetime(df['SaleDate'], errors='coerce')
        return df
    return pd.DataFrame()


def save_research_db(df):
    """Save the research database."""
    os.makedirs(os.path.dirname(RESEARCH_DB_PATH), exist_ok=True)
    df.to_csv(RESEARCH_DB_PATH, index=False)


def merge_into_research_db(new_data):
    """
    Merge new records into the persistent research database.
    Deduplicates on (Address, SaleDate).
    Returns (merged_df, new_count).
    """
    db = load_research_db()

    if db.empty:
        save_research_db(new_data)
        return new_data, len(new_data)

    # Combine and deduplicate — keep latest collection for each (Address, SaleDate)
    combined = pd.concat([db, new_data], ignore_index=True)
    before = len(combined)

    combined['_dedup_date'] = combined['SaleDate'].astype(str).str[:10]
    combined = combined.sort_values('DateCollected', ascending=False)
    combined = combined.drop_duplicates(subset=['Address', '_dedup_date'], keep='first')
    combined = combined.drop(columns=['_dedup_date'])

    new_count = len(combined) - len(db)
    save_research_db(combined)
    return combined, new_count


def print_status():
    """Print research database statistics."""
    db = load_research_db()
    if db.empty:
        print("Research database is empty. Run 'ingest' first.")
        return

    print("=" * 60)
    print("GRESHAM RESEARCH DATABASE STATUS")
    print("=" * 60)
    print(f"  Total records:     {len(db)}")

    has_rmv = db['RMV'].notna() & (db['RMV'] > 0)
    has_price = db['SalePrice'].notna() & (db['SalePrice'] > 0)
    both = has_rmv & has_price
    print(f"  With RMV:          {has_rmv.sum()}")
    print(f"  With Sale Price:   {has_price.sum()}")
    print(f"  Analysis-ready:    {both.sum()} (have both RMV and sale price)")

    if 'SaleDate' in db.columns:
        dates = pd.to_datetime(db['SaleDate'], errors='coerce').dropna()
        if not dates.empty:
            print(f"  Date range:        {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")

    if 'MatchMethod' in db.columns:
        print(f"\n  Match methods:")
        for method, count in db['MatchMethod'].value_counts().items():
            print(f"    {method}: {count}")

    unmatched = db['RMV'].isna() | (db['RMV'] == 0)
    if unmatched.any():
        print(f"\n  Unmatched addresses ({unmatched.sum()}):")
        for addr in db.loc[unmatched, 'Address'].head(10):
            print(f"    - {addr}")

    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python gresham_data_collector.py ingest <redfin_csv>")
        print("  python gresham_data_collector.py status")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'ingest':
        if len(sys.argv) < 3:
            print("Error: provide path to Redfin CSV")
            sys.exit(1)

        csv_path = sys.argv[2]
        if not os.path.exists(csv_path):
            print(f"Error: file not found: {csv_path}")
            sys.exit(1)

        print("=" * 60)
        print("GRESHAM DATA COLLECTOR — Redfin + ArcGIS Enrichment")
        print("=" * 60)

        print("\n--- Loading Redfin CSV ---")
        redfin_df = load_redfin_csv(csv_path)

        if redfin_df.empty:
            print("No Gresham records found in CSV.")
            sys.exit(1)

        print("\n--- Looking up county RMV data ---")
        enriched = enrich_with_rmv(redfin_df)

        print("\n--- Merging into research database ---")
        merged, new_count = merge_into_research_db(enriched)
        print(f"  Added {new_count} new records (database total: {len(merged)})")
        print(f"  Saved to: {RESEARCH_DB_PATH}")

        print("\n--- Summary ---")
        print_status()

    elif command == 'status':
        print_status()

    else:
        print(f"Unknown command: {command}")
        print("Use 'ingest' or 'status'")
        sys.exit(1)


if __name__ == '__main__':
    main()
