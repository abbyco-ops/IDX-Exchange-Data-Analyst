# Feature Engineering (Weeks 5-6)

# Reads the cleaned datasets from data_cleaning.py:
#   csv_files/sold_cleaned.csv      -> csv_files/sold_engineered.csv
#   csv_files/listings_cleaned.csv  -> csv_files/listings_engineered.csv

import os
import pandas as pd

sold_input = "csv_files/sold_cleaned.csv"
listings_input = "csv_files/listings_cleaned.csv"
school_districts = "csv_files/school_districts/DistrictAreas2425.shp"

summary_dir = "csv_files/week6_summaries"

date_fields = ["ListingContractDate", "PurchaseContractDate", "CloseDate", "ContractStatusChangeDate"]

os.makedirs(summary_dir, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
 
 
def load(path):
    df = pd.read_csv(path, low_memory=False)
    # CSVs don't store data types, so the date columns come back as text
    for col in date_fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    print(f"Loaded {path}: {len(df):,} rows, {df.shape[1]} columns")
    return df


# Divide safely: a missing or zero/negative denominator gives NaN instead of
# infinity or a nonsense ratio. (Week 4-5 flagged these rows; they stay in the data.)
def safe_divide(numerator, denominator):
    return numerator / denominator.where(denominator > 0)

 
# ---------------------------------------------------------------------------
# 1. Engineered metrics (sold)
# ---------------------------------------------------------------------------
def add_sold_metrics(sold):
    print("\n--- 1. Engineered metrics ---")
 
    # Price ratios: above 1.0 = sold over asking (seller's market), below 1.0 = under
    sold["price_ratio"] = safe_divide(sold["ClosePrice"], sold["ListPrice"]).round(4)
    sold["close_to_original_list_ratio"] = safe_divide(sold["ClosePrice"], sold["OriginalListPrice"]).round(4)
 
    # Price per square foot normalizes price across home sizes
    sold["price_per_sqft"] = safe_divide(sold["ClosePrice"], sold["LivingArea"]).round(2)
 
    # Time-series fields from CloseDate (yr_mo is text like "2024-03" so Tableau
    # treats it as one label per month)
    sold["close_year"] = sold["CloseDate"].dt.year
    sold["close_month"] = sold["CloseDate"].dt.month
    sold["yr_mo"] = sold["CloseDate"].dt.strftime("%Y-%m")
 
    # Timeline durations in days. Missing dates give NaN. Negative values come from
    # the date-order problems flagged in Weeks 4-5 (negative_timeline_flag,
    # purchase_after_close_flag) and are left in place for Week 7 to handle.
    sold["listing_to_contract_days"] = (sold["PurchaseContractDate"] - sold["ListingContractDate"]).dt.days
    sold["contract_to_close_days"] = (sold["CloseDate"] - sold["PurchaseContractDate"]).dt.days
 
    new_cols = ["price_ratio", "close_to_original_list_ratio", "price_per_sqft", "DaysOnMarket",
                "close_year", "close_month", "yr_mo", "listing_to_contract_days", "contract_to_close_days"]
 
    print("\nHow many rows have each new value populated:")
    print(sold[new_cols].notna().sum().to_string())
    print("\nNegative durations (date-order problems flagged in Weeks 4-5):")
    print(f"listing_to_contract_days < 0: {(sold['listing_to_contract_days'] < 0).sum():,}")
    print(f"contract_to_close_days < 0:   {(sold['contract_to_close_days'] < 0).sum():,}")
 
    print("\nDistribution of the new numeric metrics:")
    numeric_new = ["price_ratio", "close_to_original_list_ratio", "price_per_sqft", "DaysOnMarket",
                   "listing_to_contract_days", "contract_to_close_days"]
    print(sold[numeric_new].describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_string())
 
    # Sample output table for the deliverable: inputs next to the metrics they produce
    sample_cols = ["ListingKey", "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
                   "ListingContractDate", "PurchaseContractDate", "CloseDate"] + new_cols
    sample = sold[[c for c in sample_cols if c in sold.columns]].dropna(
        subset=["price_ratio", "price_per_sqft", "listing_to_contract_days"]).head(10)
    print("\nSample output table (new columns populated):")
    print(sample.to_string(index=False))
    sample.to_csv(f"{summary_dir}/sample_engineered_metrics.csv", index=False)
    return sold
 
 
# ---------------------------------------------------------------------------
# 2. School districts from Latitude/Longitude (spatial join)
# ---------------------------------------------------------------------------
def add_school_districts(df, name):
    print(f"\n--- 2. School districts ({name}) ---")
    if not os.path.exists(school_districts):
        print(f"SKIPPED: school district shapefile not found. Put DistrictAreas2425.shp "
              f"(with its .shx, .dbf, .prj, .cpg files) in csv_files/school_districts/")
        return df
 
    import geopandas as gpd  # pip install geopandas
 
    districts = gpd.read_file(school_districts).to_crs("EPSG:4326")
    # Shapefiles cut column names to 10 characters, so find the name column instead
    # of guessing its exact spelling. The full list is printed so you can check it.
    print(f"District boundary columns: {list(districts.columns)}")
    # Prefer a column like "DistrictName"/"DistrictNa" over others like "CountyName"
    name_col = next((c for c in districts.columns if "district" in c.lower() and "na" in c.lower()), None) \
        or next((c for c in districts.columns if "name" in c.lower()), None)
    if name_col is None:
        print("SKIPPED: no district name column found. Set name_col by hand from the list above.")
        return df
    print(f"Using '{name_col}' as the district name ({len(districts):,} districts)")
 
    # Only use coordinates that passed the Week 4-5 geographic checks
    usable = df["Latitude"].notna() & df["Longitude"].notna()
    if "geo_invalid_flag" in df.columns:
        usable &= ~df["geo_invalid_flag"].astype(bool)
    points = gpd.GeoDataFrame(
        index=df.index[usable],
        geometry=gpd.points_from_xy(df.loc[usable, "Longitude"], df.loc[usable, "Latitude"]),
        crs="EPSG:4326",
    )
 
    # A home can sit inside two districts: an elementary district AND a high school
    # district that overlap. Keep every match, joined like "Goleta Union / Santa Barbara Unified".
    joined = gpd.sjoin(points, districts[[name_col, "geometry"]], how="left", predicate="within")
    matches = joined[name_col].dropna().groupby(level=0)
    df["school_district"] = matches.apply(lambda s: " / ".join(sorted(s.unique())))
    df["school_district_count"] = matches.nunique()
    df["school_district"] = df["school_district"].fillna("Unknown")
    df["school_district_count"] = df["school_district_count"].fillna(0).astype(int)
 
    matched = (df["school_district"] != "Unknown").sum()
    print(f"Rows with usable coordinates: {usable.sum():,}")
    print(f"Rows matched to a school district: {matched:,} ({matched / len(df):.1%})")
    print(f"Rows inside two or more overlapping districts: {(df['school_district_count'] > 1).sum():,}")
    print("\nTop 10 school districts by row count:")
    print(df["school_district"].value_counts().head(10).to_string())
    return df
 
 
# ---------------------------------------------------------------------------
# 3. Segment analysis
# ---------------------------------------------------------------------------
def segment_summary(sold, group_cols, sort_by="sales", top=None):
    summary = (
        sold.groupby(group_cols, dropna=False)
        .agg(
            sales=("ClosePrice", "size"),
            sales_volume=("ClosePrice", "sum"),
            median_close_price=("ClosePrice", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_price_ratio=("price_ratio", "median"),
            median_close_to_original=("close_to_original_list_ratio", "median"),
            avg_days_on_market=("DaysOnMarket", "mean"),
            median_days_on_market=("DaysOnMarket", "median"),
            median_listing_to_contract=("listing_to_contract_days", "median"),
            median_contract_to_close=("contract_to_close_days", "median"),
        )
        .sort_values(sort_by, ascending=False)
        .round(3)
    )
    return summary.head(top) if top else summary
 
 
def run_segments(sold):
    print("\n--- 3. Segment analysis ---")
    # Medians are used for prices and ratios because a few extreme sales would pull
    # an average around. Week 7 handles the outliers themselves.
    segments = {
        "property_type_subtype": (["PropertyType", "PropertySubType"], "sales", None),
        "county": (["CountyOrParish"], "sales", None),
        "mls_area_top25": (["MLSAreaMajor"], "sales", 25),
        "list_office_top25": (["ListOfficeName"], "sales_volume", 25),
        "buyer_office_top25": (["BuyerOfficeName"], "sales_volume", 25),
    }
    if "school_district" in sold.columns:
        segments["school_district_top25"] = (["school_district"], "sales", 25)
 
    for name, (cols, sort_by, top) in segments.items():
        cols = [c for c in cols if c in sold.columns]
        if not cols:
            print(f"Skipping {name}: column missing")
            continue
        table = segment_summary(sold, cols, sort_by, top)
        table.to_csv(f"{summary_dir}/segment_{name}.csv")
        print(f"\nSegment summary: {name} (sorted by {sort_by})")
        print(table.head(15).to_string())
 
 
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70 + "\nSOLD\n" + "=" * 70)
    sold = load(sold_input)
    sold = add_sold_metrics(sold)
    sold = add_school_districts(sold, "sold")
    run_segments(sold)
    sold.to_csv("csv_files/sold_engineered.csv", index=False)
    print(f"\nSaved csv_files/sold_engineered.csv ({len(sold):,} rows, {sold.shape[1]} columns)")
 
    # Listings: the price and timeline metrics need a sale, so listings only get the
    # listing month (for "new listings" counts in Tableau) and the school district.
    print("\n" + "=" * 70 + "\nLISTINGS\n" + "=" * 70)
    listings = load(listings_input)
    listings["list_year"] = listings["ListingContractDate"].dt.year
    listings["list_month"] = listings["ListingContractDate"].dt.month
    listings["list_yr_mo"] = listings["ListingContractDate"].dt.strftime("%Y-%m")
    print("\nNew listings per month (first 12):")
    print(listings["list_yr_mo"].value_counts().sort_index().head(12).to_string())
    listings = add_school_districts(listings, "listings")
    listings.to_csv("csv_files/listings_engineered.csv", index=False)
    print(f"\nSaved csv_files/listings_engineered.csv ({len(listings):,} rows, {listings.shape[1]} columns)")
 
    print(f"\nDone. Segment tables are in {summary_dir}/")
 