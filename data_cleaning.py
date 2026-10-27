# Data Cleaning and Preparation (Weeks 4-5)

# Cleans BOTH enriched datasets from the mortgage rate step:
#   csv_files/sold_with_mortgage_rates.csv      -> csv_files/sold_cleaned.csv
#   csv_files/listings_with_mortgage_rates.csv  -> csv_files/listings_cleaned.csv

# The same steps run on both. The differences for listings:
#   - Listings include Active, Expired and Withdrawn records, which never have a
#     CloseDate or ClosePrice, so those rows are NOT removed. Instead a listing must
#     have a ListingContractDate and a ListPrice (needed for "new listings" counts).
#   - The "always invalid" price rule uses ListPrice for listings and ClosePrice for sold.
#   - Duplicate listing IDs keep the latest status change, since a listing can show up
#     in more than one monthly file as its status changes.
 
import pandas as pd
 
date_fields = ["ListingContractDate", "PurchaseContractDate", "CloseDate", "ContractStatusChangeDate"]
 
numeric_fields = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres", "BedroomsTotal",
                  "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt", "Latitude", "Longitude", "rate_30yr_fixed"]
 
# for grouping text fields for Week 6
category_fields = ["PropertyType", "PropertySubType", "CountyOrParish", "City", "MLSAreaMajor",
                   "ListOfficeName", "BuyerOfficeName"]
 
# Never drop these, even if they are mostly empty or have only one value
core_fields = set(date_fields + numeric_fields + category_fields + ["year_month"])
 
# California bounding box (smallest possible)
CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.1
CA_LON_MIN, CA_LON_MAX = -124.5, -114.1
 
# Settings that differ between the two datasets
DATASETS = {
    "sold": {
        "input": "csv_files/sold_with_mortgage_rates.csv",
        "required": ["CloseDate", "ClosePrice"],   # can't analyze a sale without these
        "price_col": "ClosePrice",                 # price <= 0 is never valid
        "dedupe_sort": "CloseDate",                # keep the latest close per listing ID
    },
    "listings": {
        "input": "csv_files/listings_with_mortgage_rates.csv",
        "required": ["ListingContractDate", "ListPrice"],  # can't count a new listing without these
        "price_col": "ListPrice",
        "dedupe_sort": "ContractStatusChangeDate",         # keep the latest status per listing ID
    },
}
 
 
# only use columns that exist in the dataset.
def present(df, cols):
    return [c for c in cols if c in df.columns]
 
 
def clean(name, settings):
    print("\n" + "=" * 70)
    print(f"CLEANING {name.upper()}  ({settings['input']})")
    print("=" * 70)
 
    df = pd.read_csv(settings["input"], low_memory=False)
    price_col = settings["price_col"]
    cleaning_log = []
 
    # Record row/column count after a step so before/after counts are documented
    def log_step(step, df, why):
        cleaning_log.append({"step": step, "rows": len(df), "columns": df.shape[1], "why": why})
        print(f"[{step}] rows: {len(df):,} | columns: {df.shape[1]}  -- {why}")
 
    rows_start = len(df)
    log_step("0. Loaded", df, f"starting point (Residential {name} + mortgage rate)")
 
    # 1. Convert date fields to datetime
    print("\n--- 1. Date conversion ---")
    for col in present(df, date_fields):
        before_missing = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        newly_missing = df[col].isna().sum() - before_missing
        print(f"{col}: converted | unparseable values turned to NaT: {newly_missing}")
 
    # 2. Ensure numeric fields are numeric
    print("\n--- 2. Numeric type conversion ---")
    for col in present(df, numeric_fields):
        before_missing = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        newly_missing = df[col].isna().sum() - before_missing
        if newly_missing:
            print(f"{col}: {newly_missing} non-numeric values turned to NaN")
 
    # 3. Remove unnecessary columns (>90% empty or a single value)
    print("\n--- 3. Drop unnecessary columns ---")
    missing_pct = df.isna().mean() * 100
    high_missing = [c for c in df.columns if missing_pct[c] > 90 and c not in core_fields]
    single_value = [c for c in df.columns
                    if df[c].nunique(dropna=True) <= 1 and c not in core_fields and c not in high_missing]
    print(f"Dropping {len(high_missing)} columns >90% missing: {high_missing}")
    print(f"Dropping {len(single_value)} columns with a single value: {single_value}")
    df = df.drop(columns=high_missing + single_value)
    log_step("3. Dropped columns", df, ">90% missing or single-value columns removed")
 
    # 4. Remove duplicate records
    print("\n--- 4. Duplicates ---")
    df = df.drop_duplicates()
    log_step("4a. Exact duplicates removed", df, "identical rows")
 
    id_col = next((c for c in ["ListingKey", "ListingKeyNumeric", "ListingId"] if c in df.columns), None)
    sort_col = settings["dedupe_sort"] if settings["dedupe_sort"] in df.columns else None
    if id_col and sort_col:
        # Keep the most recent record for each listing ID
        df = df.sort_values(sort_col).drop_duplicates(subset=id_col, keep="last")
        log_step(f"4b. Duplicate {id_col} removed", df, f"same listing ID; kept latest {sort_col}")
 
    # 5. Handle missing values
    print("\n--- 5. Missing values ---")
    required = present(df, settings["required"])
    df = df.dropna(subset=required)
    log_step(f"5a. Missing {'/'.join(required)} removed", df, "required for this dataset's analysis")
 
    for col in present(df, category_fields):
        n = df[col].isna().sum()
        df[col] = df[col].astype("string").str.strip().fillna("Unknown")
        if n:
            print(f"{col}: {n:,} missing filled with 'Unknown'")
 
    print("\nRemaining missing values in key numeric fields (left as NaN on purpose):")
    print(df[present(df, numeric_fields)].isna().sum().to_string())
 
    # 6. Invalid numeric values
    print("\n--- 6. Invalid numeric values ---")
    invalid_price = (df[price_col] <= 0).sum()
    df = df[df[price_col] > 0]
    print(f"{price_col} <= 0 removed: {invalid_price}")
    log_step(f"6. {price_col} <= 0 removed", df, "a price of zero or less is never valid")
 
    # Listings: some rows are sold and have a ClosePrice. Flag (don't remove) a bad one.
    if name == "listings" and "ClosePrice" in df.columns:
        df["invalid_close_price_flag"] = df["ClosePrice"] <= 0
        print(f"invalid_close_price_flag: {df['invalid_close_price_flag'].sum():,}")
 
    df["invalid_living_area_flag"] = df["LivingArea"] <= 0 if "LivingArea" in df else False
    df["negative_dom_flag"] = df["DaysOnMarket"] < 0 if "DaysOnMarket" in df else False
    df["negative_bed_bath_flag"] = False
    for col in present(df, ["BedroomsTotal", "BathroomsTotalInteger"]):
        df["negative_bed_bath_flag"] |= df[col] < 0
    df["invalid_numeric_flag"] = (
        df["invalid_living_area_flag"] | df["negative_dom_flag"] | df["negative_bed_bath_flag"]
    )
    for flag in ["invalid_living_area_flag", "negative_dom_flag", "negative_bed_bath_flag", "invalid_numeric_flag"]:
        print(f"{flag}: {df[flag].sum():,}")
 
    # 7. Date consistency checks
    # Comparisons with a missing date are False, so listings that never went under
    # contract or closed are never flagged; only rows with both dates are checked.
    print("\n--- 7. Date consistency flags ---")
    df["listing_after_close_flag"] = df["ListingContractDate"] > df["CloseDate"]
    df["purchase_after_close_flag"] = df["PurchaseContractDate"] > df["CloseDate"]
    df["negative_timeline_flag"] = df["ListingContractDate"] > df["PurchaseContractDate"]
    df["date_issue_flag"] = (
        df["listing_after_close_flag"] | df["purchase_after_close_flag"] | df["negative_timeline_flag"]
    )
    for flag in ["listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag", "date_issue_flag"]:
        print(f"{flag}: {df[flag].sum():,}")
 
    # 8. Geographic data checks
    print("\n--- 8. Geographic flags ---")
    lat, lon = df["Latitude"], df["Longitude"]
    df["missing_coords_flag"] = lat.isna() | lon.isna()
    df["zero_coords_flag"] = (lat == 0) | (lon == 0)
    df["positive_longitude_flag"] = lon > 0
    has_real_coords = ~df["missing_coords_flag"] & ~df["zero_coords_flag"]
    df["out_of_state_flag"] = has_real_coords & ~(
        lat.between(CA_LAT_MIN, CA_LAT_MAX) & lon.between(CA_LON_MIN, CA_LON_MAX)
    )
 
    # state or province check
    if "StateOrProvince" in df.columns:
        print("StateOrProvince values:")
        print(df["StateOrProvince"].value_counts(dropna=False).to_string())
        state = df["StateOrProvince"].astype("string").str.strip().str.upper()
        df["non_ca_state_flag"] = state.notna() & (state != "CA")
    else:
        df["non_ca_state_flag"] = False
 
    df["geo_invalid_flag"] = (
        df["missing_coords_flag"] | df["zero_coords_flag"]
        | df["positive_longitude_flag"] | df["out_of_state_flag"]
        | df["non_ca_state_flag"]
    )
 
    geo_flags = ["missing_coords_flag", "zero_coords_flag", "positive_longitude_flag",
                 "out_of_state_flag", "non_ca_state_flag", "geo_invalid_flag"]
    geo_summary = pd.DataFrame({
        "records_flagged": [int(df[f].sum()) for f in geo_flags],
        "pct_of_records": [round(df[f].mean() * 100, 2) for f in geo_flags],
    }, index=geo_flags)
    print("\nGeographic data quality summary:")
    print(geo_summary.to_string())
 
    if df["out_of_state_flag"].any():
        print("\nSample out-of-state / implausible coordinates:")
        print(df.loc[df["out_of_state_flag"],
                     present(df, ["City", "CountyOrParish", "StateOrProvince", "Latitude", "Longitude"])].head(10).to_string())
 
    if df["non_ca_state_flag"].any():
        print("\nSample records with StateOrProvince other than CA:")
        print(df.loc[df["non_ca_state_flag"],
                     present(df, ["City", "StateOrProvince", "Latitude", "Longitude"])].head(10).to_string())
 
    # Final confirmation and save to csv files
    print(f"\n=== FINAL SUMMARY: {name.upper()} ===")
    print(f"Rows before cleaning: {rows_start:,}")
    print(f"Rows after cleaning:  {len(df):,}  (removed {rows_start - len(df):,})")
 
    print("\nData type confirmation:")
    print(df[present(df, date_fields + numeric_fields)].dtypes.to_string())
    for col in present(df, date_fields):
        assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} is not datetime"
    for col in present(df, numeric_fields):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"
    assert (df[price_col] > 0).all()
    print(f"PASSED: all date fields are datetime, all numeric fields are numeric, no {price_col} <= 0")
 
    cleaned_file = f"csv_files/{name}_cleaned.csv"
    geo_file = f"csv_files/{name}_geo_quality_summary.csv"
    log_file = f"csv_files/{name}_cleaning_log.csv"
    df.to_csv(cleaned_file, index=False)
    geo_summary.to_csv(geo_file, index_label="flag")
    pd.DataFrame(cleaning_log).to_csv(log_file, index=False)
    print(f"\nSaved: {cleaned_file}, {geo_file}, {log_file}")
    return df
 
 
if __name__ == "__main__":
    for name, settings in DATASETS.items():
        clean(name, settings)
    print("\nDone. Both datasets cleaned.")