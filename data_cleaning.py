# Data Cleaning and Preparation

import pandas as pd

print("Loading data...")

sold = pd.read_csv("csv_files/sold_with_mortgage_rates.csv", low_memory = False)
cleaned_data = "csv_files/sold_cleaned.csv"
geo_summary_file = "csv_files/sold_geo_quality_summary.csv"
log_file = "csv_files/sold_cleaning_log.csv"

date_fields = ["ListingContractDate", "PurchaseContractDate", "CloseDate", "ContractStatusChangeDate"]

numeric_fields = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres", "BedroomsTotal", 
                  "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt", "Latitude", "Longitude", "rate_30yr_fixed"]

# for grouping text fields for Week 6
category_fields = [ "PropertyType", "PropertySubType", "CountyOrParish", "City", "MLSAreaMajor", 
                   "ListOfficeName", "BuyerOfficeName"]

# Never drop these, even if they are mostly empty or have only one value
core_fields = set(date_fields + numeric_fields + category_fields + ["year_month"])

#print(sold["StateOrProvince"].value_counts(dropna=False))

# California bounding box (smallest possible)
CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.1
CA_LON_MIN, CA_LON_MAX = -124.5, -114.1

cleaning_log = []

# Record row/column count after a step so before/after counts are documented
def log_step(step, df, why):
    cleaning_log.append({"step": step, "rows": len(df), "columns": df.shape[1], "why": why})
    print(f"[{step}] rows: {len(df):,} | columns: {df.shape[1]}  -- {why}")
 
 # only use columns that exist in the dataset.
def present(df, cols):
    return [c for c in cols if c in df.columns]

rows_start = len(sold)
log_step("0. Loaded", sold, "starting point (Residential sold + mortgage rate)")

# Convert date fields to datetime
print("\n--- 1. Date conversion ---")
for col in present(sold, date_fields):
    before_missing = sold[col].isna().sum()
    sold[col] = pd.to_datetime(sold[col], errors="coerce")
    newly_missing = sold[col].isna().sum() - before_missing
    print(f"{col}: converted | unparseable values turned to NaT: {newly_missing}")

# ensure numeric fields are numeric
print("\n--- 2. Numeric type conversion ---")
for col in present(sold, numeric_fields):
    before_missing = sold[col].isna().sum()
    sold[col] = pd.to_numeric(sold[col], errors="coerce")
    newly_missing = sold[col].isna().sum() - before_missing
    if newly_missing:
        print(f"{col}: {newly_missing} non-numeric values turned to NaN")
 
 # remove unnecessary columns (>90% empty columns)
print("\n--- 3. Drop unnecessary columns ---")
missing_pct = sold.isna().mean() * 100
high_missing = [c for c in sold.columns if missing_pct[c] > 90 and c not in core_fields]
single_value = [c for c in sold.columns
                if sold[c].nunique(dropna=True) <= 1 and c not in core_fields and c not in high_missing]
print(f"Dropping {len(high_missing)} columns >90% missing: {high_missing}")
print(f"Dropping {len(single_value)} columns with a single value: {single_value}")
sold = sold.drop(columns=high_missing + single_value)
log_step("3. Dropped columns", sold, ">90% missing or single-value columns removed")

# remove all duplicate records
print("\n--- 4. Duplicates ---")
sold = sold.drop_duplicates()
log_step("4a. Exact duplicates removed", sold, "identical rows")
 
id_col = next((c for c in ["ListingKey", "ListingKeyNumeric", "ListingId"] if c in sold.columns), None)
if id_col:
    # Keep the most recent record for each listing ID
    sold = (sold.sort_values("CloseDate")
                .drop_duplicates(subset=id_col, keep="last"))
    log_step(f"4b. Duplicate {id_col} removed", sold, "same listing ID; kept latest CloseDate")

# handle missing values
print("\n--- 5. Missing values ---")
sold = sold.dropna(subset=present(sold, ["CloseDate", "ClosePrice"]))
log_step("5a. Missing CloseDate/ClosePrice removed", sold, "can't analyze a sale without these")
 
for col in present(sold, category_fields):
    n = sold[col].isna().sum()
    sold[col] = sold[col].astype("string").str.strip().fillna("Unknown")
    if n:
        print(f"{col}: {n:,} missing filled with 'Unknown'")
 
print("\nRemaining missing values in key numeric fields (left as NaN on purpose):")
print(sold[present(sold, numeric_fields)].isna().sum().to_string())

# Invalid numeric fields
print("\n--- 6. Invalid numeric values ---")
invalid_price = (sold["ClosePrice"] <= 0).sum()
sold = sold[sold["ClosePrice"] > 0]
print(f"ClosePrice <= 0 removed: {invalid_price}")
log_step("6. ClosePrice <= 0 removed", sold, "a sale price of zero or less is never valid")
 
sold["invalid_living_area_flag"] = sold["LivingArea"] <= 0 if "LivingArea" in sold else False
sold["negative_dom_flag"] = sold["DaysOnMarket"] < 0 if "DaysOnMarket" in sold else False
sold["negative_bed_bath_flag"] = False
for col in present(sold, ["BedroomsTotal", "BathroomsTotalInteger"]):
    sold["negative_bed_bath_flag"] |= sold[col] < 0
sold["invalid_numeric_flag"] = (
    sold["invalid_living_area_flag"] | sold["negative_dom_flag"] | sold["negative_bed_bath_flag"]
)
for flag in ["invalid_living_area_flag", "negative_dom_flag", "negative_bed_bath_flag", "invalid_numeric_flag"]:
    print(f"{flag}: {sold[flag].sum():,}")


# Date consistency checks
print("\n--- 7. Date consistency flags ---")
sold["listing_after_close_flag"] = sold["ListingContractDate"] > sold["CloseDate"]
sold["purchase_after_close_flag"] = sold["PurchaseContractDate"] > sold["CloseDate"]
sold["negative_timeline_flag"] = sold["ListingContractDate"] > sold["PurchaseContractDate"]
sold["date_issue_flag"] = (
    sold["listing_after_close_flag"] | sold["purchase_after_close_flag"] | sold["negative_timeline_flag"]
)
for flag in ["listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag", "date_issue_flag"]:
    print(f"{flag}: {sold[flag].sum():,}")

# geographic data checks
print("\n--- 8. Geographic flags ---")
lat, lon = sold["Latitude"], sold["Longitude"]
sold["missing_coords_flag"] = lat.isna() | lon.isna()
sold["zero_coords_flag"] = (lat == 0) | (lon == 0)
sold["positive_longitude_flag"] = lon > 0
has_real_coords = ~sold["missing_coords_flag"] & ~sold["zero_coords_flag"]
sold["out_of_state_flag"] = has_real_coords & ~(
    lat.between(CA_LAT_MIN, CA_LAT_MAX) & lon.between(CA_LON_MIN, CA_LON_MAX)
)

# state or province check
if "StateOrProvince" in sold.columns:
    print("StateOrProvince values:")
    print(sold["StateOrProvince"].value_counts(dropna=False).to_string())
    state = sold["StateOrProvince"].astype("string").str.strip().str.upper()
    sold["non_ca_state_flag"] = state.notna() & (state != "CA")
else:
    sold["non_ca_state_flag"] = False
 
sold["geo_invalid_flag"] = (
    sold["missing_coords_flag"] | sold["zero_coords_flag"]
    | sold["positive_longitude_flag"] | sold["out_of_state_flag"]
    | sold["non_ca_state_flag"]
)

geo_flags = ["missing_coords_flag", "zero_coords_flag", "positive_longitude_flag",
             "out_of_state_flag", "non_ca_state_flag", "geo_invalid_flag"]
geo_summary = pd.DataFrame({
    "records_flagged": [int(sold[f].sum()) for f in geo_flags],
    "pct_of_records": [round(sold[f].mean() * 100, 2) for f in geo_flags],
}, index=geo_flags)
print("\nGeographic data quality summary:")
print(geo_summary.to_string())
 
if sold["out_of_state_flag"].any():
    print("\nSample out-of-state / implausible coordinates:")
    print(sold.loc[sold["out_of_state_flag"],
                   present(sold, ["City", "CountyOrParish", "StateOrProvince", "Latitude", "Longitude"])].head(10).to_string())
 
if sold["non_ca_state_flag"].any():
    print("\nSample records with StateOrProvince other than CA:")
    print(sold.loc[sold["non_ca_state_flag"],
                   present(sold, ["City", "StateOrProvince", "Latitude", "Longitude"])].head(10).to_string())

 # final confirmation and save to csv files 
print("\n=== FINAL SUMMARY ===")
print(f"Rows before cleaning: {rows_start:,}")
print(f"Rows after cleaning:  {len(sold):,}  (removed {rows_start - len(sold):,})")
 
print("\nData type confirmation:")
print(sold[present(sold, date_fields + numeric_fields)].dtypes.to_string())
for col in present(sold, date_fields):
    assert pd.api.types.is_datetime64_any_dtype(sold[col]), f"{col} is not datetime"
for col in present(sold, numeric_fields):
    assert pd.api.types.is_numeric_dtype(sold[col]), f"{col} is not numeric"
assert (sold["ClosePrice"] > 0).all()
print("PASSED: all date fields are datetime, all numeric fields are numeric, no ClosePrice <= 0")
 
sold.to_csv(cleaned_data, index=False)
geo_summary.to_csv(geo_summary_file, index_label="flag")
pd.DataFrame(cleaning_log).to_csv(log_file, index=False)
print(f"\nSaved: {cleaned_data}, {geo_summary_file}, {log_file}")
