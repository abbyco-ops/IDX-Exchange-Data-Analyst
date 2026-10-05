import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os                                   # add near the other imports
os.makedirs("plots", exist_ok=True)         # add after the imports

print("Yoo gimme a sec, loading data...")
sold = pd.read_csv("concatenated_sold_all.csv", low_memory=False)
listings = pd.read_csv("concatenated_listings_all.csv", low_memory=False)

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# Inspect data
'''print(list(sold.columns))
print(sold.head)'''

key_numeric_fields = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres", "BedroomsTotal",
                      "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt"]

### Dataset Understanding 

print(f"Sold shape: {sold.shape}")
print(f"Listings shape: {listings.shape}")

print("\n--- SOLD DATASET ---")
print("Rows:", len(sold))
print("Columns:", len(sold.columns))

print("\nData types:")
print(sold.dtypes)

print("\n--- LISTINGS DATASET ---")
print("Rows:", len(listings))
print("Columns:", len(listings.columns))

print("\nData types:")
print(listings.dtypes)

### Property Types

print("\n--- Sold Property Types ---")
print(sold["PropertyType"].value_counts(dropna=False))

print("\n--- Listings Property Types ---")
print(listings["PropertyType"].value_counts(dropna=False))

print(f"Sold Residential share: {(sold['PropertyType'] == 'Residential').mean() * 100:.2f}%")
print(f"Listings Residential share: {(listings['PropertyType'] == 'Residential').mean() * 100:.2f}%")

# Keep only PropertyType == 'Residential'
# Filter Residential
sold = sold[sold["PropertyType"] == "Residential"].copy()
listings = listings[listings["PropertyType"] == "Residential"].copy()

print("\nAfter Residential filter:")
print("Sold:", sold.shape)
print("Listings:", listings.shape)

### Missing Value Analysis
def missing_value_report(df, name):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })

    report = report.sort_values("missing_percent", ascending=False)

    print(f"\n--- Missing Value Report: {name} ---")
    print(report)

    high_missing = report[report["missing_percent"] > 90]

    print(f"\nColumns with >90% missing values ({len(high_missing)}):")
    print(high_missing)

    return report

sold_missing = missing_value_report(sold, "Sold")
listings_missing = missing_value_report(listings, "Listings")

sold_missing.to_csv("sold_missing_value_report.csv")
listings_missing.to_csv("listings_missing_value_report.csv")

sold_drop = sold_missing[sold_missing["missing_percent"] > 90].index.difference(key_numeric_fields)
listings_drop = listings_missing[listings_missing["missing_percent"] > 90].index.difference(key_numeric_fields)

print("Dropping from sold:", list(sold_drop))
print("Dropping from listings:", list(listings_drop))

sold = sold.drop(columns=sold_drop)
listings = listings.drop(columns=listings_drop)

print("Sold shape after dropping:", sold.shape)
print("Listings shape after dropping:", listings.shape)

# Check Sold shape after dropping columns with >90% missing values
print(sold.shape)

# Kept Fields based on Missing Value Analysis

sold.to_csv("sold_residential_filtered.csv", index=False)
listings.to_csv("listings_residential_filtered.csv", index=False)

### Numeric Distribution Analysis

def dist_summary(df, columns, name):
    print(f"\n--- Distribution Summary: {name} ---")
    summary = df[columns].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).T

    print(summary)

    return summary

deliverable_numeric_fields = ["ClosePrice", "LivingArea", "DaysOnMarket"]

# Some fields were extremely right-skewed, resulting in the a squashed flat box at the bottom
# hence need to use LOG scale
log_scale_fields = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres"]


sold_distribution_summary = dist_summary(
    sold,
    key_numeric_fields,
    "Sold"
)

listings_distribution_summary = dist_summary(
    listings,
    key_numeric_fields,
    "Listings"
)

sold_distribution_summary.to_csv("sold_distribution_summary.csv")
listings_distribution_summary.to_csv("listings_distribution_summary.csv")

### Generate Histograms and Boxplots

# Sold Dataset
for name, df in [("sold", sold), ("listings", listings)]:
    for column in key_numeric_fields:
        s = df[column].dropna()
        low, high = s.quantile([0.01, 0.99])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(s[(s >= low) & (s <= high)], bins=50)   # trim extremes so the shape is visible
        ax1.set_title(f"{name} - {column} (1st-99th percentile)")
        if column in log_scale_fields:
            ax2.boxplot(s[s > 0]); ax2.set_yscale("log")
        else:
            ax2.boxplot(s)
        ax2.set_title(f"{name} - {column} boxplot")
        fig.savefig(f"plots/{name}_{column}.png")
        plt.close(fig)


# Identify Outliers on key numeric fields -- based on IQR rule
# IQR = Q3 - Q1.   Mild fences = Q1 - 1.5*IQR / Q3 + 1.5*IQR.   Extreme fences use 3*IQR.
def outlier_summary(df, fields):
    rows = {}
    for col in fields:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        xlo, xhi = q1 - 3 * iqr, q3 + 3 * iqr
        rows[col] = {
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_fence": lo, "upper_fence": hi,
            "below_lower": int((s < lo).sum()), "above_upper": int((s > hi).sum()),
            "pct_outliers": round(float(((s < lo) | (s > hi)).mean() * 100), 2),
            "extreme_outliers_3xIQR": int(((s < xlo) | (s > xhi)).sum()),
            "values_le_0": int((s <= 0).sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")

# Mean vs median close price (mean > median means a few very expensive homes pull it up)

print(f"\n--- Mean vs. Median Close Prices ---")
cp = sold.loc[sold["ClosePrice"] > 0, "ClosePrice"]
print("Median:", cp.median(), "Mean:", cp.mean())

# Days on market buckets 
print(f"\n--- Days on market Analysis ---")
dom = sold["DaysOnMarket"].dropna()
buckets = pd.cut(dom, bins=[-float("inf"), 0, 7, 30, 60, float("inf")],
                 labels=["0 or negative", "1-7", "8-30", "31-60", "61+"])
print(buckets.value_counts().sort_index())

# Sold above vs below list price
print(f"\n--- Sold above vs below list price? ---")
v = sold[(sold["ClosePrice"] > 0) & (sold["ListPrice"] > 0)]
print("Above list %:", (v["ClosePrice"] > v["ListPrice"]).mean() * 100)
print("Below list %:", (v["ClosePrice"] < v["ListPrice"]).mean() * 100)

# Date problems (just count them for now)
print(f"\n--- Date problems (count) ---")
for c in ["ListingContractDate", "PurchaseContractDate", "CloseDate"]:
    sold[c] = pd.to_datetime(sold[c], errors="coerce")
print("Listed after close:", (sold["ListingContractDate"] > sold["CloseDate"]).sum())
print("Purchase after close:", (sold["PurchaseContractDate"] > sold["CloseDate"]).sum())
print("Listed after purchase:", (sold["ListingContractDate"] > sold["PurchaseContractDate"]).sum())

# Counties with the highest median price (CAUTION: ignore tiny counties)
print(f"\n--- Counties with highest median price ---")
county = sold[sold["ClosePrice"] > 0].groupby("CountyOrParish")["ClosePrice"].agg(["median", "count"])
print(county[county["count"] >= 30].sort_values("median", ascending=False).head(10))

print(f"\n--- Sold Outlier Summary ---")
sold_outliers = outlier_summary(sold, key_numeric_fields)
listings_outliers = outlier_summary(listings, key_numeric_fields)
print(sold_outliers)

# csv's
sold_outliers.to_csv("sold_outlier_summary.csv")
listings_outliers.to_csv("listings_outlier_summary.csv")

sold_distribution_summary.loc[deliverable_numeric_fields].to_csv("sold_distribution_summary_required3.csv")