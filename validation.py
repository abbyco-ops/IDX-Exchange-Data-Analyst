import pandas as pd
'''
sold = pd.read_csv("concatenated_residential_sold.csv")
listings = pd.read_csv("concatenated_residential_listings.csv")

print("Sold shape:", sold.shape)
print("Listings shape:", listings.shape)

'''

print("🚀 Loading data...")
sold = pd.read_csv("concatenated_listings_all.csv", low_memory=False)
listings = pd.read_csv("concatenated_sold_all.csv", low_memory=False)

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

    print(f"\nColumns with >90% missing values:")
    print(report[report["missing_percent"] > 90])

    high_missing = report[report["missing_percent"] > 90]

    print(f"\nColumns with >90% missing values ({len(high_missing)}):")
    print(high_missing)

    return report

sold_missing = missing_value_report(sold, "Sold")
listings_missing = missing_value_report(listings, "Listings")

sold_missing.to_csv("sold_missing_value_report.csv")
listings_missing.to_csv("listings_missing_value_report.csv")

### Distribution Analysis

def dist_summary(df, columns, name):
    print(f"\n--- Distribution Summary: {name} ---")
    summary = df[columns].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).T

    print(summary)

    return summary

key_numeric_fields = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket"
]

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
import matplotlib.pyplot as plt

for column in key_numeric_fields:

    plt.figure(figsize=(8, 5))
    plt.hist(sold[column].dropna(), bins=50)
    plt.title(f"Sold - {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.boxplot(sold[column].dropna())
    plt.title(f"Sold - {column} Boxplot")
    plt.ylabel(column)
    plt.show()

for column in key_numeric_fields:
    plt.figure(figsize=(8, 5))
    plt.hist(listings[column].dropna(), bins=50)
    plt.title(f"Listings - {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(listings[column].dropna())
    plt.title(f"Listings - {column} Boxplot")
    plt.ylabel(column)
    plt.show()