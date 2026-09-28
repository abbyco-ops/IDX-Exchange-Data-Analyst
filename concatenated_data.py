import pandas as pd
import os
import glob
import re
from datetime import datetime

def get_most_recent_completed_month():
    # Returns (year, month) of the most recently completed calendar month.
    today = datetime.today()
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1

def process_datasets():
    # 1. Define date bounds: January 2024 through most recently completed calendar month (April 2026)
    start_year, start_month = 2024, 1
    end_year, end_month = get_most_recent_completed_month()
    start_key = (start_year, start_month)
    end_key = (end_year, end_month)

    # Paths to ignored directory
    data_dir = "csv_files"

    # Target groupings: internal label -> filename prefix used by CRMLS exports
    categories = {'listings': 'Listing', 'sold': 'Sold'}

    # Matches filenames like: CRMLSListing202508.csv / CRMLSSold202412.csv / CRMLSSold202501_filled.csv
    filename_pattern = re.compile(r"^CRMLS(Listing|Sold)(\d{4})(\d{2})(?:_.*)?\.csv$", re.IGNORECASE)

    for category, prefix in categories.items():
        combined_dfs = []
        individual_row_sum = 0
 
        print(f"--- Processing {category.upper()} Datasets ---")

        # 2. Glob once per category using a wildcard, then filter/validate by date 
        # parsed out of each filename instead of one file name per month
        candidate_files = sorted(glob.glob(os.path.join(data_dir, f"CRMLS{prefix}*.csv")))
 
        matched_files = []
        for file_path in candidate_files:
            fname = os.path.basename(file_path)
            m = filename_pattern.match(fname)
            if not m:
                continue  # doesn't match expected naming convention, skip
            _, yr_str, mo_str = m.groups()
            file_key = (int(yr_str), int(mo_str))
            if start_key <= file_key <= end_key:
                matched_files.append(file_path)

        for file_path in matched_files:
            try:
                df = pd.read_csv(file_path)
                row_count = len(df)
                individual_row_sum += row_count
                print(f"Found: {os.path.basename(file_path)} | Rows: {row_count}")
                combined_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # COMMENT CONFIRMING ROW COUNTS BEFORE CONCATENATION:
        # Sum of rows counted directly from individual files = individual_row_sum
        print(f"\n[Count Check] Total rows counted across individual files BEFORE concatenation: {individual_row_sum}")

        if not combined_dfs: 
            print(f"No files found for category: {category}\n")
            continue

        # 3. Contatenate ALL datasets
        master_df = pd.concat(combined_dfs, ignore_index=True)

        # COMMENT CONFIRMING ROW COUNTS AFTER CONCATENATION:
        # master_df length should exactly equal individual_row_sum
        print(f"[Count Check] Total rows in master dataframe AFTER concatenation: {len(master_df)}")
        assert len(master_df) == individual_row_sum, "Row count mismatch after concatenation!"

        # Save the complete unfiltered dataset
        all_output_filename = f"concatenated_{category}_all.csv"
        master_df.to_csv(all_output_filename, index=False)
        
        print(f"SUCCESS: Saved unfiltered dataset to '{all_output_filename}'\n")

        # 4. Filter for 'Residential' type only
        if 'PropertyType' in master_df.columns:
            # COMMENT CONFIRMING ROW COUNTS BEFORE THE RESIDENTIAL FILTER:
            print(f"[Count Check] Rows BEFORE 'Residential' filter: {len(master_df)}")
 
            filtered_df = master_df[master_df['PropertyType'] == 'Residential']
 
            # COMMENT CONFIRMING ROW COUNTS AFTER THE RESIDENTIAL FILTER:
            print(f"[Count Check] Rows AFTER 'Residential' filter: {len(filtered_df)}")
 
            # 5. Save the final clean file outside your hidden folder
            output_filename = f"concatenated_residential_{category}.csv"
            filtered_df.to_csv(output_filename, index=False)
            print(f"SUCCESS: Saved clean dataset to '{output_filename}'\n")
        else:
            print(f"ERROR: 'PropertyType' column missing from {category} dataset.\n")
 
if __name__ == "__main__":
    process_datasets()

         