# from dotenv import load_dotenv
# import os

# load_dotenv()

# cred_json = os.getenv('GOOGLE_SHEETS_CRED')
# print(cred_json)

# import pandas as pd

# # Load your Parquet file
# df = pd.read_parquet("data/processed/processed_20260111.parquet")  # put your actual file path

# # Preview the data
# print(df.head())      # shows first 5 rows
# print(df.info())      # shows columns, types, non-null counts
# print(df.describe())  # summary stats for numeric columns


# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
# import os, json

# # Load creds
# with open(os.getenv("GOOGLE_SHEETS_CREDS_FILE")) as f:
#     creds_dict = json.load(f)
# scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
# client = gspread.authorize(creds)

# # Open spreadsheet
# sheet_url = "https://docs.google.com/spreadsheets/d/1M0UhXG8gnSTMSmalPbSmhpJE6yPoor3-ZtjsbUpa7Zk/edit?gid=0#gid=0"
# spreadsheet = client.open_by_url(sheet_url)

# # List all worksheets
# for ws in spreadsheet.worksheets():
#     print(ws.title)

# import pandas as pd
# import glob
# import os

# # Load the file you just saved
# list_of_files = glob.glob('data/processed/*_enriched.parquet')
# latest_file = max(list_of_files, key=os.path.getctime)
# df = pd.read_parquet(latest_file)

# print("--- Cluster Distribution ---")
# print(df['area_cluster'].value_counts())

# print("\n--- The Remaining 'Unclustered' Rows ---")
# # See the addresses we missed so you can add them to the dictionary later
# failures = df[df['area_cluster'].str.contains('Unclustered', na=False)]['customer_address'].head(10)
# print(failures)


import pandas as pd
import glob
import os

def save_unclustered_rows():
    # 1. Find the latest enriched file
    list_of_files = glob.glob('data/processed/*_enriched.parquet')
    if not list_of_files:
        print("No enriched files found.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Reading: {os.path.basename(latest_file)}")
    
    df = pd.read_parquet(latest_file)

    # 2. Filter for "Unclustered"
    # We check for "Lagos - Unclustered" OR just "Unclustered"
    unclustered_mask = df['area_cluster'].astype(str).str.contains('Unclustered', na=False, case=False)
    failures_df = df[unclustered_mask]

    if failures_df.empty:
        print("Good news! No unclustered addresses found.")
        return

    # 3. Select relevant columns to make it easy to read
    # We try to grab the address and state columns for context
    cols_to_save = []
    
    # Check for address column (using the name from your snippet)
    if 'customer_address' in df.columns:
        cols_to_save.append('customer_address')
    elif 'address' in df.columns:
        cols_to_save.append('address')
        
    if 'state' in df.columns:
        cols_to_save.append('state')
        
    # If we found columns, verify them, otherwise just save everything
    if cols_to_save:
        output_df = failures_df[cols_to_save]
    else:
        output_df = failures_df

    # 4. Save to CSV
    output_filename = "unclustered_addresses.csv"
    output_df.to_csv(output_filename, index=False)
    
    print(f"\n--- SUCCESS ---")
    print(f"Found {len(failures_df)} unclustered rows.")
    print(f"Saved to: {output_filename}")
    print("\nNext Step: Open that CSV, copy the addresses, and paste them here!")

if __name__ == "__main__":
    save_unclustered_rows()