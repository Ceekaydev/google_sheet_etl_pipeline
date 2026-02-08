"""
upload_to_sheets.py
Uploads multiple aggregated results to separate Google Sheet tabs
Each query gets its own worksheet

FIX (2025-02-08): Removed dynamic monthly suffix - worksheets now have constant names
"""
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json
import os
import glob

from dotenv import load_dotenv
load_dotenv()

def get_sheets_client():
    """Initialize Google Sheets client"""

    creds_dict = None

    # 1. Try file path from env
    creds_file = os.getenv("GOOGLE_SHEETS_CREDS_FILE")
    if creds_file and os.path.exists(creds_file):
        with open(creds_file, "r") as f:
            creds_dict = json.load(f)

    # 2. Try JSON string from env
    elif os.getenv("GOOGLE_SHEETS_CRED"):
        creds_dict = json.loads(os.getenv("GOOGLE_SHEETS_CRED"))

    # 3. Fallback to local file
    elif os.path.exists("credentials.json"):
        with open("credentials.json", "r") as f:
            creds_dict = json.load(f)

    else:
        raise FileNotFoundError(
            "Google Sheets credentials not found.\n"
            "Set GOOGLE_SHEETS_CREDS_FILE or GOOGLE_SHEETS_CRED or add credentials.json"
        )

    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

def upload_dataframe_to_sheet(df, spreadsheet, sheet_name):
    """
    Upload a dataframe to a specific worksheet
    Creates worksheet if it doesn't exist
    """
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        print(f"  → Found existing sheet: {sheet_name}")
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        print(f"  → Creating new sheet: {sheet_name}")
        worksheet = spreadsheet.add_worksheet(
            title=sheet_name,
            rows=max(len(df) + 1, 100),  # Add buffer rows
            cols=max(len(df.columns), 10)
        )
    
    # Convert datetime columns to string
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Replace infinities
    df = df.replace([float('inf'), float('-inf')], None)

    # Replace pandas NA with real Python None
    df = df.replace({pd.NA: None})

    # Ensure object dtype so JSON encoder is safe
    df = df.astype(object)

    # Prepare data for upload
    data = [df.columns.values.tolist()] + df.values.tolist()
    
    # Upload in batches if large dataset
    if len(data) > 1000:
        print(f"  → Uploading {len(data)} rows in batches...")
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if i == 0:
                worksheet.update(batch, value_input_option='RAW')
            else:
                worksheet.append_rows(batch, value_input_option='RAW')
    else:
        worksheet.update(data, value_input_option='RAW')
    
    print(f"  ✓ Uploaded {len(df)} rows to '{sheet_name}'")
    return worksheet

def upload_all_aggregations(sheet_url):
    """
    Find all aggregation files and upload each to separate worksheets
    
    FIX (2025-02-08): Removed dynamic monthly suffix - keeps constant worksheet names
    
    Naming convention:
    - File: customers_without_h2_ppm_lagos_20260208.parquet (date in filename)
    - Sheet: customers_without_h2_ppm_lagos (NO date suffix - CONSTANT)
    
    Benefits:
    - Same 14 worksheets updated each run (no proliferation of tabs)
    - Latest data always in the same place
    - Cleaner, more maintainable spreadsheet
    """
    agg_files = glob.glob('data/aggregated/*_*.parquet')
    
    if not agg_files:
        print("No aggregation files found!")
        return
    
    print("=" * 60)
    print(f"Uploading {len(agg_files)} aggregations to Google Sheets")
    print("=" * 60)
    
    # Initialize Google Sheets client
    client = get_sheets_client()
    spreadsheet = client.open_by_url(sheet_url)
    
    # FIX (2025-02-08): REMOVED this line that added dynamic month suffix
    # OLD: current_month = datetime.now().strftime('%Y_%m')
    # NEW: No month variable - use query name directly
    
    uploaded_count = 0
    
    for file_path in agg_files:
        # Extract query name from filename
        # Example: data/aggregated/customers_without_h2_ppm_lagos_20260208.parquet
        filename = os.path.basename(file_path)
        query_name = '_'.join(filename.split('_')[:-1])  # Remove date suffix
        
        # FIX (2025-02-08): Use query_name directly WITHOUT adding month suffix
        # OLD: sheet_name = f"{query_name}_{current_month}"  # ❌ Creates new tabs monthly
        # NEW: sheet_name = query_name                        # ✅ Constant worksheet names
        sheet_name = query_name
        
        print(f"\n[{uploaded_count + 1}/{len(agg_files)}] {query_name}")
        
        try:
            # Load parquet file
            df = pd.read_parquet(file_path)
            print(f"  → Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Upload to Google Sheets
            upload_dataframe_to_sheet(df, spreadsheet, sheet_name)
            uploaded_count += 1
            
        except Exception as e:
            print(f"  ✗ Failed to upload: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully uploaded {uploaded_count}/{len(agg_files)} worksheets")
    print(f"✓ All worksheets use constant names (no date suffix)")
    print("=" * 60)
    
    return uploaded_count

def upload_single_query(sheet_url, query_name):
    """
    Upload a specific query result to Google Sheets
    Useful for testing individual queries
    
    FIX (2025-02-08): Also updated to use constant sheet names
    """
    # Find the latest file for this query
    pattern = f'data/aggregated/{query_name}_*.parquet'
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found for query: {query_name}")
        return
    
    latest_file = max(files)
    df = pd.read_parquet(latest_file)
    
    client = get_sheets_client()
    spreadsheet = client.open_by_url(sheet_url)
    
    # FIX (2025-02-08): Use query_name directly without month suffix
    # OLD: current_month = datetime.now().strftime('%Y_%m')
    # OLD: sheet_name = f"{query_name}_{current_month}"
    # NEW: sheet_name = query_name
    sheet_name = query_name
    
    upload_dataframe_to_sheet(df, spreadsheet, sheet_name)
    print(f"Uploaded {query_name} to worksheet: {sheet_name}")

def main():
    OUTPUT_SHEET_URL = os.getenv('OUTPUT_SHEET_URL')
    
    if not OUTPUT_SHEET_URL:
        print("ERROR: OUTPUT_SHEET_URL not set!")
        return
    
    upload_all_aggregations(OUTPUT_SHEET_URL)

if __name__ == "__main__":
    main()