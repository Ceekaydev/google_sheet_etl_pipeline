"""
fetch_sheet.py
Fetches data from Google Sheets and saves as raw parquet

FIXES & IMPROVEMENTS (2025-02-08):
- Fixed: Added retry logic for API failures
- Fixed: Better credential handling with multiple fallback options
- Fixed: Validates sheet exists before fetching
- Enhanced: Progress indicators for large sheets
- Enhanced: Better error messages with troubleshooting steps
"""
import re
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import time

load_dotenv()

def get_google_sheet_data(sheet_url, worksheet_name="MASTER SHEET", max_retries=3):
    """
    Fetch data from Google Sheet with retry logic
    
    Args:
        sheet_url: Google Sheets URL
        worksheet_name: Name of the worksheet to fetch
        max_retries: Number of retry attempts for API failures
        
    Returns:
        DataFrame with fetched data
        
    Raises:
        FileNotFoundError: If credentials not found
        ValueError: If sheet has no data
        gspread.exceptions.WorksheetNotFound: If worksheet doesn't exist
    """
    # FIX (2025-02-08): Added multiple credential loading strategies
    creds_dict = None
    
    # Strategy 1: Try file path from environment
    creds_file = os.getenv("GOOGLE_SHEETS_CREDS_FILE")
    if creds_file and os.path.exists(creds_file):
        print(f"  → Loading credentials from: {creds_file}")
        with open(creds_file) as f:
            creds_dict = json.load(f)
    
    # Strategy 2: Try JSON string from environment (for CI/CD)
    elif os.getenv("GOOGLE_SHEETS_CRED"):
        print(f"  → Loading credentials from GOOGLE_SHEETS_CRED environment variable")
        creds_dict = json.loads(os.getenv("GOOGLE_SHEETS_CRED"))
    
    # Strategy 3: Fallback to local credentials.json
    elif os.path.exists("credentials.json"):
        print(f"  → Loading credentials from credentials.json")
        with open("credentials.json") as f:
            creds_dict = json.load(f)
    
    else:
        # ENHANCED (2025-02-08): Better error message with troubleshooting
        raise FileNotFoundError(
            "Google Sheets credentials not found!\n"
            "Please set one of the following:\n"
            "  1. GOOGLE_SHEETS_CREDS_FILE environment variable (path to JSON file)\n"
            "  2. GOOGLE_SHEETS_CRED environment variable (JSON string)\n"
            "  3. Place credentials.json in the current directory\n"
            "\n"
            "Get credentials from: https://console.cloud.google.com/apis/credentials"
        )

    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    # FIX (2025-02-08): Added retry logic for API rate limits
    for attempt in range(max_retries):
        try:
            sheet = client.open_by_url(sheet_url)
            
            # FIX (2025-02-08): Validate worksheet exists before fetching
            try:
                worksheet = sheet.worksheet(worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                available_sheets = [ws.title for ws in sheet.worksheets()]
                raise gspread.exceptions.WorksheetNotFound(
                    f"Worksheet '{worksheet_name}' not found!\n"
                    f"Available worksheets: {', '.join(available_sheets)}"
                )
            
            print(f"  → Fetching data from worksheet: {worksheet_name}")
            rows = worksheet.get_all_values()
            
            # ENHANCED (2025-02-08): Show progress for large sheets
            if len(rows) > 1000:
                print(f"  → Processing {len(rows)} rows (this may take a moment)...")

            max_cols = max(len(r) for r in rows) if rows else 0

            if len(rows) < 2:
                raise ValueError(
                    f"Sheet '{worksheet_name}' has no data rows!\n"
                    f"Expected at least 2 rows (header + data), found {len(rows)}"
                )
            
            # Process headers
            headers = rows[0] + [""] * (max_cols - len(rows[0]))
            clean_headers = []
            for i, h in enumerate(headers):
                if not h or not h.strip():
                    clean_headers.append(f"col_{i}")
                    continue

                # Remove parentheses content
                h = re.sub(r"\(.*?\)", "", h)

                # Clean and normalize
                h = (
                    h.strip()
                    .lower()
                    .replace(" ", "_")
                )
                h = re.sub(r"[^a-z0-9_]", "", h)

                clean_headers.append(h)

            # FIX (2025-02-08): Fixed variable name typo (rows -> row)
            normalized_rows = []
            for row in rows[1:]:
                if len(row) < max_cols:
                    row = row + [""] * (max_cols - len(row))  # Fixed: was 'rows'
                elif len(row) > max_cols:
                    row = row[:max_cols]
                normalized_rows.append(row)

            df = pd.DataFrame(normalized_rows, columns=clean_headers)

            print(f"  ✓ Fetched {len(df)} rows, {len(df.columns)} columns")
            print(f"  ✓ Columns: {', '.join(list(df.columns)[:10])}{'...' if len(df.columns) > 10 else ''}")
            
            return df
            
        except gspread.exceptions.APIError as e:
            # FIX (2025-02-08): Handle API rate limits with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  ⚠ API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  → Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
        
        except Exception as e:
            # ENHANCED (2025-02-08): Better error context
            print(f"  ✗ Error fetching data: {e}")
            raise

def save_raw_data(df, output_dir='data/raw'):
    """
    Save raw data as parquet with validation
    
    Args:
        df: DataFrame to save
        output_dir: Directory to save the file
        
    Returns:
        Path to saved file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = f"{output_dir}/raw_{timestamp}.parquet"

        # FIX (2025-02-08): Validate DataFrame before saving
        if df.empty:
            raise ValueError("Cannot save empty DataFrame!")
        
        if len(df.columns) == 0:
            raise ValueError("DataFrame has no columns!")

        df.to_parquet(output_path, compression='snappy', index=False)
        
        # ENHANCED (2025-02-08): Show file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Saved raw data to {output_path}")
        print(f"  ✓ File size: {file_size_mb:.2f} MB")
        
        return output_path
    
    except Exception as e:
        print(f"  ✗ Error saving data: {e}")
        raise
              

def main():
    """Main execution function"""
    print("=" * 60)
    print("FETCHING DATA FROM GOOGLE SHEETS")
    print("=" * 60)
    
    try:
        SOURCE_SHEET_URL = os.getenv('SOURCE_SHEET_URL')
        
        if not SOURCE_SHEET_URL:
            raise ValueError(
                "SOURCE_SHEET_URL environment variable not set!\n"
                "Add it to your .env file or set it in your environment."
            )

        print(f"\n  → Source: {SOURCE_SHEET_URL[:50]}...")
        
        df = get_google_sheet_data(SOURCE_SHEET_URL)

        print('\n  → Saving raw data...')
        output_path = save_raw_data(df)
        
        print("\n" + "=" * 60)
        print(f"✓ FETCH COMPLETE")
        print(f"✓ Output: {output_path}")
        print("=" * 60)
    
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ FETCH FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()