"""
fetch_sheets.py
Fetches data from Google Sheets and saves as raw parquet
"""
import re
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()

def get_google_sheet_data(sheet_url, worksheet_name="MASTER SHEET"):

    try:
        """Fetch data from Google Sheet"""
        with open(os.getenv("GOOGLE_SHEETS_CREDS_FILE")) as f:
            creds_dict = json.load(f)


        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.worksheet(worksheet_name)


        # data = worksheet.get_all_records()
        rows = worksheet.get_all_values()

        max_cols = max(len(r) for r in rows)

        if len(rows) < 2:
            raise ValueError("Sheet has no data rows")
        
        headers = rows[0] + [""] * (max_cols - len(rows[0]))
        clean_headers = []
        for i, h in enumerate(headers):
            if not h or not h.strip():
                clean_headers.append(f"col_{i}")
                continue

            h = re.sub(r"\(.*?\)", "", h)

            h = (
                h.strip()
                .lower()
                .replace(" ", "_")
            )
            h = re.sub(r"[^a-z0-9_]", "", h)

            clean_headers.append(h)

        normalized_rows = []
        for row in rows[1:]:
            if len(row) < max_cols:
                rows = row + [""] * (max_cols - len(row))

            elif len(row) > max_cols:
                row = row[:max_cols]
            normalized_rows.append(row)

        df = pd.DataFrame(normalized_rows, columns=clean_headers)

        print(f"Fetched {len(df)} rows, {len(df.columns)} columns")
        print("Columns:", list(df.columns))
        return df
    
    except Exception:
        raise

def save_raw_data(df, output_dir='data/raw'):
    """Save raw data as parquet"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = f"{output_dir}/raw_{timestamp}.parquet"

        df.to_parquet(output_path, compression='snappy', index=False)
        print(f"Saved raw data to {output_path}")
        return output_path
    
    except Exception:
        raise
              

def main():
    try:
            
        SOURCE_SHEET_URL = os.getenv('SOURCE_SHEET_URL')

        print("Fetching data from Google Sheets...")
        df = get_google_sheet_data(SOURCE_SHEET_URL)

        print('Saving raw data...')
        output_path = save_raw_data(df)
        
        print(f"Fetch complete: {output_path}")
    
    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    main()