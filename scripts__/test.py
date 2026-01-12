from dotenv import load_dotenv
import os

load_dotenv()

cred_json = os.getenv('GOOGLE_SHEETS_CRED')
print(cred_json)

import pandas as pd

# Load your Parquet file
df = pd.read_parquet("data/processed/processed_20260111.parquet")  # put your actual file path

# Preview the data
print(df.head())      # shows first 5 rows
print(df.info())      # shows columns, types, non-null counts
print(df.describe())  # summary stats for numeric columns


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