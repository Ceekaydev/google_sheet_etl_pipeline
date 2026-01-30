"""
upload_pipeline_logs.py
Saves pipeline execution logs and reports to Google Sheets
This creates a "Pipeline_Logs" sheet that you can use with Apps Script for email reports
"""
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json
import os
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


def create_pipeline_report(pipeline_stats):
    """
    Create a structured pipeline report
    
    Args:
        pipeline_stats: Dictionary with pipeline execution details
        
    Returns:
        DataFrame with pipeline report
    """

    ts = pipeline_stats.get('timestamp', datetime.now())
    if isinstance(ts, (datetime, pd.Timestamp)):
        ts = ts.isoformat()

    warnings = pipeline_stats.get('warnings', '')
    if isinstance(warnings, list):
        warnings = " | ".join(warnings)



    report_data = {
        'timestamp': [ts],
        'status': [pipeline_stats.get('status', 'Success')],
        'enrichment_status': [pipeline_stats.get('enrichment_status', 'N/A')],
        'total_rows_loaded': [pipeline_stats.get('rows_loaded', 0)],
        'total_rows_final': [pipeline_stats.get('rows_final', 0)],
        'rows_filtered': [pipeline_stats.get('rows_filtered', 0)],
        'rows_duplicates_removed': [pipeline_stats.get('rows_duplicates', 0)],
        'rows_null_pk_removed': [pipeline_stats.get('rows_null_pk', 0)],
        'columns_cleaned': [pipeline_stats.get('columns_cleaned', 0)],
        'date_columns_converted': [pipeline_stats.get('date_columns', 0)],
        'pipeline_version': [pipeline_stats.get('pipeline_version', '1.0')],
        'source_file': [pipeline_stats.get('source_file', 'N/A')],
        'output_file': [pipeline_stats.get('output_file', 'N/A')],
        'execution_time_seconds': [pipeline_stats.get('execution_time', 0)],
        'error_message': [pipeline_stats.get('error_message', '')],
        'warnings': [warnings],
    }
    
    return pd.DataFrame(report_data)

def upload_pipeline_log(sheet_url, pipeline_stats, log_sheet_name='Pipeline_logs_1'):
    """
    Upload pipeline execution log to Google Sheets
    Appends to existing log sheet or creates new one
    
    Args:
        sheet_url: URL of the Google Sheet
        pipeline_stats: Dictionary with pipeline execution details
        log_sheet_name: Name of the worksheet for logs
    """
    print("\n" + "=" * 60)
    print("UPLOADING PIPELINE LOG TO GOOGLE SHEETS")
    print("=" * 60)
    
    client = get_sheets_client()
    spreadsheet = client.open_by_url(sheet_url)
    
    # Create report DataFrame
    report_df = create_pipeline_report(pipeline_stats)
    
    # Try to get existing log sheet
    try:
        worksheet = spreadsheet.worksheet(log_sheet_name)
        print(f"  → Found existing log sheet: {log_sheet_name}")
        
        # Get existing data
        existing_data = worksheet.get_all_values()
        
        if len(existing_data) > 0:
            # Append new row
            new_row = report_df.values.tolist()[0]
            worksheet.append_row(new_row, value_input_option='RAW')
            print(f"  ✓ Appended log entry (Total logs: {len(existing_data)})")
        else:
            # First entry - add headers + data
            data = [report_df.columns.tolist()] + report_df.values.tolist()
            worksheet.update(data, value_input_option='RAW')
            print(f"  ✓ Created first log entry")
            
    except gspread.exceptions.WorksheetNotFound:
        # Create new log sheet
        print(f"  → Creating new log sheet: {log_sheet_name}")
        worksheet = spreadsheet.add_worksheet(
            title=log_sheet_name,
            rows=1000,  # Room for many logs
            cols=15
        )
        
        # Add headers + first entry
        data = [report_df.columns.tolist()] + report_df.values.tolist()
        worksheet.update(data, value_input_option='RAW')
        print(f"  ✓ Created log sheet with first entry")
    
    # Format the sheet for readability
    try:
        # Bold the header row
        worksheet.format('A1:O1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
        })
        
        # Auto-resize columns
        worksheet.columns_auto_resize(0, 14)
    except:
        pass  # Formatting is optional
    
    print("=" * 60)
    print("✓ PIPELINE LOG UPLOADED SUCCESSFULLY")
    print("=" * 60)

def create_detailed_report(sheet_url, pipeline_stats, quality_report_path=None):
    """
    Create a detailed pipeline report in a separate sheet
    This includes the full quality report data
    
    Args:
        sheet_url: URL of the Google Sheet
        pipeline_stats: Dictionary with pipeline execution details
        quality_report_path: Path to quality report JSON file
    """
    print("\n--- Creating Detailed Report ---")
    
    client = get_sheets_client()
    spreadsheet = client.open_by_url(sheet_url)
    
    # Create sheet name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_sheet_name = f"Report_{timestamp}"
    
    # Create new sheet
    worksheet = spreadsheet.add_worksheet(
        title=report_sheet_name,
        rows=100,
        cols=10
    )
    
    # Build report content
    report_lines = [
        ['PIPELINE EXECUTION REPORT'],
        [''],
        ['Execution Details'],
        ['Timestamp', pipeline_stats.get('timestamp', 'N/A')],
        ['Status', pipeline_stats.get('status', 'Success')],
        ['Pipeline Version', pipeline_stats.get('pipeline_version', '1.0')],
        ['Execution Time', f"{pipeline_stats.get('execution_time', 0):.2f} seconds"],
        [''],
        ['Data Processing Summary'],
        ['Rows Loaded', pipeline_stats.get('rows_loaded', 0)],
        ['Rows After Filtering', pipeline_stats.get('rows_loaded', 0) - pipeline_stats.get('rows_filtered', 0)],
        ['Rows Duplicates Removed', pipeline_stats.get('rows_duplicates', 0)],
        ['Rows NULL PK Removed', pipeline_stats.get('rows_null_pk', 0)],
        ['Final Row Count', pipeline_stats.get('rows_final', 0)],
        [''],
        ['Column Information'],
        ['Columns Cleaned', pipeline_stats.get('columns_cleaned', 0)],
        ['Date Columns Converted', pipeline_stats.get('date_columns', 0)],
        [''],
        ['Files'],
        ['Source File', pipeline_stats.get('source_file', 'N/A')],
        ['Output File', pipeline_stats.get('output_file', 'N/A')],
    ]
    
    # Add warnings if any
    if pipeline_stats.get('warnings'):
        report_lines.extend([
            [''],
            ['Warnings'],
            [pipeline_stats.get('warnings', '')]
        ])
    
    # Add error if any
    if pipeline_stats.get('error_message'):
        report_lines.extend([
            [''],
            ['Errors'],
            [pipeline_stats.get('error_message', '')]
        ])
    
    # Load quality report if available
    if quality_report_path and os.path.exists(quality_report_path):
        try:
            with open(quality_report_path, 'r') as f:
                quality_data = json.load(f)
            
            report_lines.extend([
                [''],
                ['Data Quality Details'],
                ['Column Name', 'Data Type', 'Null Count', 'Null %', 'Unique Values']
            ])
            
            for col_name, col_info in quality_data.get('columns', {}).items():
                report_lines.append([
                    col_name,
                    col_info.get('dtype', 'N/A'),
                    col_info.get('null_count', 0),
                    f"{col_info.get('null_percentage', 0)}%",
                    col_info.get('unique_values', 0)
                ])
        except:
            pass
    
    # Upload to sheet
    worksheet.update(report_lines, value_input_option='RAW')
    
    # Format header
    worksheet.format('A1:J1', {
        'textFormat': {'bold': True, 'fontSize': 14},
        'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.8}
    })
    
    print(f"  ✓ Created detailed report sheet: {report_sheet_name}")

def main():
    """
    Test function - normally called from clean_data.py
    """
    # Example usage
    OUTPUT_SHEET_URL = os.getenv('OUTPUT_SHEET_URL')
    print(f"Output Sheet URL: {OUTPUT_SHEET_URL}")
    
    if not OUTPUT_SHEET_URL:
        print("ERROR: OUTPUT_SHEET_URL not set!")
        return
    
    # Example pipeline stats
    test_stats = {
        'timestamp': datetime.now().isoformat(),
        'status': 'Success',
        'rows_loaded': 5000,
        'rows_final': 4850,
        'rows_filtered': 50,
        'rows_duplicates': 75,
        'rows_null_pk': 25,
        'columns_cleaned': 15,
        'date_columns': 3,
        'pipeline_version': '1.0',
        'source_file': 'raw_20260111.parquet',
        'output_file': 'processed_20260111.parquet',
        'execution_time': 12.5,
        'error_message': '',
        'warnings': 'Some dates failed to parse (2.1%)',
    }
    
    upload_pipeline_log(OUTPUT_SHEET_URL, test_stats)

if __name__ == "__main__":
    main()