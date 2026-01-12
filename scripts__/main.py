"""
main.py
Orchestrates the entire pipeline with schema validation and logging
"""
import sys
import os
from datetime import datetime
import time
from dotenv import load_dotenv

from aggregate import run_all_queries
load_dotenv()
import glob

def get_pipeline_config():
    """
    Centralized pipeline configuration
    This is the SINGLE SOURCE OF TRUTH for column names
    """
    return {
        # PRIMARY KEY CONFIGURATION
        'primary_key_columns': ['device_id'],
        
        # DEDUPLICATION STRATEGY
        'dedup_strategy': 'keep_latest',
        
        # UPGRADE SCENARIO HANDLER
        'handle_upgrades': True,
        'name_column': 'customer_name',
        'date_column': 'commissioning_date',
        
        # DATA TYPES
        'type_mapping': {},
        
        # DATE COLUMNS
        'date_columns':  [
            'commissioning_date',
            'maintenance_sub_start_date',
            'maintenance_sub_end_date',
            'maintenance_sub_end_date',
            'last_maintenance_date',
            'h12025_maintenance_date',
            'next_maint_date',
            'h22025_maintenance_date',
            'h12026_maintenance_date',
        ],
        'date_formats': ['%d-%b-%Y', '%d/%m/%Y', '%Y-%m-%d'],
        
        # FILTERS
        'filter_rules': {'state': ['Test Systems', 'test', 'Test System']},
        'custom_filters': [],
        
        # MISSING VALUE STRATEGY
        'missing_value_strategy': 'default',
        
        # METADATA
        'add_metadata': True,
        'pipeline_version': '1.0',
    }

def run_pipeline():
    """Run the complete data pipeline with logging"""
    print("=" * 60)
    print(f"Starting Pipeline: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize pipeline stats
    pipeline_stats = {
        'timestamp': datetime.now(),
        'status': 'Running',
        'rows_loaded': 0,
        'rows_final': 0,
        'rows_filtered': 0,
        'rows_duplicates': 0,
        'rows_null_pk': 0,
        'columns_cleaned': 0,
        'date_columns': 0,
        'pipeline_version': '1.0',
        'source_file': 'Google Sheets',
        'output_file': '',
        'execution_time': 0,
        'error_message': '',
        'warnings': '',
    }
    
    try:
        # Import modules
        from fetch_sheet import main as fetch_data
        from aggregate import main as aggregate_data
        from upload_to_sheets import main as upload_data
        
        # Step 1: Fetch from Google Sheets
        print("\n[1/4] Fetching data from Google Sheets...")
        fetch_data()
        
        # Step 2: Clean data with schema validation
        print("\n[2/4] Cleaning data...")
        
        import glob
        import pandas as pd
        from clean import process_file
        
        # Find latest raw file
        raw_files = glob.glob('data/raw/raw_*.parquet')
        if not raw_files:
            raise FileNotFoundError("No raw data files found!")
        
        latest_raw = max(raw_files)
        
        # Load raw data for schema validation
        raw_df = pd.read_parquet(latest_raw)
        
        # Get pipeline configuration
        config = get_pipeline_config()
        
        # SCHEMA VALIDATION - Auto-correct column names if renamed
        try:
            from schema_validator import handle_schema_changes
            
            print("\n  → Running schema validation...")
            config = handle_schema_changes(raw_df, config)
            print("  ✓ Schema validation complete")
            
        except ImportError:
            print("  ⚠️  schema_validator.py not found - skipping schema validation")
        except Exception as e:
            print(f"  ⚠️  Schema validation warning: {e}")
            # Continue anyway
        
        # Run cleaning with validated configuration
        output_path, clean_stats = process_file(
            latest_raw,
            primary_key_columns=config['primary_key_columns'],
            dedup_strategy=config['dedup_strategy'],
            handle_upgrades=config['handle_upgrades'],
            name_column=config['name_column'],
            date_column=config['date_column'],
            type_mapping=config['type_mapping'],
            date_columns=config['date_columns'],
            date_formats=config['date_formats'],
            filter_rules=config['filter_rules'],
            custom_filters=config['custom_filters'],
            missing_value_strategy=config['missing_value_strategy'],
            add_metadata=config['add_metadata'],
            pipeline_version=config['pipeline_version'],
        )
        
        # Update pipeline stats with cleaning stats
        pipeline_stats.update(clean_stats)
        
        # Step 3: Run DuckDB aggregations
        # print("\n[3/4] Running SQL aggregations with DuckDB...")
        # aggregate_data()
        from aggregate import run_all_queries  # make sure to import the function
        aggregated_results = run_all_queries()  # <-- capture the dictionary

        # Keep only today's files
        today_str = datetime.now().strftime('%Y%m%d')
        today_files = [p for p in aggregated_results.values() if p and today_str in p]

        if today_files:
            pipeline_stats['output_file'] = today_files[0]  # pick main aggregation
        else:
            pipeline_stats['output_file'] = ''

        for f in glob.glob('data/aggregated/*.parquet'):
            if today_str not in f:
                os.remove(f)
        
        # Step 4: Upload back to Google Sheets
        print("\n[4/4] Uploading results to Google Sheets...")
        upload_data()
        
        # Mark as successful
        pipeline_stats['status'] = 'Success'
        pipeline_stats['execution_time'] = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✓ Pipeline completed successfully!")
        print(f"✓ Total execution time: {pipeline_stats['execution_time']:.2f} seconds")
        print("=" * 60)
        
    except Exception as e:
        pipeline_stats['status'] = 'Failed'
        pipeline_stats['error_message'] = str(e)
        pipeline_stats['execution_time'] = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"✗ Pipeline failed: {e}")
        print("=" * 60)
    
    finally:
        # Always upload pipeline log (even if failed)
        try:
            LOG_SHEET_URL = os.getenv('OUTPUT_SHEET_URL')
            if LOG_SHEET_URL:
                print("\n[5/5] Uploading pipeline log to Google Sheets...")
                from upload_pipeline_logs import upload_pipeline_log
                upload_pipeline_log(LOG_SHEET_URL, pipeline_stats)
            else:
                print("\n⚠ Skipping log upload (OUTPUT_SHEET_URL not set)")
        except Exception as log_error:
            print(f"\n⚠ Failed to upload pipeline log: {log_error}")
        
        if pipeline_stats['status'] == 'Failed':
            sys.exit(1)

if __name__ == "__main__":
    run_pipeline()