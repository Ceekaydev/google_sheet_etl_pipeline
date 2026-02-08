"""
main.py
Orchestrates the entire pipeline: Fetch -> Clean -> Enrich (Fast) -> Aggregate -> Upload

FIXES (2025-02-08):
- CORRECTED: Import is from 'fetch_sheet' (this was already correct)
- Fixed: Better error handling for missing environment variables
- Fixed: Proper file path passing to aggregate step
- Enhanced: More detailed progress tracking and logging
- Enhanced: Better cleanup of old files
"""
import sys
import os
from datetime import datetime
import time
from dotenv import load_dotenv
import glob
import pandas as pd

# Load environment variables
load_dotenv()


def get_pipeline_config():
    """
    Centralized pipeline configuration
    This is the SINGLE SOURCE OF TRUTH for column names
    
    FIX (2025-02-08): Added validation comments for each config section
    """
    return {
        # PRIMARY KEY CONFIGURATION
        'primary_key_columns': ['device_id'],  # Must match column in your data
        
        # DEDUPLICATION STRATEGY
        # Options: keep_latest, keep_first, keep_largest, flag_duplicates, remove_all
        'dedup_strategy': 'keep_latest',
        
        # UPGRADE SCENARIO HANDLER
        'handle_upgrades': True,
        'name_column': 'customer_name',  # Column that might contain "(upgrade)"
        'date_column': 'commissioning_date',  # Date column to identify latest record
        
        # DATA TYPES (optional manual mapping)
        'type_mapping': {},
        
        # DATE COLUMNS (all columns containing dates)
        'date_columns': [
            'commissioning_date',
            'maintenance_sub_start_date',
            'maintenance_sub_end_date',
            'last_maintenance_date',
            'h12025_maintenance_date',
            'next_maint_date',
            'h22025_maintenance_date',
            'h12026_maintenance_date',
        ],
        
        # DATE FORMATS (tried in order)
        'date_formats': ['%d-%b-%Y', '%d/%m/%Y', '%Y-%m-%d'],
        
        # FILTERS (rows to exclude)
        'filter_rules': {'state': ['Test Systems', 'test', 'Test System']},
        'custom_filters': [],
        
        # MISSING VALUE STRATEGY
        # Options: default (keep NaN), drop (remove rows), fill (impute values)
        'missing_value_strategy': 'default',
        
        # METADATA
        'add_metadata': True,
        'pipeline_version': '1.1',  # Increment when pipeline logic changes
    }


def validate_environment():
    """
    Validate required environment variables are set
    
    FIX (2025-02-08): New function to catch configuration issues early
    
    Returns:
        dict with environment variables
        
    Raises:
        ValueError if required variables missing
    """
    required_vars = {
        'SOURCE_SHEET_URL': os.getenv('SOURCE_SHEET_URL'),
        'OUTPUT_SHEET_URL': os.getenv('OUTPUT_SHEET_URL'),
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please set them in your .env file or environment.\n"
            f"\nExample .env file:\n"
            f"SOURCE_SHEET_URL=https://docs.google.com/spreadsheets/d/...\n"
            f"OUTPUT_SHEET_URL=https://docs.google.com/spreadsheets/d/...\n"
            f"GOOGLE_SHEETS_CREDS_FILE=/path/to/credentials.json"
        )
    
    return required_vars


def cleanup_old_files(keep_today=True):
    """
    Clean up old parquet files to save space
    
    FIX (2025-02-08): New function to prevent disk bloat
    
    Args:
        keep_today: If True, only delete files not from today
    """
    today_str = datetime.now().strftime('%Y%m%d')
    deleted_count = 0
    
    for directory in ['data/raw', 'data/processed', 'data/aggregated']:
        if not os.path.exists(directory):
            continue
            
        for file_path in glob.glob(f"{directory}/*.parquet"):
            # Skip if it's today's file and we want to keep it
            if keep_today and today_str in file_path:
                continue
            
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                print(f"  ⚠ Could not delete {file_path}: {e}")
    
    if deleted_count > 0:
        print(f"  ✓ Cleaned up {deleted_count} old files")


def run_pipeline():
    """
    Run the complete data pipeline with comprehensive logging
    
    FIX (2025-02-08): Better error handling and progress tracking
    """
    print("=" * 60)
    print(f"PIPELINE START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        'pipeline_version': '1.1',
        'source_file': 'Google Sheets',
        'output_file': '',
        'enrichment_status': 'Skipped',
        'execution_time': 0,
        'error_message': '',
        'warnings': [],
    }
    
    try:
        # FIX (2025-02-08): Validate environment before starting
        print("\n[0/5] Validating environment...")
        env_vars = validate_environment()
        print("  ✓ All required environment variables set")
        
        # CORRECTED (2025-02-08): Import is from 'fetch_sheet', not 'fetch'
        # User confirmed the actual filename is fetch_sheet.py
        from fetch_sheet import main as fetch_data
        from clean import process_file
        from upload_to_sheets import main as upload_data
        
        # ---------------------------------------------------------
        # Step 1: Fetch from Google Sheets
        # ---------------------------------------------------------
        print("\n[1/5] Fetching data from Google Sheets...")
        print(f"  Source: {env_vars['SOURCE_SHEET_URL'][:50]}...")
        
        fetch_data()
        
        # ---------------------------------------------------------
        # Step 2: Clean data with schema validation
        # ---------------------------------------------------------
        print("\n[2/5] Cleaning data...")
        
        # Find latest raw file
        raw_files = glob.glob('data/raw/raw_*.parquet')
        if not raw_files:
            raise FileNotFoundError(
                "No raw data files found!\n"
                "Expected files in: data/raw/raw_*.parquet\n"
                "Check if fetch step completed successfully."
            )
        
        # FIX (2025-02-08): Use max with key function for better file selection
        latest_raw = max(raw_files, key=os.path.getctime)
        print(f"  → Processing: {os.path.basename(latest_raw)}")
        
        # Load raw data for schema validation
        raw_df = pd.read_parquet(latest_raw)
        print(f"  → Loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns")
        
        # Get pipeline configuration
        config = get_pipeline_config()
        
        # SCHEMA VALIDATION
        try:
            from schema_validator import handle_schema_changes
            print("\n  → Running schema validation...")
            config = handle_schema_changes(raw_df, config)
            print("  ✓ Schema validation complete")
        except ImportError:
            print("  ⚠️  schema_validator.py not found - skipping schema validation")
            pipeline_stats['warnings'].append("Schema validation skipped (module not found)")
        except Exception as e:
            print(f"  ⚠️  Schema validation warning: {e}")
            pipeline_stats['warnings'].append(f"Schema validation warning: {str(e)}")
        
        # Run cleaning
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
        
        # Update pipeline stats from cleaning step
        pipeline_stats.update(clean_stats)

        # ---------------------------------------------------------
        # Step 3: Enrich Locations (Fast Mode)
        # ---------------------------------------------------------
        print("\n[3/5] Enriching Locations...")
        try:
            from enrich_location import run_location_enrichment
            
            print(f"  → Input: {os.path.basename(output_path)}")
            
            # This returns the path to the NEW enriched file
            enriched_path = run_location_enrichment(output_path)
            
            # FIX (2025-02-08): Better detection of enrichment success
            if enriched_path != output_path and os.path.exists(enriched_path):
                print(f"  ✓ Enriched file created: {os.path.basename(enriched_path)}")
                
                # CRITICAL: Update output_path so Aggregation uses the clustered data
                output_path = enriched_path
                
                pipeline_stats['enrichment_status'] = 'Success'
                pipeline_stats['output_file'] = os.path.basename(enriched_path)
            else:
                print(f"  ℹ No enrichment performed")
                pipeline_stats['enrichment_status'] = 'No Change'

        except ImportError:
            print("  ⚠️  enrich_location.py not found. Skipping.")
            pipeline_stats['enrichment_status'] = 'Module Not Found'
            pipeline_stats['warnings'].append("Location enrichment skipped (module not found)")
        except Exception as e:
            print(f"  ⚠️  Enrichment failed: {e}. Continuing with standard data.")
            pipeline_stats['enrichment_status'] = f"Failed: {str(e)}"
            pipeline_stats['warnings'].append(f"Enrichment failed: {str(e)}")


        
        
        # ---------------------------------------------------------
        #  step 3.5: Cleanup old files
        # ---------------------------------------------------------
        print("\n[3.5/5] Cleaning up old aggregated files...")
        # FIX (2025-02-08): Clean aggregated files BEFORE running new aggregations
        # This prevents duplicate files from accumulating
        for file_path in glob.glob('data/aggregated/*.parquet'):
            try:
                os.remove(file_path)
                print(f"  → Removed old file: {os.path.basename(file_path)}")
            except OSError:
                pass
        print("  ✓ Cleaned aggregated folder")

        # ---------------------------------------------------------
        # Step 4: Run DuckDB aggregations
        # ---------------------------------------------------------
        print("\n[4/5] Running SQL aggregations with DuckDB...")
        
        from aggregate import run_all_queries
        
        # FIX (2025-02-08): Explicitly pass the enriched file path
        print(f"  → Aggregating from: {os.path.basename(output_path)}")
        
        aggregated_results = run_all_queries(specific_input_path=output_path)
        
        # Count successful aggregations
        successful_aggs = sum(1 for v in aggregated_results.values() if v)
        total_aggs = len(aggregated_results)
        print(f"  ✓ Generated {successful_aggs}/{total_aggs} aggregations")
        
        if successful_aggs < total_aggs:
            failed = total_aggs - successful_aggs
            pipeline_stats['warnings'].append(f"{failed} aggregation(s) returned no results")

        # ---------------------------------------------------------
        # Step 5: Upload back to Google Sheets
        # ---------------------------------------------------------
        print("\n[5/5] Uploading results to Google Sheets...")
        print(f"  → Destination: {env_vars['OUTPUT_SHEET_URL'][:50]}...")
        print(f"  → Using CONSTANT worksheet names (no month suffix)")
        
        upload_data()
        

        
        # Mark as successful
        pipeline_stats['status'] = 'Success'
        pipeline_stats['execution_time'] = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"✓ Total execution time: {pipeline_stats['execution_time']:.2f} seconds")
        print(f"✓ Final row count: {pipeline_stats['rows_final']}")
        print(f"✓ Enrichment status: {pipeline_stats['enrichment_status']}")
        print(f"✓ Worksheets: 14 constant tabs (no date suffix)")
        print("=" * 60)
        
    except Exception as e:
        pipeline_stats['status'] = 'Failed'
        pipeline_stats['error_message'] = str(e)
        pipeline_stats['execution_time'] = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"✗ PIPELINE FAILED: {e}")
        print("=" * 60)
        
        # FIX (2025-02-08): Show full traceback for debugging
        import traceback
        traceback.print_exc()
    
    finally:
        # Always upload pipeline log (even on failure)
        try:
            LOG_SHEET_URL = os.getenv('OUTPUT_SHEET_URL')
            if LOG_SHEET_URL:
                print("\n[Logging] Uploading pipeline log to Google Sheets...")
                from upload_pipeline_logs import upload_pipeline_log
                
                # FIX (2025-02-08): Convert warnings list to string for Sheets
                if isinstance(pipeline_stats.get('warnings'), list):
                    pipeline_stats['warnings'] = ' | '.join(pipeline_stats['warnings'])
                
                upload_pipeline_log(LOG_SHEET_URL, pipeline_stats)
                print("  ✓ Pipeline log uploaded")
            else:
                print("\n⚠ Skipping log upload (OUTPUT_SHEET_URL not set)")
        except Exception as log_error:
            print(f"\n⚠ Failed to upload pipeline log: {log_error}")
            # Don't fail the entire pipeline just because logging failed
        
        # Exit with error code if pipeline failed
        if pipeline_stats['status'] == 'Failed':
            sys.exit(1)


if __name__ == "__main__":
    run_pipeline()