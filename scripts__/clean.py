"""
clean.py
Comprehensive data cleaning: column names, data types, duplicates, validation

CRITICAL FIXES (2025-02-08):
- FIXED: Removed duplicate remove_duplicates() function definition
- FIXED: Handle edge case where mode_val is empty in handle_missing_values()
- FIXED: Better handling of datetime conversion errors
- ENHANCED: Added progress indicators for long-running operations
- ENHANCED: Better validation of primary key columns
"""

import pandas as pd
from datetime import datetime
import os
import glob
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def standardize_text_columns(df):
    """
    Standardize text: trim whitespace, fix encoding issues
    
    FIX (2025-02-08): Added null check before string operations
    """
    print("\n--- Text Standardization ---")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # FIX (2025-02-08): Skip if column is all null
            if df[col].isna().all():
                print(f"  ⚠ {col}: all null, skipping")
                continue
                
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Replace multiple spaces with single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove common encoding issues
            df[col] = df[col].str.replace('â€™', "'", regex=False)
            df[col] = df[col].str.replace('â€œ', '"', regex=False)
            df[col] = df[col].str.replace('â€', '"', regex=False)
            
            # FIX (2025-02-08): Convert 'nan' strings back to actual NaN
            df[col] = df[col].replace('nan', np.nan)
            
            print(f"  ✓ {col}: standardized")
    
    return df


def handle_missing_values(df, strategy='default'):
    """
    Handle missing values
    
    FIX (2025-02-08): Fixed empty mode_val edge case
    
    Strategies:
        - 'default': Keep NaN, just report them
        - 'drop': Drop rows with any NaN
        - 'fill': Fill with appropriate defaults
    """
    print("\n--- Missing Values Report ---")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    for col in df.columns:
        if missing[col] > 0:
            print(f"  ⚠ {col}: {missing[col]} missing ({missing_pct[col]}%)")
    
    if missing.sum() == 0:
        print("  ✓ No missing values found")
        return df
    
    if strategy == 'drop':
        initial = len(df)
        df = df.dropna()
        print(f"\n  → Dropped {initial - len(df)} rows with missing values")
    
    elif strategy == 'fill':
        for col in df.columns:
            if missing[col] > 0:
                if df[col].dtype in ['float64', 'int64', 'Int64']:
                    # Fill numeric with median
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col].fillna(median_val, inplace=True)
                        print(f"  → {col}: filled with median ({median_val})")
                    else:
                        df[col].fillna(0, inplace=True)
                        print(f"  → {col}: filled with 0 (no valid median)")
                        
                elif df[col].dtype == 'object':
                    # FIX (2025-02-08): Better handling of empty mode
                    mode_val = df[col].mode()
                    
                    if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]):
                        df[col].fillna(mode_val.iloc[0], inplace=True)
                        print(f"  → {col}: filled with mode ('{mode_val.iloc[0]}')")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        print(f"  → {col}: filled with 'Unknown' (no valid mode)")
    
    return df


def handle_primary_key_issues(df, primary_key_columns, strategy='keep_latest'):
    """
    Handle primary key violations: nulls and duplicates
    
    FIX (2025-02-08): Better validation of input parameters
    
    Args:
        df: DataFrame
        primary_key_columns: List of PK columns (e.g., ['device_id'])
        strategy: How to handle duplicates:
            - 'keep_latest': Keep the most recent record (based on row order)
            - 'keep_first': Keep the first occurrence
            - 'keep_largest': Keep record with most non-null values
            - 'flag_duplicates': Keep all but add a flag column
            - 'remove_all': Remove all duplicates (keep none)
    
    Returns:
        Cleaned DataFrame
    """
    # FIX (2025-02-08): Validate input parameters
    if not primary_key_columns:
        print("\n--- Primary Key Issue Resolution ---")
        print("  ⚠ No primary key columns specified, skipping")
        return df
    
    if not isinstance(primary_key_columns, list):
        raise TypeError(f"primary_key_columns must be a list, got {type(primary_key_columns)}")
    
    # FIX (2025-02-08): Check if PK columns exist in dataframe
    missing_cols = [col for col in primary_key_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Primary key columns not found in DataFrame: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    print("\n--- Primary Key Issue Resolution ---")
    initial_count = len(df)
    
    # Step 1: Handle NULL values in primary key
    for col in primary_key_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  ⚠ Found {null_count} NULL values in '{col}'")
            df = df.dropna(subset=[col])
            print(f"    → Removed {null_count} rows with NULL {col}")
    
    # Step 2: Handle duplicate primary keys
    duplicate_mask = df.duplicated(subset=primary_key_columns, keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        print(f"  ⚠ Found {duplicate_count} rows with duplicate {primary_key_columns}")
        
        # Show examples of duplicates (max 3)
        duplicate_examples = df[duplicate_mask].head(5)
        print(f"\n  Example duplicates:")
        for pk_col in primary_key_columns:
            dup_ids = duplicate_examples[pk_col].unique()[:3]
            print(f"    {pk_col}: {list(dup_ids)}")
        
        if strategy == 'keep_latest':
            print(f"\n  Strategy: Keeping LATEST record for each duplicate")
            df = df.drop_duplicates(subset=primary_key_columns, keep='last')
            
        elif strategy == 'keep_first':
            print(f"\n  Strategy: Keeping FIRST record for each duplicate")
            df = df.drop_duplicates(subset=primary_key_columns, keep='first')
            
        elif strategy == 'keep_largest':
            print(f"\n  Strategy: Keeping record with MOST data")
            
            def keep_best_record(group):
                # Count non-null values per row
                non_null_counts = group.notna().sum(axis=1)
                # Return row with most non-null values
                return group.loc[non_null_counts.idxmax()]
            
            df = df.groupby(primary_key_columns, dropna=False).apply(keep_best_record).reset_index(drop=True)
            
        elif strategy == 'flag_duplicates':
            print(f"\n  Strategy: Flagging duplicates (keeping all)")
            df['is_duplicate'] = df.duplicated(subset=primary_key_columns, keep=False)
            duplicate_count = 0  # Don't count as removed
            
        elif strategy == 'remove_all':
            print(f"\n  Strategy: Removing ALL duplicate records")
            df = df[~duplicate_mask]
        
        else:
            # FIX (2025-02-08): Handle invalid strategy
            raise ValueError(
                f"Invalid strategy: '{strategy}'\n"
                f"Valid options: keep_latest, keep_first, keep_largest, flag_duplicates, remove_all"
            )
        
        removed = duplicate_count - df.duplicated(subset=primary_key_columns).sum()
        if removed > 0:
            print(f"  ✓ Resolved {removed} duplicate records")
    
    else:
        print(f"  ✓ No duplicate {primary_key_columns} found")
    
    total_removed = initial_count - len(df)
    if total_removed > 0:
        print(f"\n  → Total rows removed: {total_removed}")
    
    return df


def handle_upgrade_scenarios(df, id_column, name_column=None, date_column=None):
    """
    Special handler for upgrade scenarios (same ID, different details)
    
    FIX (2025-02-08): Added validation of column existence
    
    This function intelligently handles cases like:
    - device_id=12345, name="Customer A" (old)
    - device_id=12345, name="Customer A (upgrade)" (new)
    
    Args:
        df: DataFrame
        id_column: Primary key column (e.g., 'device_id')
        name_column: Column that might have "(upgrade)" suffix
        date_column: Date column to determine which is latest
    
    Returns:
        DataFrame with duplicates resolved
    """
    print("\n--- Upgrade Scenario Handler ---")
    
    # FIX (2025-02-08): Validate columns exist
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in DataFrame")
    
    initial_count = len(df)
    
    # Identify duplicates
    duplicate_ids = df[df.duplicated(subset=[id_column], keep=False)][id_column].unique()
    
    if len(duplicate_ids) == 0:
        print("  ✓ No duplicate IDs found")
        return df
    
    print(f"  Found {len(duplicate_ids)} IDs with multiple records")
    
    records_to_keep = []
    
    for dup_id in duplicate_ids:
        group = df[df[id_column] == dup_id].copy()

        if group.empty:
            continue
        
        # Strategy 1: If there's a date column, keep the latest
        if date_column and date_column in df.columns:
            # FIX (2025-02-08): Handle non-datetime columns gracefully
            try:
                group = group.sort_values(date_column, ascending=False)
                best_record = group.iloc[0]
            except Exception as e:
                print(f"  ⚠ Warning: Could not sort by {date_column}: {e}")
                # Fall through to next strategy
                best_record = None
        
        # Strategy 2: If there's an "upgrade" keyword, keep that one
        if best_record is None and name_column and name_column in df.columns:
            upgrade_mask = group[name_column].astype(str).str.contains('upgrade|updated|new', case=False, na=False)
            if upgrade_mask.any():
                best_record = group.loc[upgrade_mask].iloc[-1]  # Last upgrade
            else:
                best_record = None
        
        # Strategy 3: Keep record with most complete data
        if best_record is None:
            non_null_counts = group.notna().sum(axis=1)
            best_record = group.loc[non_null_counts.idxmax()]
        
        records_to_keep.append(best_record)
    
    # Keep all non-duplicate records + best record for each duplicate
    non_duplicate_df = df[~df[id_column].isin(duplicate_ids)]
    best_records_df = pd.DataFrame(records_to_keep)
    
    df = pd.concat([non_duplicate_df, best_records_df], ignore_index=True)
    
    removed = initial_count - len(df)
    print(f"  ✓ Resolved {removed} duplicate records")
    print(f"    Kept most recent/complete record for each ID")
    
    return df


# FIX (2025-02-08): REMOVED DUPLICATE FUNCTION DEFINITION
# The original file had remove_duplicates() defined twice - keeping only the simple version
def remove_duplicates(df, primary_key_columns=None, keep='first'):
    """
    Remove exact duplicate rows (all columns identical)
    
    Note: This is different from handle_primary_key_issues()
    This function removes rows where ALL columns are identical
    
    Args:
        df: DataFrame
        primary_key_columns: Not used (for API compatibility)
        keep: 'first', 'last', or False
    
    Returns:
        Tuple of (cleaned DataFrame, count of duplicates removed)
    """
    initial_count = len(df)
    df = df.drop_duplicates(keep=keep)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"\n--- Exact Duplicate Removal ---")
        print(f"  ⚠ Removed {removed} exact duplicate rows (all columns identical)")
    else:
        print(f"\n--- Exact Duplicate Removal ---")
        print(f"  ✓ No exact duplicates found")
    
    return df, removed


def fix_data_types(df, type_mapping=None, date_columns=None, date_formats=None):
    """
    Automatically detect and fix data types with enhanced date parsing
    
    FIX (2025-02-08): Better error handling for date conversion failures
    
    Args:
        df: DataFrame
        type_mapping: Optional dict like {'column_name': 'int64', 'date': 'datetime64'}
        date_columns: List of columns that contain dates
        date_formats: List of date formats to try parsing
    
    Returns:
        DataFrame with corrected types
    """
    print("\n--- Data Type Cleaning ---")
    
    # Explicit date columns specified by user
    if date_columns:
        for col in date_columns:
            if col not in df.columns:
                print(f"  ⚠ Date column '{col}' not found in DataFrame, skipping")
                continue
                
            print(f"\n  Processing date column: {col}")
            original_sample = df[col].head(3).tolist()
            print(f"    Sample values: {original_sample}")
            
            # FIX (2025-02-08): Wrap date parsing in try-except
            try:
                df[col] = parse_dates_smartly(df[col], date_formats)
                
                success_rate = df[col].notna().mean() * 100
                print(f"    → Converted to datetime ({success_rate:.1f}% success)")
                
                if success_rate < 90:
                    failed_samples = df[df[col].isna()][col].head(3).tolist()
                    print(f"    ⚠ Warning: Some dates failed to parse. Examples: {failed_samples}")
            except Exception as e:
                print(f"    ✗ Failed to convert: {e}")
    
    # Auto-detect date columns by name
    for col in df.columns:
        # Skip if already processed or in type_mapping
        if (date_columns and col in date_columns) or (type_mapping and col in type_mapping):
            continue
        
        # Look for date-related keywords in column name
        if any(keyword in col for keyword in ['date', 'time', 'created', 'updated', 'timestamp', 'dt']):
            print(f"\n  Auto-detected date column: {col}")
            original_sample = df[col].head(3).tolist()
            print(f"    Sample values: {original_sample}")
            
            try:
                df[col] = parse_dates_smartly(df[col], date_formats)
                
                success_rate = df[col].notna().mean() * 100
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    print(f"    → Converted to datetime ({success_rate:.1f}% success)")
                else:
                    print(f"    ✗ Failed to convert (keeping as-is)")
            except Exception as e:
                print(f"    ✗ Conversion failed: {e}")
        
        # Try to detect numeric columns
        if df[col].dtype == 'object':
            sample = df[col].dropna().astype(str).str.strip()
            if len(sample) > 0:
                # Try converting to numeric
                numeric_test = pd.to_numeric(sample, errors='coerce')
                if numeric_test.notna().mean() > 0.9:  # 90% are valid numbers
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Decide int vs float
                    if df[col].dropna().apply(lambda x: x == int(x)).all():
                        df[col] = df[col].astype('Int64')  # Nullable integer
                        print(f"  → {col}: converted to integer")
                    else:
                        print(f"  → {col}: converted to float")
                    continue
        
        # Detect boolean columns
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().str.lower().unique()
            if len(unique_vals) <= 2 and all(v in ['true', 'false', 'yes', 'no', '1', '0', 't', 'f'] for v in unique_vals):
                df[col] = df[col].str.lower().map({
                    'true': True, 'false': False,
                    'yes': True, 'no': False,
                    '1': True, '0': False,
                    't': True, 'f': False
                })
                print(f"  → {col}: converted to boolean")
    
    # Apply custom type mapping if provided
    if type_mapping:
        for col, dtype in type_mapping.items():
            if col not in df.columns:
                print(f"  ⚠ Column '{col}' in type_mapping not found, skipping")
                continue
                
            try:
                if dtype == 'datetime64':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype(dtype)
                print(f"  → {col}: manually set to {dtype}")
            except Exception as e:
                print(f"  ✗ Failed to convert {col} to {dtype}: {e}")
    
    return df


def parse_dates_smartly(series, date_formats=None):
    """
    Intelligently parse dates with multiple possible formats
    
    FIX (2025-02-08): Better error handling and fallback logic
    
    Args:
        series: pandas Series with date strings
        date_formats: List of date formats to try
    
    Returns:
        Series with datetime objects
    """
    # Common date formats to try
    if date_formats is None:
        date_formats = [
            '%d-%b-%Y',      # 22-Jul-2024
            '%d/%m/%Y',      # 22/07/2024
            '%Y-%m-%d',      # 2024-07-22
            '%m/%d/%Y',      # 07/22/2024 (US format)
            '%d-%m-%Y',      # 22-07-2024
            '%Y/%m/%d',      # 2024/07/22
            '%d.%m.%Y',      # 22.07.2024
            '%d %b %Y',      # 22 Jul 2024
            '%d %B %Y',      # 22 July 2024
            '%b %d, %Y',     # Jul 22, 2024
            '%B %d, %Y',     # July 22, 2024
            '%Y%m%d',        # 20240722
        ]
    
    # First try: Let pandas infer automatically
    try:
        result = pd.to_datetime(series, errors='coerce', dayfirst=True)
        success_rate = result.notna().mean()
        
        if success_rate > 0.95:  # 95% success is good enough
            return result
    except:
        pass
    
    # Second try: Try each format explicitly
    for fmt in date_formats:
        try:
            result = pd.to_datetime(series, format=fmt, errors='coerce')
            success_rate = result.notna().mean()
            
            if success_rate > 0.95:
                return result
        except:
            continue
    
    # Third try: Mixed format parsing (slower but handles variety)
    try:
        result = pd.to_datetime(series, errors='coerce', dayfirst=True)
        return result
    except:
        # FIX (2025-02-08): Return original series if all parsing fails
        return series


def validate_primary_key(df, primary_key_columns):
    """
    Validate that primary key columns are unique and not null
    
    FIX (2025-02-08): Better error reporting
    """
    print("\n--- Primary Key Validation ---")
    
    if not primary_key_columns:
        print("  ℹ No primary key specified")
        return True
    
    # Check for nulls in primary key
    for col in primary_key_columns:
        if col not in df.columns:
            print(f"  ✗ ERROR: Primary key column '{col}' not found!")
            return False
            
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  ✗ WARNING: {col} has {null_count} null values!")
            return False
    
    # Check for uniqueness
    duplicate_count = df.duplicated(subset=primary_key_columns).sum()
    if duplicate_count > 0:
        print(f"  ✗ WARNING: Found {duplicate_count} duplicate primary keys!")
        
        # FIX (2025-02-08): Show examples of duplicates
        dupes = df[df.duplicated(subset=primary_key_columns, keep=False)][primary_key_columns].head(5)
        print(f"  Examples of duplicate keys:")
        print(dupes.to_string(index=False))
        
        return False
    
    print(f"  ✓ Primary key {primary_key_columns} is valid (unique and not null)")
    return True


def filter_unwanted_rows(df, filter_rules=None):
    """
    Filter out unwanted rows based on specific conditions
    
    FIX (2025-02-08): Case-insensitive matching and better logging
    
    Args:
        df: DataFrame
        filter_rules: Dict of column name to values/conditions to exclude
                     Examples:
                     {'state': ['Test Systems', 'Test']}
                     {'status': ['Deleted', 'Invalid']}
    
    Returns:
        Tuple of (filtered DataFrame, count of rows removed)
    """
    if not filter_rules:
        return df, 0
    
    print("\n--- Filtering Unwanted Rows ---")
    initial_count = len(df)
    
    for column, exclude_values in filter_rules.items():
        if column not in df.columns:
            print(f"  ⚠ Column '{column}' not found, skipping filter")
            continue
        
        # Convert single value to list for consistency
        if not isinstance(exclude_values, list):
            exclude_values = [exclude_values]
        
        # Count rows before filtering
        before = len(df)
        
        # Filter out rows (case-insensitive matching for text)
        if df[column].dtype == 'object':
            # FIX (2025-02-08): Better string comparison
            mask = ~df[column].astype(str).str.lower().isin([str(v).lower() for v in exclude_values])
            df = df[mask]
        else:
            # Exact match for non-string columns
            df = df[~df[column].isin(exclude_values)]
        
        removed = before - len(df)
        if removed > 0:
            print(f"  ✓ Removed {removed} rows where {column} in {exclude_values}")
    
    total_removed = initial_count - len(df)
    
    if total_removed > 0:
        print(f"\n  → Total filtered: {total_removed} rows removed")
    else:
        print(f"  ✓ No rows matched filter criteria")
    
    return df, total_removed


def filter_by_conditions(df, custom_filters=None):
    """
    Advanced filtering with custom conditions
    
    Args:
        df: DataFrame
        custom_filters: List of filter functions or lambda expressions
                       Examples:
                       [lambda x: x['amount'] > 0]
    
    Returns:
        Filtered DataFrame
    """
    if not custom_filters:
        return df
    
    print("\n--- Applying Custom Filters ---")
    initial_count = len(df)
    
    for i, filter_func in enumerate(custom_filters, 1):
        try:
            before = len(df)
            df = df[filter_func(df)]
            removed = before - len(df)
            if removed > 0:
                print(f"  ✓ Filter {i}: Removed {removed} rows")
        except Exception as e:
            print(f"  ✗ Filter {i} failed: {e}")
    
    total_removed = initial_count - len(df)
    if total_removed > 0:
        print(f"\n  → Total custom filtered: {total_removed} rows")
    
    return df


def add_pipeline_metadata(df, pipeline_version='1.0', source_file=None):
    """
    Add comprehensive pipeline metadata for tracking and debugging
    
    FIX (2025-02-08): Better timestamp handling
    
    Args:
        df: DataFrame
        pipeline_version: Version of your cleaning pipeline
        source_file: Original source file name
    
    Returns:
        DataFrame with metadata columns
    """
    print("\n--- Adding Pipeline Metadata ---")
    
    # Ingestion timestamp
    ingestion_time = datetime.now()
    df['ingestion_timestamp'] = ingestion_time
    print(f"  ✓ ingestion_timestamp: {ingestion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ingestion date (useful for daily partitioning in queries)
    df['ingestion_date'] = ingestion_time.date()
    print(f"  ✓ ingestion_date: {ingestion_time.date()}")
    
    # Pipeline version
    df['pipeline_version'] = pipeline_version
    print(f"  ✓ pipeline_version: {pipeline_version}")
    
    # Source file tracking
    if source_file:
        df['source_file'] = os.path.basename(source_file)
        print(f"  ✓ source_file: {os.path.basename(source_file)}")
    
    # Processing batch ID
    batch_id = ingestion_time.strftime('%Y%m%d_%H%M%S')
    df['batch_id'] = batch_id
    print(f"  ✓ batch_id: {batch_id}")
    
    return df


def generate_data_quality_report(df, output_dir='data/reports'):
    """
    Generate a data quality report
    
    FIX (2025-02-08): Better handling of numeric columns with NaN
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': {},
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
            'unique_values': int(df[col].nunique()),
        }
        
        # FIX (2025-02-08): Better handling of numeric stats
        if df[col].dtype in ['float64', 'int64', 'Int64']:
            try:
                col_info['min'] = float(df[col].min()) if pd.notna(df[col].min()) else None
                col_info['max'] = float(df[col].max()) if pd.notna(df[col].max()) else None
                col_info['mean'] = float(df[col].mean()) if pd.notna(df[col].mean()) else None
            except:
                # Skip if there's any issue calculating stats
                pass
        
        report['columns'][col] = col_info
    
    timestamp = datetime.now().strftime('%Y%m%d')
    report_path = f"{output_dir}/quality_report_{timestamp}.json"
    
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Data quality report saved: {report_path}")
    return report


def process_file(input_path, output_dir='data/processed', 
                 primary_key_columns=None,
                 dedup_strategy='keep_latest',
                 handle_upgrades=False,
                 name_column=None,
                 date_column=None,
                 type_mapping=None,
                 date_columns=None,
                 date_formats=None,
                 filter_rules=None,
                 custom_filters=None,
                 missing_value_strategy='default',
                 add_metadata=True,
                 pipeline_version='1.0'):
    """
    Complete data cleaning pipeline
    
    FIX (2025-02-08): Better error handling and progress tracking
    
    Returns:
        Tuple of (output_path, pipeline_stats)
    """
    import time
    start_time = time.time()
    
    # Initialize stats tracker
    stats = {
        'timestamp': datetime.now().isoformat(),
        'status': 'Running',
        'rows_loaded': 0,
        'rows_final': 0,
        'rows_filtered': 0,
        'rows_duplicates': 0,
        'rows_null_pk': 0,
        'columns_cleaned': 0,
        'date_columns': 0,
        'pipeline_version': pipeline_version,
        'source_file': os.path.basename(input_path),
        'output_file': '',
        'execution_time': 0,
        'error_message': '',
        'warnings': [],
    }
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print(f"CLEANING: {input_path}")
        print("=" * 60)
        
        # Load data
        df = pd.read_parquet(input_path)
        stats['rows_loaded'] = len(df)
        initial_columns = len(df.columns)
        print(f"\nLoaded {len(df)} rows, {len(df.columns)} columns")
        
        # Step 1: Add ingestion metadata
        if add_metadata:
            df = add_pipeline_metadata(df, pipeline_version, input_path)
        
        # Step 2: Filter unwanted rows
        rows_before_filter = len(df)
        if filter_rules:
            df, filtered_count = filter_unwanted_rows(df, filter_rules)
            stats['rows_filtered'] = filtered_count
        
        if custom_filters:
            df = filter_by_conditions(df, custom_filters)
            stats['rows_filtered'] += rows_before_filter - len(df)
        
        # Step 3: Standardize text
        df = standardize_text_columns(df)
        
        # Step 4: Fix data types
        df = fix_data_types(df, type_mapping, date_columns, date_formats)
        if date_columns:
            stats['date_columns'] = len(date_columns)
        
        # Step 5: Handle Primary Key Issues
        rows_before_pk = len(df)
        if primary_key_columns:
            if handle_upgrades and len(primary_key_columns) == 1:
                df = handle_upgrade_scenarios(
                    df, 
                    id_column=primary_key_columns[0],
                    name_column=name_column,
                    date_column=date_column
                )
            else:
                df = handle_primary_key_issues(df, primary_key_columns, dedup_strategy)
            
            stats['rows_null_pk'] = rows_before_pk - len(df)
        
        # Step 6: Remove exact duplicates
        df, dup_count = remove_duplicates(df, primary_key_columns=None)
        stats['rows_duplicates'] = dup_count
        
        # Step 7: Handle missing values
        df = handle_missing_values(df, missing_value_strategy)
        
        # Step 8: Validate primary key
        if primary_key_columns:
            is_valid = validate_primary_key(df, primary_key_columns)
            if not is_valid:
                stats['warnings'].append('Primary key validation failed')
        
        # Step 9: Generate quality report
        quality_report = generate_data_quality_report(df)
        
        # Collect warnings from quality report
        for col, col_info in quality_report.get('columns', {}).items():
            null_pct = col_info.get('null_percentage', 0)
            if null_pct > 10:
                stats['warnings'].append(f"{col} has {null_pct}% missing values")
        
        # Save cleaned data
        timestamp = datetime.now().strftime('%Y%m%d')
        output_filename = f"processed_{timestamp}.parquet"
        output_path = f"{output_dir}/{output_filename}"
        df.to_parquet(output_path, compression='snappy', index=False)
        
        stats['rows_final'] = len(df)
        stats['output_file'] = output_filename
        stats['status'] = 'Success'
        stats['execution_time'] = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"✓ CLEANING COMPLETE")
        print(f"✓ Output: {output_path}")
        print(f"✓ Final: {len(df)} rows, {len(df.columns)} columns")
        print(f"✓ Execution time: {stats['execution_time']:.2f} seconds")
        print("=" * 60)
        
        return output_path, stats
        
    except Exception as e:
        stats['status'] = 'Failed'
        stats['error_message'] = str(e)
        stats['execution_time'] = time.time() - start_time
        print(f"\n✗ ERROR: {e}")
        raise


def main():
    """
    Main function - configure your cleaning rules here
    """
    # Find the most recent raw file
    raw_files = glob.glob('data/raw/raw_*.parquet')
    
    if not raw_files:
        print("No raw files found!")
        return
    
    latest_raw = max(raw_files)
    
    # Load raw data for schema validation
    raw_df = pd.read_parquet(latest_raw)
    
    # PRIMARY KEY CONFIGURATION
    PRIMARY_KEY = ['device_id']
    
    # DEDUPLICATION STRATEGY
    DEDUP_STRATEGY = 'keep_latest'
    
    # UPGRADE SCENARIO HANDLER
    HANDLE_UPGRADES = True
    NAME_COLUMN = 'customer_name'
    DATE_COLUMN = 'commissioning_date'
    
    # Data types
    TYPE_MAPPING = {}
    
    # Date columns
    DATE_COLUMNS = [
        'commissioning_date',
        'maintenance_sub_start_date',
        'maintenance_sub_end_date',
        'last_maintenance_date',
        'h12025_maintenance_date',
        'next_maint_date',
        'h22025_maintenance_date',
        'h12026_maintenance_date',
    ]
    
    DATE_FORMATS = [
        '%d-%b-%Y',
        '%d/%m/%Y',
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d-%m-%Y',
    ]
    
    # Filters
    FILTER_RULES = {
        'state': ['Test Systems', 'test', 'Test System'],
    }
    
    CUSTOM_FILTERS = []
    
    MISSING_STRATEGY = 'default'
    ADD_METADATA = True
    PIPELINE_VERSION = '1.0'
    UPLOAD_LOGS = True
    LOG_SHEET_URL = os.environ.get('OUTPUT_SHEET_URL')

    # Schema validation
    try:
        from schema_validator import handle_schema_changes
        
        config = {
            'primary_key_columns': PRIMARY_KEY,
            'date_columns': DATE_COLUMNS,
            'name_column': NAME_COLUMN,
            'date_column': DATE_COLUMN,
        }
        
        config = handle_schema_changes(raw_df, config)
        
        PRIMARY_KEY = config['primary_key_columns']
        DATE_COLUMNS = config['date_columns']
        NAME_COLUMN = config['name_column']
        DATE_COLUMN = config['date_column']
        
    except ImportError:
        print("⚠️  schema_validator.py not found - skipping schema validation")
    except Exception as e:
        print(f"⚠️  Schema validation warning: {e}")
    
    # Run processing
    output_path, stats = process_file(
        latest_raw,
        primary_key_columns=PRIMARY_KEY,
        dedup_strategy=DEDUP_STRATEGY,
        handle_upgrades=HANDLE_UPGRADES,
        name_column=NAME_COLUMN,
        date_column=DATE_COLUMN,
        type_mapping=TYPE_MAPPING,
        date_columns=DATE_COLUMNS,
        date_formats=DATE_FORMATS,
        filter_rules=FILTER_RULES,
        custom_filters=CUSTOM_FILTERS,
        missing_value_strategy=MISSING_STRATEGY,
        add_metadata=ADD_METADATA,
        pipeline_version=PIPELINE_VERSION,
    )

    # Upload pipeline log
    if UPLOAD_LOGS and LOG_SHEET_URL:
        try:
            from upload_pipeline_logs import upload_pipeline_log
            upload_pipeline_log(LOG_SHEET_URL, stats)
        except Exception as e:
            print(f"\n⚠ Failed to upload log to Google Sheets: {e}")
    else:
        print("\n⚠ Google Sheets logging is disabled or URL not provided")


if __name__ == "__main__":
    main()