
"""
clean_data.py
Comprehensive data cleaning: column names, data types, duplicates, validation
"""

import pandas as pd
from datetime import datetime
import os
import glob
import numpy as np
from dotenv import load_dotenv

load_dotenv()



def standardize_text_columns(df):
    """Standardize text: trim whitespace, fix encoding issues"""
    print("\n--- Text Standardization ---")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Replace multiple spaces with single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove common encoding issues
            df[col] = df[col].str.replace('â€™', "'", regex=False)
            df[col] = df[col].str.replace('â€œ', '"', regex=False)
            df[col] = df[col].str.replace('â€', '"', regex=False)
            
            print(f"  ✓ {col}: standardized")
    
    return df

def handle_missing_values(df, strategy='default'):
    """
    Handle missing values
    
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
                    df[col].fillna(df[col].median(), inplace=True)
                    print(f"  → {col}: filled with median")
                elif df[col].dtype == 'object':
                    # Fill categorical with mode or 'Unknown'
                    mode_val = df[col].mode()
                    df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 'Unknown', inplace=True)
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                        print(f"  → {col}: filled with mode")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        print(f"  → {col}: filled with 'Unknown'")
    
    return df



def handle_primary_key_issues(df, primary_key_columns, strategy='keep_latest'):
    """
    Handle primary key violations: nulls and duplicates
    
    Args:
        df: DataFrame
        primary_key_columns: List of PK columns (e.g., ['device_id'])
        strategy: How to handle duplicates:
            - 'keep_latest': Keep the most recent record (based on row order or date)
            - 'keep_first': Keep the first occurrence
            - 'keep_largest': Keep record with most non-null values
            - 'flag_duplicates': Keep all but add a flag column
            - 'remove_all': Remove all duplicates (keep none)
    
    Returns:
        Cleaned DataFrame
    """
    if not primary_key_columns:
        return df
    
    print("\n--- Primary Key Issue Resolution ---")
    initial_count = len(df)
    
    # Step 1: Handle NULL values in primary key
    for col in primary_key_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  ⚠ Found {null_count} NULL values in '{col}'")
            
            # Option 1: Drop rows with null PK
            df = df.dropna(subset=[col])
            print(f"    → Removed {null_count} rows with NULL {col}")
    
    # Step 2: Handle duplicate primary keys
    duplicate_mask = df.duplicated(subset=primary_key_columns, keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        print(f"  ⚠ Found {duplicate_count} rows with duplicate {primary_key_columns}")
        
        # Show examples of duplicates
        duplicate_examples = df[duplicate_mask].head(5)
        print(f"\n  Example duplicates:")
        for pk_col in primary_key_columns:
            dup_ids = duplicate_examples[pk_col].unique()[:3]
            print(f"    {pk_col}: {list(dup_ids)}")
        
        if strategy == 'keep_latest':
            # Keep the last occurrence (assumes newer records come later)
            print(f"\n  Strategy: Keeping LATEST record for each duplicate")
            df = df.drop_duplicates(subset=primary_key_columns, keep='last')
            
        elif strategy == 'keep_first':
            print(f"\n  Strategy: Keeping FIRST record for each duplicate")
            df = df.drop_duplicates(subset=primary_key_columns, keep='first')
            
        elif strategy == 'keep_largest':
            # Keep record with most non-null values
            print(f"\n  Strategy: Keeping record with MOST data")
            
            def keep_best_record(group):
                # Count non-null values per row
                non_null_counts = group.notna().sum(axis=1)
                # Return row with most non-null values
                return group.loc[non_null_counts.idxmax()]
            
            df = df.groupby(primary_key_columns, dropna=False).apply(keep_best_record).reset_index(drop=True)
            
        elif strategy == 'flag_duplicates':
            # Keep all duplicates but add flag column
            print(f"\n  Strategy: Flagging duplicates (keeping all)")
            df['is_duplicate'] = df.duplicated(subset=primary_key_columns, keep=False)
            duplicate_count = 0  # Don't count as removed
            
        elif strategy == 'remove_all':
            # Remove ALL duplicates (keep none)
            print(f"\n  Strategy: Removing ALL duplicate records")
            df = df[~duplicate_mask]
        
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
            group = group.sort_values(date_column, ascending=False)
            best_record = group.iloc[0]
            
        # Strategy 2: If there's an "upgrade" keyword, keep that one
        elif name_column and name_column in df.columns:
            upgrade_mask = group[name_column].astype(str).str.contains('upgrade|updated|new', case=False, na=False)
            if upgrade_mask.any():
                best_record = group.loc[upgrade_mask].iloc[-1]  # Last upgrade
            else:
                # Keep the one with most non-null values
                non_null_counts = group.notna().sum(axis=1)
                best_record = group.loc[non_null_counts.idxmax()]
        
        # Strategy 3: Keep record with most complete data
        else:
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

def remove_duplicates(df, primary_key_columns=None, keep='first'):
    """
    Remove duplicate rows (SIMPLE VERSION - for non-PK duplicates)
    
    Args:
        df: DataFrame
        primary_key_columns: Not used here (handled by handle_primary_key_issues)
        keep: 'first', 'last', or False
    
    Returns:
        Cleaned DataFrame and count of duplicates removed
    """
    initial_count = len(df)
    
    # Remove exact duplicate rows (all columns identical)
    df = df.drop_duplicates(keep=keep)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"\n--- Exact Duplicate Removal ---")
        print(f"  ⚠ Removed {removed} exact duplicate rows (all columns identical)")
    
    return df, removed


def fix_data_types(df, type_mapping=None, date_columns=None, date_formats=None):
    """
    Automatically detect and fix data types with enhanced date parsing
    
    Args:
        df: DataFrame
        type_mapping: Optional dict like {'column_name': 'int64', 'date': 'datetime64'}
        date_columns: List of columns that contain dates (e.g., ['transaction_date', 'created_at'])
        date_formats: List of date formats to try parsing
    
    Returns:
        DataFrame with corrected types
    """
    print("\n--- Data Type Cleaning ---")
    
    # Explicit date columns specified by user
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                print(f"\n  Processing date column: {col}")
                original_sample = df[col].head(3).tolist()
                print(f"    Sample values: {original_sample}")
                
                df[col] = parse_dates_smartly(df[col], date_formats)
                
                success_rate = df[col].notna().mean() * 100
                print(f"    → Converted to datetime ({success_rate:.1f}% success)")
                
                if success_rate < 90:
                    failed_samples = df[df[col].isna()][col].head(3).tolist()
                    print(f"    ⚠ Warning: Some dates failed to parse. Examples: {failed_samples}")
    
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
            
            df[col] = parse_dates_smartly(df[col], date_formats)
            
            success_rate = df[col].notna().mean() * 100
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"    → Converted to datetime ({success_rate:.1f}% success)")
            else:
                print(f"    ✗ Failed to convert (keeping as-is)")
        
        # Try to detect numeric columns (if they contain only numbers or empty)
        if df[col].dtype == 'object':
            # Check if it's numeric stored as string
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
            if col in df.columns:
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
    
    Args:
        series: pandas Series with date strings
        date_formats: List of date formats to try (e.g., ['%d-%b-%Y', '%Y-%m-%d'])
    
    Returns:
        Series with datetime objects
    """
    # Common date formats to try (in order of likelihood)
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
        return series

def remove_duplicates(df, primary_key_columns=None, keep='first'):
    """
    Remove duplicate rows
    
    Args:
        df: DataFrame
        primary_key_columns: List of columns that should be unique (e.g., ['id', 'transaction_id'])
                            If None, checks all columns
        keep: 'first', 'last', or False (remove all duplicates)
    
    Returns:
        Cleaned DataFrame and count of duplicates removed
    """

    """Remove exact duplicate rows only (all columns identical)"""
    initial_count = len(df)
    df = df.drop_duplicates(keep=keep)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"⚠ Removed {removed} exact duplicate rows (all columns identical)")
    else:
        print(f"✓ No exact duplicates found")
    
    return df, removed

def validate_primary_key(df, primary_key_columns):
    """
    Validate that primary key columns are unique and not null
    """
    print("\n--- Primary Key Validation ---")
    
    if not primary_key_columns:
        print("  ℹ No primary key specified")
        return True
    
    # Check for nulls in primary key
    for col in primary_key_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  ✗ WARNING: {col} has {null_count} null values!")
            return False
    
    # Check for uniqueness
    duplicate_count = df.duplicated(subset=primary_key_columns).sum()
    if duplicate_count > 0:
        print(f"  ✗ WARNING: Found {duplicate_count} duplicate primary keys!")
        return False
    
    print(f"  ✓ Primary key {primary_key_columns} is valid (unique and not null)")
    return True

            
        

def filter_unwanted_rows(df, filter_rules=None):
    """
    Filter out unwanted rows based on specific conditions
    
    Args:
        df: DataFrame
        filter_rules: Dict of column name to values/conditions to exclude
                     Examples:
                     {'state': ['Test Systems', 'Test']}
                     {'status': ['Deleted', 'Invalid']}
                     {'customer_type': 'Internal Test'}
    
    Returns:
        Filtered DataFrame and count of rows removed
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
            # String comparison - case insensitive
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
                       [lambda x: ~x['name'].str.contains('test', case=False)]
    
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


def add_ingestion_metadata(df, timestamp_column='ingestion_timestamp', include_date_only=False):
    """
    Add metadata columns to track when data was ingested
    
    Args:
        df: DataFrame
        timestamp_column: Name of the timestamp column to add
        include_date_only: If True, also add a date-only column
    
    Returns:
        DataFrame with metadata columns added
    """
    print("\n--- Adding Ingestion Metadata ---")
    
    # Add full timestamp
    ingestion_time = datetime.now()
    df[timestamp_column] = ingestion_time
    print(f"  ✓ Added '{timestamp_column}': {ingestion_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Optionally add date-only column (useful for partitioning)
    if include_date_only:
        date_column = 'ingestion_date'
        df[date_column] = ingestion_time.date()
        print(f"  ✓ Added '{date_column}': {ingestion_time.date()}")
    
    return df


def add_pipeline_metadata(df, pipeline_version='1.0', source_file=None):
    """
    Add comprehensive pipeline metadata for tracking and debugging
    
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
    
    # Pipeline version (helps track which cleaning rules were applied)
    df['pipeline_version'] = pipeline_version
    print(f"  ✓ pipeline_version: {pipeline_version}")
    
    # Source file tracking
    if source_file:
        df['source_file'] = os.path.basename(source_file)
        print(f"  ✓ source_file: {os.path.basename(source_file)}")
    
    # Processing batch ID (useful for debugging)
    batch_id = ingestion_time.strftime('%Y%m%d_%H%M%S')
    df['batch_id'] = batch_id
    print(f"  ✓ batch_id: {batch_id}")
    
    return df

def generate_data_quality_report(df, output_dir='data/reports'):
    """Generate a data quality report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': {},
    }
    
    for col in df.columns:
        report['columns'][col] = {
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
            'unique_values': int(df[col].nunique()),
        }
        
        if df[col].dtype in ['float64', 'int64', 'Int64']:
            report['columns'][col]['min'] = float(df[col].min()) if pd.notna(df[col].min()) else None
            report['columns'][col]['max'] = float(df[col].max()) if pd.notna(df[col].max()) else None
            report['columns'][col]['mean'] = float(df[col].mean()) if pd.notna(df[col].mean()) else None
    
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
    
    Args:
        input_path: Path to raw parquet file
        output_dir: Where to save cleaned data
        primary_key_columns: List of columns that form primary key (e.g., ['device_id'])
        dedup_strategy: 'keep_latest', 'keep_first', 'keep_largest', 'flag_duplicates', 'remove_all'
        handle_upgrades: If True, uses special upgrade scenario handler
        name_column: Column that might contain "(upgrade)" (e.g., 'customer_name')
        date_column: Date column to identify latest record (e.g., 'installation_date')
        type_mapping: Dict of column name to desired data type
        date_columns: List of columns containing dates
        date_formats: List of date formats to try
        filter_rules: Dict of columns and values to exclude
        custom_filters: List of lambda functions for advanced filtering
        missing_value_strategy: 'default', 'drop', or 'fill'
        add_metadata: If True, adds ingestion timestamp and pipeline metadata
        pipeline_version: Version string for tracking (e.g., '1.0', '2.1')
    
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
        rows_before_dates = len(df)
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
    
    # ========================================
    # CONFIGURE YOUR DATA CLEANING HERE
    # ========================================

    raw_df = pd.read_parquet(latest_raw)
    
    # PRIMARY KEY CONFIGURATION
    PRIMARY_KEY = ['device_id']  # Your actual primary key
    
    # DEDUPLICATION STRATEGY
    # Choose one: 'keep_latest', 'keep_first', 'keep_largest', 'flag_duplicates', 'remove_all'
    DEDUP_STRATEGY = 'keep_latest'  # Keeps the most recent record
    
    # UPGRADE SCENARIO HANDLER (for your specific case!)
    HANDLE_UPGRADES = True  # Set to True to use smart upgrade detection
    NAME_COLUMN = 'customer_name'  # Column that might have "(upgrade)" suffix
    DATE_COLUMN = 'commissioning_date'  # Date column to determine latest record
    
    # Define specific data types for columns
    TYPE_MAPPING = {
        # 'inverter_capacity': 'Int64',
        # 'panel_rating': 'Int64',
        # 'installed_generation_capacity': 'float64',
        # 'installed_storage_capacity': 'float64',
        # 'year_of_installation': 'Int64',
    }
    
    # Explicitly specify date columns
    DATE_COLUMNS = [
        'commissioning_date',
        'maintenance_sub_start_date',
        'maintenance_sub_end_date',
        'maintenance_sub_end_date',
        'last_maintenance_date',
        'h12025_maintenance_date',
        'next_maint_date',
        'h22025_maintenance_date',
        'h12026_maintenance_date',
    ]
    
    # Date formats to try (in order of priority)
    DATE_FORMATS = [
        '%d-%b-%Y',      # 22-Jul-2024 (YOUR FORMAT)
        '%d/%m/%Y',      # 22/07/2024
        '%Y-%m-%d',      # 2024-07-22
        '%m/%d/%Y',      # 07/22/2024
        '%d-%m-%Y',      # 22-07-2024
    ]
    
    # FILTER UNWANTED ROWS
    FILTER_RULES = {
        'state': ['Test Systems', 'test', 'Test System'],
        # Add more filters as needed
    }
    
    # ADVANCED CUSTOM FILTERS (Optional)
    CUSTOM_FILTERS = [
        # lambda df: df['system_size'] > 0,  # Only positive values
    ]
    
    # Missing value strategy: 'default', 'drop', or 'fill'
    MISSING_STRATEGY = 'default'
    
    # INGESTION METADATA
    ADD_METADATA = True  # Add timestamp columns
    PIPELINE_VERSION = '1.0'  # Track which version of cleaning was used

    # GOOGLE SHEETS LOGGING
    UPLOAD_LOGS = True
    LOG_SHEET_URL = os.environ.get('OUTPUT_SHEET_URL')

    # ========================================
    # SCHEMA VALIDATION & AUTO-CORRECTION
    # ========================================
    
    try:
        from schema_validator import handle_schema_changes
        
        config = {
            'primary_key_columns': PRIMARY_KEY,
            'date_columns': DATE_COLUMNS,
            'name_column': NAME_COLUMN,
            'date_column': DATE_COLUMN,
        }
        
        # Validate schema and auto-correct column names if renamed
        config = handle_schema_changes(raw_df, config)
        
        # Update variables with corrected names
        PRIMARY_KEY = config['primary_key_columns']
        DATE_COLUMNS = config['date_columns']
        NAME_COLUMN = config['name_column']
        DATE_COLUMN = config['date_column']
        
    except ImportError:
        print("⚠️  schema_validator.py not found - skipping schema validation")
    except Exception as e:
        print(f"⚠️  Schema validation warning: {e}")
        # Continue anyway - let the pipeline try to run
    
    # ========================================
    
    # ========================================
    
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

    # Upload pipeline log to Google Sheets
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