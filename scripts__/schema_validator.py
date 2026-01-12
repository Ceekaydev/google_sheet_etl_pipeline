"""
schema_validator.py
Detects schema changes and ensures pipeline compatibility
"""
import pandas as pd
import json
import os
from datetime import datetime

SCHEMA_FILE = 'data/schema_history.json'

def save_schema(df, schema_version='auto'):
    """
    Save current schema for future comparison
    
    Args:
        df: DataFrame
        schema_version: Version identifier (auto-generates if 'auto')
    """
    os.makedirs('data', exist_ok=True)
    
    if schema_version == 'auto':
        schema_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    schema = {
        'version': schema_version,
        'timestamp': datetime.now().isoformat(),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': df.shape,
    }
    
    # Load existing schemas
    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE, 'r') as f:
            schemas = json.load(f)
    else:
        schemas = {'schemas': []}
    
    # Add new schema
    schemas['schemas'].append(schema)
    schemas['latest'] = schema
    
    # Save
    with open(SCHEMA_FILE, 'w') as f:
        json.dump(schemas, f, indent=2)
    
    print(f"✓ Schema saved: {len(schema['columns'])} columns")
    return schema

def detect_schema_changes(df):
    """
    Compare current dataframe schema with saved schema
    
    Returns:
        dict with changes detected
    """
    if not os.path.exists(SCHEMA_FILE):
        print("ℹ No previous schema found - saving current schema")
        save_schema(df)
        return {'is_first_run': True, 'has_changes': False}
    
    # Load previous schema
    with open(SCHEMA_FILE, 'r') as f:
        schemas = json.load(f)
    
    prev_schema = schemas.get('latest', {})
    prev_columns = set(prev_schema.get('columns', []))
    current_columns = set(df.columns)
    
    # Detect changes
    changes = {
        'is_first_run': False,
        'has_changes': False,
        'new_columns': list(current_columns - prev_columns),
        'removed_columns': list(prev_columns - current_columns),
        'renamed_columns': [],
        'dtype_changes': {},
    }
    
    # Check for dtype changes in existing columns
    prev_dtypes = prev_schema.get('dtypes', {})
    for col in current_columns & prev_columns:
        current_dtype = str(df[col].dtype)
        prev_dtype = prev_dtypes.get(col, '')
        if current_dtype != prev_dtype:
            changes['dtype_changes'][col] = {
                'old': prev_dtype,
                'new': current_dtype
            }
    
    # Mark if there are any changes
    changes['has_changes'] = bool(
        changes['new_columns'] or 
        changes['removed_columns'] or 
        changes['dtype_changes']
    )
    
    return changes

def validate_required_columns(df, config):
    """
    Check if required columns exist in dataframe
    Handles both single values and lists
    
    Args:
        df: DataFrame
        config: Pipeline configuration dict
    
    Returns:
        dict with validation results
    """
    validation = {
        'is_valid': True,
        'missing_columns': {},
        'warnings': [],
    }
    
    # Check primary key columns (list)
    if config.get('primary_key_columns'):
        for col in config['primary_key_columns']:
            if col and col not in df.columns:
                validation['is_valid'] = False
                validation['missing_columns'][f'primary_key:{col}'] = col
                validation['warnings'].append(f"Primary key column '{col}' not found!")
    
    # Check date columns (list)
    if config.get('date_columns'):
        for col in config['date_columns']:
            if col and col not in df.columns:
                validation['is_valid'] = False
                validation['missing_columns'][f'date_column:{col}'] = col
                validation['warnings'].append(f"Date column '{col}' not found!")
    
    # Check name column (single value)
    if config.get('name_column'):
        col = config['name_column']
        if col and col not in df.columns:
            # This is not critical - just a warning
            validation['warnings'].append(f"Name column '{col}' not found (upgrade detection disabled)")
    
    # Check date reference column (single value)
    if config.get('date_column'):
        col = config['date_column']
        if col and col not in df.columns:
            # This is not critical - just a warning
            validation['warnings'].append(f"Date reference column '{col}' not found (may affect deduplication)")
    
    return validation

def suggest_column_mapping(df, old_column_name):
    """
    Suggest possible new column names based on similarity
    Uses fuzzy matching to find renamed columns
    
    Args:
        df: DataFrame
        old_column_name: The old column name to find
    
    Returns:
        list of suggested column names
    """
    from difflib import get_close_matches
    
    suggestions = get_close_matches(
        old_column_name.lower(), 
        [col.lower() for col in df.columns],
        n=3,
        cutoff=0.6
    )
    
    # Map back to original case
    result = []
    for suggestion in suggestions:
        for col in df.columns:
            if col.lower() == suggestion:
                result.append(col)
                break
    
    return result

def auto_map_renamed_columns(df, config):
    """
    Automatically map renamed columns using fuzzy matching
    Handles both single values and lists
    
    Args:
        df: DataFrame
        config: Pipeline configuration dict
    
    Returns:
        Updated config with corrected column names
    """
    updated_config = config.copy()
    changes_made = []
    
    # Handle PRIMARY_KEY (list of columns)
    if config.get('primary_key_columns'):
        pk_list = config['primary_key_columns']
        new_pk_list = []
        
        for old_col in pk_list:
            if old_col not in df.columns:
                suggestions = suggest_column_mapping(df, old_col)
                if suggestions:
                    new_col = suggestions[0]
                    new_pk_list.append(new_col)
                    changes_made.append(f"primary_key: '{old_col}' → '{new_col}'")
                else:
                    new_pk_list.append(old_col)  # Keep old name (will fail validation)
            else:
                new_pk_list.append(old_col)  # Column exists, keep it
        
        updated_config['primary_key_columns'] = new_pk_list
    
    # Handle DATE_COLUMNS (list of columns)
    if config.get('date_columns'):
        date_list = config['date_columns']
        new_date_list = []
        
        for old_col in date_list:
            if old_col not in df.columns:
                suggestions = suggest_column_mapping(df, old_col)
                if suggestions:
                    new_col = suggestions[0]
                    new_date_list.append(new_col)
                    changes_made.append(f"date_column: '{old_col}' → '{new_col}'")
                else:
                    new_date_list.append(old_col)  # Keep old name
            else:
                new_date_list.append(old_col)  # Column exists
        
        updated_config['date_columns'] = new_date_list
    
    # Handle NAME_COLUMN (single value)
    if config.get('name_column'):
        old_col = config['name_column']
        if old_col not in df.columns:
            suggestions = suggest_column_mapping(df, old_col)
            if suggestions:
                new_col = suggestions[0]
                updated_config['name_column'] = new_col
                changes_made.append(f"name_column: '{old_col}' → '{new_col}'")
    
    # Handle DATE_COLUMN (single value - reference date for deduplication)
    if config.get('date_column'):
        old_col = config['date_column']
        if old_col not in df.columns:
            suggestions = suggest_column_mapping(df, old_col)
            if suggestions:
                new_col = suggestions[0]
                updated_config['date_column'] = new_col
                changes_made.append(f"date_column: '{old_col}' → '{new_col}'")
    
    # Handle FILTER_RULES (dict with column names as keys)
    if config.get('filter_rules'):
        old_filter_rules = config['filter_rules']
        new_filter_rules = {}
        
        for old_col, filter_values in old_filter_rules.items():
            if old_col not in df.columns:
                suggestions = suggest_column_mapping(df, old_col)
                if suggestions:
                    new_col = suggestions[0]
                    new_filter_rules[new_col] = filter_values
                    changes_made.append(f"filter_column: '{old_col}' → '{new_col}'")
                else:
                    new_filter_rules[old_col] = filter_values  # Keep old name
            else:
                new_filter_rules[old_col] = filter_values  # Column exists
        
        updated_config['filter_rules'] = new_filter_rules
    
    # Print changes if any were made
    if changes_made:
        print("\n  🔄 Auto-mapped columns:")
        for change in changes_made:
            print(f"     {change}")
    
    return updated_config, changes_made

def handle_schema_changes(df, config):
    """
    Main function to handle schema changes gracefully
    Properly handles both single values and lists
    
    Args:
        df: DataFrame
        config: Pipeline configuration dict with expected columns
    
    Returns:
        Updated config with corrected column names, or raises error if critical columns missing
    """
    print("\n" + "=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)
    
    # Detect changes
    changes = detect_schema_changes(df)
    
    if changes['is_first_run']:
        print("✓ First run - schema baseline created")
        return config
    
    if not changes['has_changes']:
        print("✓ No schema changes detected")
        return config
    
    # Report changes
    print("\n⚠️  SCHEMA CHANGES DETECTED:")
    
    if changes['new_columns']:
        print(f"\n  📌 New columns ({len(changes['new_columns'])}):")
        for col in changes['new_columns']:
            print(f"     + {col}")
    
    if changes['removed_columns']:
        print(f"\n  ❌ Removed columns ({len(changes['removed_columns'])}):")
        for col in changes['removed_columns']:
            print(f"     - {col}")
    
    if changes['dtype_changes']:
        print(f"\n  🔄 Data type changes ({len(changes['dtype_changes'])}):")
        for col, change in changes['dtype_changes'].items():
            print(f"     {col}: {change['old']} → {change['new']}")
    
    # Try auto-mapping renamed columns
    updated_config, mappings = auto_map_renamed_columns(df, config)
    
    # Validate that all required columns exist (after mapping)
    validation = validate_required_columns(df, updated_config)
    
    if not validation['is_valid']:
        print("\n" + "=" * 60)
        print("❌ CRITICAL: Required columns missing!")
        print("=" * 60)
        
        for purpose_col, col_name in validation['missing_columns'].items():
            purpose = purpose_col.split(':')[0]
            print(f"\n  Missing: '{col_name}' ({purpose})")
            suggestions = suggest_column_mapping(df, col_name)
            
            if suggestions:
                print(f"  Possible matches:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"    {i}. {suggestion}")
                
                # Show config update example
                if 'primary_key' in purpose:
                    print(f"\n  💡 Update your config:")
                    print(f"     PRIMARY_KEY = ['{suggestions[0]}']")
                elif 'date_column' in purpose:
                    print(f"\n  💡 Update your config:")
                    print(f"     DATE_COLUMNS = ['{suggestions[0]}']")
            else:
                print(f"  ⚠️  No similar columns found in data")
        
        print("\n" + "=" * 60)
        raise ValueError(f"Pipeline cannot continue: Required columns missing. Check config!")
    
    # Show warnings for non-critical columns
    if validation['warnings']:
        print("\n  ⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"     {warning}")
    
    # Save new schema
    save_schema(df)
    
    print("\n" + "=" * 60)
    print("✓ SCHEMA VALIDATION COMPLETE")
    if mappings:
        print(f"✓ Auto-corrected {len(mappings)} column name(s)")
    print("=" * 60)
    
    return updated_config

def main():
    """Test function"""
    # Example usage
    test_df = pd.DataFrame({
        'device_id': [1, 2, 3],
        'customer_name': ['A', 'B', 'C'],
        'system_size': [10, 20, 30],
        'new_column': ['x', 'y', 'z']  # New column added
    })
    
    test_config = {
        'primary_key_columns': ['device_id'],
        'date_columns': ['installation_date'],
        'name_column': 'customer_name',
    }
    
    updated_config = handle_schema_changes(test_df, test_config)
    print(f"\nUpdated config: {updated_config}")

if __name__ == "__main__":
    main()