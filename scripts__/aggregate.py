"""
aggregate_duckdb.py
Runs multiple SQL queries on parquet files using DuckDB
Each query produces a separate aggregation saved as individual parquet files
"""
import duckdb
import pandas as pd
from datetime import datetime
import os

def run_query(query_name, sql_query, parquet_pattern):
    """
    Execute a SQL query.
    parquet_pattern can be a glob path OR a specific file path.
    """
    con = duckdb.connect()
    
    try:
        # Check if parquet_pattern is already wrapped in quotes, if not, wrap it
        if not parquet_pattern.startswith("'"):
            parquet_path_sql = f"'{parquet_pattern}'"
        else:
            parquet_path_sql = parquet_pattern

        # Replace placeholder
        sql_query = sql_query.replace('{{parquet_files}}', parquet_path_sql)
        
        print(f"\nRunning query: {query_name}")
        result = con.execute(sql_query).df()
        print(f"  → Returned {len(result)} rows, {len(result.columns)} columns")
        
        con.close()
        return result
    
    except Exception as e:
        print(f"  ✗ Query failed: {e}")
        con.close()
        return pd.DataFrame()
    
def save_query_result(df, query_name, output_dir='data/aggregated'):
    """
    Save query result as individual parquet file
    Naming: aggregated_{query_name}_{date}.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d')
    output_path = f"{output_dir}/{query_name}_{timestamp}.parquet"
    
    df.to_parquet(output_path, compression='snappy', index=False)
    print(f"  → Saved to {output_path}")
    
    return output_path


# =============================================================================
# DEFINE YOUR QUERIES HERE
# Each query will become a separate worksheet in Google Sheets
# =============================================================================

QUERIES = {
    # Query 1: Customer without H2 PPM in Lagos
    "customers_without_h2_ppm_lagos": """
        SELECT 
            device_id, customer_name, customer_address, area_cluster, customer_phone_number, maintenance_service_plan, commissioning_date, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id

        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date IS NULL 
        AND State = 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 2: Customer without H2 PPM outside Lagos
    "customers_without_h2_ppm_outside_lagos": """
        SELECT 
            device_id, customer_name, customer_address, customer_phone_number,  maintenance_service_plan, commissioning_date, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id

        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date IS NULL 
        AND State != 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 3: Customer for January PPM in Lagos
    "January_ppm_2026(lagos)": """
        SELECT 
            device_id, customer_name, customer_address, area_cluster, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id

        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-07-01'
            AND h22025_maintenance_date < '2025-08-01'
            AND h12026_maintenance_date IS NULL 
        AND State = 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,
    # Query 4: Customer for January PPM outside Lagos
    "January_ppm_2026(outside_lagos)": """
        SELECT 
            device_id, customer_name, customer_address, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id

        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-07-01'
          AND h22025_maintenance_date < '2025-08-01'
          AND h12026_maintenance_date IS NULL
        AND State != 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 5: Customer for February PPM in Lagos
    "February_ppm_2026(lagos)": """
        SELECT 
            device_id, customer_name, customer_address, area_cluster, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id

        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-08-01'
            AND h22025_maintenance_date < '2025-09-01'
            AND h12026_maintenance_date IS NULL
        AND State = 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 6: Customer for February PPM outside Lagos
    "February_ppm_2026(outside_lagos)": """
        SELECT 
            device_id, customer_name, customer_address, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-08-01'
            AND h22025_maintenance_date < '2025-09-01'
            AND h12026_maintenance_date IS NULL
        AND State != 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 7: Customer for March PPM in Lagos
    "March_ppm_2026(lagos)": """
        SELECT 
            device_id, customer_name, customer_address, area_cluster, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-09-01'
            AND h22025_maintenance_date < '2025-10-01'
            AND h12026_maintenance_date IS NULL
        AND State = 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,
    # Query 8: Customer for March PPM outside Lagos
    "March_ppm_2026(outside_lagos)": """
        SELECT 
            device_id, customer_name, customer_address, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-09-01'
            AND h22025_maintenance_date < '2025-10-01'
            AND h12026_maintenance_date IS NULL
        AND State != 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 9: Customer for April PPM outside Lagos
    "April_ppm_2026(lagos)": """
        SELECT 
            device_id, customer_name, customer_address, area_cluster, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-10-01'
            AND h22025_maintenance_date < '2025-11-01'
            AND h12026_maintenance_date IS NULL
        AND State = 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 10: Customer for April PPM outside Lagos
    "April_ppm_2026(outside_lagos)": """
        SELECT 
            device_id, customer_name, customer_address, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-10-01'
            AND h22025_maintenance_date < '2025-11-01'
            AND h12026_maintenance_date IS NULL
        AND State != 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 11: Customer for May PPM outside Lagos
    "May_ppm_2026(lagos)": """
        SELECT 
            device_id, customer_name, customer_address, area_cluster, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-11-01'
            AND h22025_maintenance_date < '2025-12-01'
            AND h12026_maintenance_date IS NULL
        AND State = 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 12: Customer for May PPM outside Lagos
    "May_ppm_2026(outside_lagos)": """
        SELECT 
            device_id, customer_name, customer_address, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-11-01'
            AND h22025_maintenance_date < '2025-12-01'
            AND h12026_maintenance_date IS NULL
        AND State != 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 13: Customer for June PPM outside Lagos
    "June_ppm_2026(lagos)": """
        SELECT 
            device_id, customer_name, customer_address, area_cluster, customer_phone_number, maintenance_service_plan, cluster, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-12-01'
            AND h22025_maintenance_date < '2026-01-01'
            AND h12026_maintenance_date IS NULL
        AND State = 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

    # Query 14: Customer for June PPM outside Lagos
    "June_ppm_2026(outside_lagos)": """
        SELECT 
            device_id, customer_name, customer_address, customer_phone_number, cluster, maintenance_service_plan, state, 
            lo_status, current_status, customer_profile, 
            h22025_maintenance_date AS last_maintenance, ingestion_timestamp,
            ingestion_date,
            pipeline_version, source_file, batch_id
        FROM read_parquet({{parquet_files}})
        WHERE h22025_maintenance_date >= '2025-12-01'
            AND h22025_maintenance_date < '2026-01-01'
            AND h12026_maintenance_date IS NULL
        AND State != 'Lagos'
            AND Current_Status = 'Active'
            AND (
                    (lo_status = 'OS' 
                    AND (maintenance_sub_expiration_duration IS NULL 
                        OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
                OR (lo_status = 'LO' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'IPP' AND free_maintenance_status = 'Active')
		        OR (lo_status = 'ES' AND free_maintenance_status = 'Active')
		        OR ( lo_status = 'LO (PAYMENT COMPLETE)'
		            AND (maintenance_sub_expiration_duration IS NULL 
			            OR maintenance_sub_expiration_duration LIKE '%Remaining%'))
        );
    """,

}

def run_all_queries(specific_input_path=None):
    """
    Execute all queries.
    Args:
        specific_input_path: If provided, runs queries ONLY on this file.
                             If None, defaults to pattern matching (risky).
    """
    results = {}
    
    # LOGIC CHANGE: Use the specific file passed from main.py if available
    if specific_input_path:
        # We wrap it in single quotes for SQL syntax
        parquet_source = f"'{specific_input_path}'"
        print("=" * 60)
        print(f"Running {len(QUERIES)} SQL queries on SPECIFIC FILE: {specific_input_path}")
        print("=" * 60)
    else:
        # Fallback (Old behavior) - RISKY as it might match multiple files
        today_str = datetime.now().strftime('%Y%m%d')
        # We use a glob pattern here
        parquet_source = f"'data/processed/processed_{today_str}*.parquet'"
        print("=" * 60)
        print(f"Running {len(QUERIES)} SQL queries on PATTERN: {parquet_source}")
        print("=" * 60)
    
    for query_name, sql_query in QUERIES.items():
        # Replace the placeholder in the SQL with the actual path/pattern
        # We pass the formatted string directly now
        
        # Note: We need to modify run_query slightly to handle this, 
        # or just do the replacement here before calling run_query.
        # Let's adjust run_query call:
        
        # Create a temporary version of the query with the file path injected
        final_sql = sql_query.replace('{{parquet_files}}', parquet_source)
        
        # We pass the final SQL directly to run_query, bypassing the internal replace logic
        # You will need to update run_query signature slightly or just rely on the fact 
        # that run_query does a replace on {{parquet_files}} which we handled.
        
        # ACTUALLY, simpler fix: Pass the raw pattern to run_query
        df = run_query(query_name, sql_query, parquet_pattern=specific_input_path if specific_input_path else f'data/processed/processed_{today_str}*.parquet')
        
        if not df.empty:
            output_path = save_query_result(df, query_name)
            results[query_name] = output_path
        else:
            print(f"  ✗ Skipping save (empty result)")
            results[query_name] = None
    
    print("\n" + "=" * 60)
    print(f"✓ Completed {len([v for v in results.values() if v])} queries")
    print("=" * 60)
    
    return results

def run_custom_query(query_name, sql_query):
    """
    Run a single custom query (useful for testing)
    """
    df = run_query(query_name, sql_query)
    
    if not df.empty:
        output_path = save_query_result(df, query_name)
        return output_path
    
    return None

def main():
    print("Running DuckDB aggregations...")
    results = run_all_queries()
    
    # Print summary
    print("\nQuery Results Summary:")
    for query_name, output_path in results.items():
        if output_path:
            print(f"  ✓ {query_name}: {output_path}")
        else:
            print(f"  ✗ {query_name}: Failed or empty")

if __name__ == "__main__":
    main()