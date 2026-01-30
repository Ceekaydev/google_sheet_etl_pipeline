"""
enrich_locations_fast.py
Zero-Wait Location Clustering.
Uses ONLY Logic and Keywords. No APIs. No Delays.
"""

import pandas as pd
import os
import glob  # <--- Used in the standalone block below

# ==========================================
# CONFIGURATION
# ==========================================
POSSIBLE_ADDRESS_COLS = ['address', 'home_address', 'customer_address', 'location', 'street']
POSSIBLE_STATE_COLS = ['state', 'customer_state', 'region', 'province']

AREA_KEYWORDS = {
    # --- ISLAND: HIGH BROW ---
    'Victoria Island': [
        'victoria island', 'v.i.', ' vi ', 'vi,', ', vi', 'adeola odeku', 'ozumba mbadiwe', 
        '1004', 'oniru', 'bishops gate', 'eko atlantic', 'adeleke adedoyin', 'elsie femi pearse'
    ],
    'Ikoyi': [
        'ikoyi', 'banana island', 'bourdillon', 'alexander', 'glover', 'parkview', 
        'dolphin estate', 'osborne', 'falomo', 'alfred rewane', 'awolowo road'
    ],
    'Lekki Phase 1': [
        'lekki phase 1', 'lekki phase i', 'lekki 1', 'admiralty way', 'freedom way', 
        'fola osibo', 'orange island', 'periwinkle'
    ],
    
    # --- ISLAND: LEKKI / AJAH AXIS ---
    'Ikate': ['ikate', 'elegushi', 'kusenla', 'nike art'],
    'Osapa London': ['osapa', 'agungi', 'igbo efon', 'idado', 'ajiran', 'western foreshore'],
    'Agungi/Ologolo': ['ologolo', 'agungi', 'meimunat shitta'],
    'Chevron/Orchid': ['chevron', 'orchid hotel', 'orchid road', 'carlton gate', 'lake view park'],
    'VGC': ['vgc', 'victoria garden city'],
    'Ikota': ['ikota', 'eleganza', 'mega chicken'],
    'Ajah': [
        'ajah', 'aja ', 'badore', 'addoh', 'langbasa', 'thomas estate', 'royal garden', 
        'abraham adesanya', 'edo line', 'oke-ira', 'oke ira', 'christian garden', 'u3 estate'
    ],
    'Sangotedo': ['sangotedo', 'shoprite sangotedo', 'crown estate', 'fara park', 'monastery', 'abijo', 'destiny homes', 'lufasi'],
    'Awoyaya': ['awoyaya', 'mayfair', 'meridian', 'greensprings'],
    'Lakowe': ['lakowe', 'golf estate', 'beechwood'],
    'Ibeju-Lekki': ['ibeju lekki', 'ibeju-lekki', 'eleko', 'dangote', 'free trade', 'bogije', 'amen estate', 'akodo', 'folu ise'],
    'Epe': ['epe town', 'epe,', 'alaro city', 'poka', 'mojoda', 'epe express', 'eredo', 'lagos road,t junction'],

    # --- MAINLAND: CENTRAL ---
    'Yaba': [
        'yaba', 'akoka', 'unilag', 'onike', 'sabo', 'adekunle', 'jibowu', 
        'tejuosho', 'herbert macaulay', 'alagomeji', 'makoko', 'erejuwa'
    ],
    'Ebute Metta': [
        'ebute metta', 'ebutte metta', 'ebute-metta', 'ebutte-metta', 'ebutte meta',
        'costain', 'brickfield', 'apapa road', 'otto', 'iddo', 'edegbe park', 'court road'
    ],
    'Surulere': [
        'surulere', 'suru-lere', 'bode thomas', 'adeniran ogunsanya', 'ojuelegba', 
        'lawanson', 'iganmu', 'stadium', 'eric moore', 'kilo', 'small london', 
        'itire', 'iponri', 'akerele', 'aguda', 'moshalashi street', 'alaka'
    ],
    'Shomolu/Bariga': [
        'shomolu', 'somolu', 'bariga', 'fadipe', 'bajulaiye', 'ilaje', 'palmgrove', 
        'onipanu', 'onipan', 'fadeyi', 'banga', 'akoka road'
    ],
    'Ikeja': ['ikeja', 'alausa', 'allen avenue', 'opebi', 'toyin street', 'computer village', 'g.r.a ikeja', 'obafemi awolowo'],
    'Maryland': ['maryland', 'mende', 'anthony village', 'idi iroko', 'anthony'],
    
    # --- MAINLAND: RESIDENTIAL NORTH ---
    'Magodo': ['magodo', 'shangisha', 'isheri', 'cmd road'],
    'Ogba': ['ogba', 'college road'],
    'Berger': ['berger', 'ojodu', 'omole', 'arepo', 'magboro', 'warewa', 'mowe', 'ibafo', 'wawa'],
    'Agege': ['agege', 'pen cinema', 'oko oba', 'oko-oba'],
    'Gbagada': ['gbagada', 'ifako gbagada', 'sholuyi', 'millennium estate', 'atunrase', 'alunrase'],
    'Ogudu': ['ogudu', 'ojota'],
    'Ketu': ['ketu', 'mile 12', 'mile-12', 'alapere', 'ikosi'],
    'Abule Egba/Alagbado': [
        'abule egba', 'abule-egba', 'new oko-oba', 'ekoro', 'agbado', 'ijaye', 
        'ifako ijaiye', 'meiran', 'alankuko', 'alakuko', 'ajasa command', 'salolo'
    ],

    # --- MAINLAND: DENSE / WEST ---
    'Alimosho': [
        'alimosho', 'egbeda', 'gowon estate', 'akowonjo', 'shasha', 'idimu', 
        'igando', 'ikotun', 'ipaja', 'ayobo', 'command', 'ijegun', 'bucknor', 'baruwa'
    ],
    'Oshodi/Isolo': [
        'oshodi', 'isolo', 'mafoluku', 'ajao estate', 'okota', 'ago palace', 
        'ejigbo', 'ilasamaja', 'mushin', 'ilupeju', 'shogunle', 'cele bus', 'cele express', 
        'ago lagos', 'community road ago'
    ],
    'Ajegunle/Amukoko': [
        'ajegunle', 'boundary market', 'wilmer', 'ajeromi', 'kirikiri', 
        'amukoko', 'amukuku', 'alafia street', 'moshalashi street', 'olowolagba', 
        'fatai bello', 'adegboyega', 'makanjuola'
    ],
    'Festac/Satellite': [
        'festac', 'satellite town', 'satalite town', 'amuwo odofin', 'amwuo odofin', 'amuwo-odofin', 
        'mile 2', 'trade fair', 'maza maza', 'mazamaza', 'abule ado', 'abule oshun', 'abule-oshun', 
        'second rainbow', '2nd rainbow', 'kuje amuwo', 'finiger'
    ],
    'Orile': ['orile', 'sari iganmu', 'ganmu', 'alafia bus stop', 'osho street'],
    'Apapa': ['apapa', 'wharf', 'tincan', 'ijora', 'olodi'],
    'Ojo': ['ojo', 'okoko', 'lasu', 'iyana iba', 'iyaniba', 'alaba'],
    'Badagry': ['badagry', 'agbara', 'whispering palms', 'aradagun', 'muwo'],
    'Ikorodu': ['ikorodu', 'agric', 'ogolonto', 'imota', 'imonta', 'igbogbo', 'ebute ikorodu', 'itaoluwo', 'ita oluwo'],
    'Ijesha': ['ijesha', 'ijeshatedo', 'jeshatedo'],
    
    # --- GENERIC FALLBACKS ---
    'Lagos Island (General)': [
        'lagos island', 'idumota', 'balogun', 'cms', 'marina', 'gbajumo', 'okepopo', 
        'obalende', 'campbell street', 'apongbon', 'iga idunganran', 'inabeere', 'okepopo'
    ],
    'Lekki (General)': ['lekki', 'osi community'], 
    'Mainland (General)': ['mainland'],
    
    # --- OUT OF STATE ---
    'Ogun State': ['sagamu', 'sango ota', 'ota,', 'ogijo', 'makun', 'iperu', 'akute'],
    'Delta State': ['delta state', 'warri', 'asaba', 'ughelli', 'ogwashi', 'udu road', 'ovwain'],
    'Abuja': ['abuja', 'fct', 'wuse', 'garki', 'maitama', 'asokoro'],
}

def get_cluster_keyword(address_string):
    """Checks for keywords in the address string"""
    if not address_string or pd.isna(address_string): return None
    clean_addr = str(address_string).lower()
    matches = []
    
    for cluster, keywords in AREA_KEYWORDS.items():
        for keyword in keywords:
            if keyword in clean_addr:
                matches.append((cluster, len(keyword)))
    
    if not matches: return None
    # Sort by length descending (Longest match wins)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]

def process_data(df):
    """Pure Logic Processing - No APIs"""
    
    # 1. Identify Address Column
    target_col = None
    for col in df.columns:
        if col.lower() in POSSIBLE_ADDRESS_COLS:
            target_col = col; break
    if not target_col:
        possible = [c for c in df.columns if 'address' in c.lower()]
        target_col = possible[0] if possible else None

    if not target_col:
        print(f"  ⚠ No address column found. Skipping enrichment.")
        df['area_cluster'] = "Unknown"
        return df

    # 2. Keyword Match (Fast)
    print("  → Running Keyword Match...")
    df['area_cluster'] = df[target_col].apply(get_cluster_keyword)
    
    # 3. State Column Logic (Fast)
    state_col = None
    for col in df.columns:
        if col.lower() in POSSIBLE_STATE_COLS:
            state_col = col; break
            
    if state_col:
        missing_mask = df['area_cluster'].isna()
        
        # Check if state contains 'Lagos'
        is_lagos = df[state_col].astype(str).str.lower().str.contains('lagos', na=False)
        
        # Logic A: If State is NOT Lagos -> Use State Name
        non_lagos_mask = missing_mask & (~is_lagos)
        df.loc[non_lagos_mask, 'area_cluster'] = df.loc[non_lagos_mask, state_col]
        
        # Logic B: If State IS Lagos but no Keyword -> "Lagos - Unclustered"
        lagos_missing_mask = missing_mask & is_lagos
        df.loc[lagos_missing_mask, 'area_cluster'] = "Lagos - Unclustered"
        
        # Logic C: If State is NaN/Null -> "Unclustered"
        null_state_mask = missing_mask & df[state_col].isna()
        df.loc[null_state_mask, 'area_cluster'] = "Unclustered"
        
    else:
        # Fallback if no State column exists
        df['area_cluster'] = df['area_cluster'].fillna("Unclustered")

    df['area_cluster'] = df['area_cluster'].fillna("Unclustered")
    
    return df

def run_location_enrichment(input_path):
    """Wrapper function called by main.py"""
    try:
        print(f"\n--- Location Enrichment (Fast Mode) ---")
        df = pd.read_parquet(input_path)
        
        # Process
        df = process_data(df)
        
        # Save new file
        output_path = input_path.replace('.parquet', '_enriched.parquet')
        df.to_parquet(output_path, index=False)
        
        # Stats for the log
        clustered_count = len(df[~df['area_cluster'].str.contains('Unclustered', na=False)])
        print(f"✓ Clustered: {clustered_count}/{len(df)} rows")
        print(f"✓ Saved: {os.path.basename(output_path)}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Enrichment Failed: {e}")
        return input_path

# ==========================================
# STANDALONE TESTING
# ==========================================
if __name__ == "__main__":
    print("Running in Standalone Mode...")
    
    # Find latest PROCESSED file (usually output of clean.py)
    # Note: Ensure this path matches where clean.py saves files
    list_of_files = glob.glob('data/processed/processed_*.parquet')
    
    # Filter out files that are ALREADY enriched to avoid re-running on result
    list_of_files = [f for f in list_of_files if '_enriched' not in f]
    
    if list_of_files:
        latest = max(list_of_files, key=os.path.getctime)
        print(f"Found latest file: {latest}")
        run_location_enrichment(latest)
    else:
        print("No suitable processed files found in data/processed/ to test.")