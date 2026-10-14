from src.data.load_data import load_excel
from src.utils.paths import PROCESSED_DIR, VISUALIZATION_DIR

import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_df(station_name, path):
    df = load_excel(path)

    BEFORE_CLEANING_NULL_VALUES = VISUALIZATION_DIR / "before_cleaning"
    BEFORE_CLEANING_NULL_VALUES.mkdir(parents=True, exist_ok=True)
    AFTER_CLEANING_NULL_VALUES = VISUALIZATION_DIR / "after_cleaning"
    AFTER_CLEANING_NULL_VALUES.mkdir(parents=True, exist_ok=True)

    # Visualization of null values before data cleaning
    null_df = df.isna()
    plt.figure(figsize=(10, 6))
    sns.heatmap(data=null_df, cmap='viridis', cbar_kws={'label': 'Missing Data'})
    plt.title(f"Before: Missing Values Heatmap of {station_name}", fontsize=16, pad=2, fontweight='bold')
    plt.ylabel("Timestamp", fontsize=12, fontweight='bold')
    plt.xlabel("Features", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BEFORE_CLEANING_NULL_VALUES, f"{station_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # step 1. Dropped stations having almost no values in their meteorological features [AT, RH, WS, WD, RF, SR, BP, NH3, SO2]
    rm_stations_names1 = {'aya_nagar', 'burari_crossing', 'crri_mathura_road', 'igi_airport_t3', 'lodhi_road', 'north_campus_du'}

    if station_name in rm_stations_names1:
        print(f"\n{station_name} skipped for this study.")
        return None

    # step 2. Dropped cols containing very much missing values
    df = df.drop(columns=['Benzene (µg/m³)', 'Toluene (µg/m³)', 'Xylene (µg/m³)', 'O Xylene (µg/m³)',
                          'Eth-Benzene (µg/m³)', 'MP-Xylene (µg/m³)', 'TOT-RF (mm)', 'VWS (m/s)'])

    # step 3. Removing some stations having values missing in one or more required meteorological features
    rm_stations_names2 = {'alipur', 'anand_vihar', 'chandni_chowk', 'dilshad_garden', 'dtu',
                         'mandir_marg', 'nsit_dwarka', 'punjabi_bagh', 'shadipur', 'sirifort', 'r_k_puram'}

    if station_name in rm_stations_names2:
        print(f"\n{station_name} left for this study.")
        return None

    # method used: linear interpolation, bfill and ffill
    df = df.interpolate(method='linear').bfill().ffill()  # to fill boundary NaNs

    # Visualization of null values after data cleaning
    null_df = df.isna()
    plt.figure(figsize=(10, 6))
    sns.heatmap(data=null_df, cmap='viridis', cbar_kws={'label': 'Missing Data'})
    plt.title(f"After: Missing Values Heatmap of {station_name}", fontsize=16, pad=2, fontweight='bold')
    plt.ylabel("Timestamp", fontsize=12, fontweight='bold')
    plt.xlabel("Features", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(AFTER_CLEANING_NULL_VALUES, f"{station_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    out_path = os.path.join(PROCESSED_DIR, station_name)
    df.to_csv(out_path, columns=df.columns, index=True)

    print(f"\n{station_name} data cleaned successfully.")

    return station_name

if __name__ == "__main__":
    clean_df("station_name", "path.xlsx")