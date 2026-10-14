import os
import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DIR
OUTPUT_FILE = "stations_data.npz"

stations_file = [f for f in os.listdir(PROCESSED_DIR)]

station_names = []
station_data = {}

print(f"Found {len(stations_file)} AQI monitoring stations")

for idx, file in enumerate(stations_file):
    station_name = file
    df = pd.read_csv(os.path.join(PROCESSED_DIR, file))
    station_data[f"station_{station_name}"] = df.values
    station_names.append(station_name)
    print(f"{idx}: {station_name} -> {df.shape}")

np.savez_compressed(
    os.path.join(PROCESSED_DIR, OUTPUT_FILE),
    station_names=np.array(station_names, dtype=object),
    **station_data
)

print(f"\nSaved: {OUTPUT_FILE}")