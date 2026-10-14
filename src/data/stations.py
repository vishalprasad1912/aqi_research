from src.models.model_utils import build_adjacency
from src.utils.paths import INTERIM_DIR, EXTERNAL_DIR, PROCESSED_DIR
from src.data.clean_data import clean_df

import os
import pandas as pd
from pathlib import Path

def main():
    files_path_dict = {}
    selected_stations = []
    coordinates_file_path = EXTERNAL_DIR / "stations_coordinates.csv"

    for file in os.listdir(INTERIM_DIR):
        if file.endswith(".xlsx"):
            filename = os.path.splitext(file)[0]
            files_path_dict[filename] = os.path.join(INTERIM_DIR, file)

    files_path_dict = dict(sorted(files_path_dict.items()))

    for station_name, file_path in files_path_dict.items():
        if Path(str(os.path.join(PROCESSED_DIR, station_name))).exists():
            print(f"\nAlready cleaned {station_name}")
            selected_stations.append(station_name)
            continue

        result = clean_df(station_name, file_path)

        if result is not None:
            selected_stations.append(result)
        else:
            continue

    selected_stations = set(selected_stations)

    print("#" * 80)
    print(f"\nSelected stations ({len(selected_stations)}):\n", selected_stations)

    stations_coor_df = pd.read_csv(coordinates_file_path)
    stations_coor_df = stations_coor_df[stations_coor_df['station_name'].isin(selected_stations)]

    adj = build_adjacency(stations_coor_df.sort_values(by="station_name").reset_index(drop=True))

    print(f"\nAdjacency matrix created for above selected stations at location: data / external / Adjacency_distance_matrix.csv")
    print("#" * 80)

if __name__ == "__main__":
    main()