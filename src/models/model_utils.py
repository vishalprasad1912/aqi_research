import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from src.utils.paths import VISUALIZATION_DIR, EXTERNAL_DIR
import matplotlib.pyplot as plt
import seaborn as sns

# Haversine Distance Function
def haversine(phi_1, lembda_1, phi_2, lembda_2):
    R = 6371.0  # Earth's radius in km (average)
    # equatorial radius-> 6,378 kilometers, polar radius-> 6,357 kilometers

    phi_1 = np.radians(phi_1)
    lembda_1 = np.radians(lembda_1)
    phi_2 = np.radians(phi_2)
    lembda_2 = np.radians(lembda_2)

    dis_lat = phi_1 - phi_2
    dis_lon = lembda_1 - lembda_2

    a = np.sin(dis_lat / 2) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(dis_lon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))

    return d

# Euclidean distance function
def euclidean(phi_1, lemda_1, phi_2, lemda_2):
    dis_lat = phi_1 - phi_2
    dis_lon = lemda_1 - lemda_2
    dist = np.sqrt(dis_lat**2 + dis_lon**2)

    return dist

def dist_plot(df, type, threshold):
    DISTANCE_MATRIX_DIR = VISUALIZATION_DIR / "distance_matrix"
    DISTANCE_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(os.path.join(EXTERNAL_DIR, f"{type}_distance_matrix.csv"), columns=df.columns, index=True)

    plt.figure(figsize=(18, 12))
    sns.heatmap(data=df, cmap='coolwarm', cbar_kws={'label': f'Distance in Kilometers (Km)' if type != "adj" else None}, annot=True)
    if type == "Adjacency":
        plt.title(f"Heatmap of the {type} between the AQI stations", fontsize=20, pad=4, fontweight="bold")
    else:
        plt.title(f"Heatmap of the {type} Distances between the AQI stations (threshold: {threshold} km)", fontsize=16, pad=4, fontweight="bold")
    plt.xticks(fontsize=12, rotation=90, fontweight="bold")
    plt.yticks(fontsize=12, rotation=0, fontweight="bold")
    plt.ylabel("AQI station", fontsize=16, fontweight="bold")
    plt.xlabel("AQI station", fontsize=16, fontweight="bold")
    plt.tight_layout()

    plt.savefig(os.path.join(DISTANCE_MATRIX_DIR, f"{type}_distance_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

def build_adjacency(stations_coor_df):
    # Average-distance-threshold binary adjacency - Two stations are connected (1) if their Euclidean distance is below
    # the mean pairwise distance, else (0).
    # Self-loops are always included.

    stations = stations_coor_df["station_name"].values
    latitudes = stations_coor_df["latitude"].values
    longitudes = stations_coor_df["longitude"].values

    n = len(stations_coor_df)
    haversine_distance_matrix = np.zeros((n, n))
    euclidean_distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            haversine_distance_matrix[i, j] = haversine(
                latitudes[i], longitudes[i],
                latitudes[j], longitudes[j]
            )
            euclidean_distance_matrix[i, j] = euclidean(
                latitudes[i], longitudes[i],
                latitudes[j], longitudes[j]
            ) * 100

    haversine_distance_df = pd.DataFrame(
        haversine_distance_matrix,
        index=stations,
        columns=stations
    ).round(2)
    euclidean_distance_df = pd.DataFrame(
        euclidean_distance_matrix,
        index=stations,
        columns=stations
    ).round(2)

    off_diag = haversine_distance_matrix[~np.eye(n, dtype=bool)]
    threshold = off_diag.mean().round(2) if n > 1 else 0.0

    dist_plot(haversine_distance_df, "Haversine", threshold)
    dist_plot(euclidean_distance_df, "Euclidean", threshold)

    adj = (haversine_distance_matrix < threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)

    np.save(EXTERNAL_DIR / "adjacency_matrix.npy", adj)
    adjacency_df = pd.DataFrame(adj, index=stations, columns=stations).round(2)
    dist_plot(adjacency_df, "Adjacency", threshold)

    return adjacency_df


def normalize_adjacency(adj):
    # Symmetric normalisation D^-1/2 (A+I) D^-1/2
    deg = adj.sum(-1) # horizontal sum along rows
    d_inverse_sqrt = deg.pow(-0.5)
    # d_inverse_sqrt[torch.isinf(d_inverse_sqrt)] = 0.0
    return d_inverse_sqrt[:, None] * adj * d_inverse_sqrt[None, :]


class PredictionHead(nn.Module):
    # Fully connected task layer: hidden -> 256 -> len(target_idx)

    def __init__(self, hidden, horizon, dropout=0.5, mid=32): # changed mid from 256 to 32
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(mid, horizon)
            nn.Linear(mid, 2 * self.horizon)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.horizon, 2)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)