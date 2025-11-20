# crime_pipeline_hdbscan_local.py
import os
import math
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ---------- User-configurable paths (relative) ----------
INPUT_CSV = "crime_filtered.csv"
OUT_PARQUET = "crime_clusters.parquet"
OUT_CSV = "crime_clusters_with_labels.csv"
CLUSTER_SUMMARY_CSV = "cluster_summary.csv"
MAP_OUT = "crime_clusters_map.html"

# ---------- Parameters you may tune ----------
MIN_CLUSTER_SIZE = 50   # try 50,100,200 depending on density
SAMPLE_FOR_MAP = 5000     # number of points to draw on map (keeps map responsive)

# ---------- Optional imports with clear error messages ----------
try:
    import hdbscan
except Exception as e:
    raise ImportError("hdbscan is required. Install: pip install hdbscan") from e

try:
    import folium
except Exception as e:
    raise ImportError("folium is required. Install: pip install folium") from e

# Parquet optional
_try_parquet = True
try:
    import pyarrow  # only to ensure parquet will work
except Exception:
    _try_parquet = False
    # We'll still write CSV if pyarrow isn't available

# ---------- Helpers ----------
EARTH_RADIUS_M = 6371000.0  # meters

def haversine_distance(coord1, coord2):
    """Haversine distance in meters between two (lat, lon) in degrees."""
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return EARTH_RADIUS_M * c

def pairwise_haversine_meters(a, b):
    """
    Vectorized haversine distances (meters) between arrays:
      a: (N,2), b: (M,2) -> returns (N,M)
    Both a and b are in degrees.
    """
    a_rad = np.radians(a)
    b_rad = np.radians(b)
    lat1 = a_rad[:, 0][:, None]
    lon1 = a_rad[:, 1][:, None]
    lat2 = b_rad[:, 0][None, :]
    lon2 = b_rad[:, 1][None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat2 = np.sin(dlat / 2.0) ** 2
    sin_dlon2 = np.sin(dlon / 2.0) ** 2
    a_term = sin_dlat2 + np.cos(lat1) * np.cos(lat2) * sin_dlon2
    c = 2 * np.arctan2(np.sqrt(a_term), np.sqrt(1 - a_term))
    return EARTH_RADIUS_M * c

# ---------- Load & clean ----------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV} (place it in this folder or change INPUT_CSV)")

print("Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV, parse_dates=['Date'], dayfirst=True, infer_datetime_format=True)
print("Initial rows:", len(df))

# Drop bad coords
df = df.dropna(subset=['Latitude', 'Longitude']).copy()
df = df[df['Latitude'].between(-90, 90) & df['Longitude'].between(-180, 180)]
print("Rows after coordinate cleaning:", len(df))

coords_deg = df[['Latitude','Longitude']].to_numpy()

# ---------- HDBSCAN clustering (haversine metric expects radians) ----------
coords_rad = np.radians(coords_deg)
print(f"Running HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}) ...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='haversine', core_dist_n_jobs=4)
labels = clusterer.fit_predict(coords_rad)
df['cluster'] = labels  # -1 = noise
print("Unique clusters (including -1 noise):", np.unique(labels).shape[0])

# ---------- Summarize clusters ----------
cluster_summary = df[df['cluster'] != -1].groupby('cluster').agg(
    n_points = ('cluster','size'),
    mean_lat = ('Latitude','mean'),
    mean_lon = ('Longitude','mean')
).reset_index()

# approximate cluster radius (meters) and circular area (km^2)
max_radii = []
areas_km2 = []
for _, row in cluster_summary.iterrows():
    cid = int(row['cluster'])
    centroid = (row['mean_lat'], row['mean_lon'])
    pts = df[df['cluster'] == cid][['Latitude','Longitude']].to_numpy()
    if len(pts) == 0:
        max_r = 0.0
    else:
        dists = pairwise_haversine_meters(np.array([centroid]), pts).flatten()
        max_r = float(dists.max())
    max_radii.append(max_r)
    areas_km2.append(math.pi * (max_r/1000.0)**2)  # km^2

cluster_summary['max_radius_m'] = max_radii
cluster_summary['area_km2'] = areas_km2
cluster_summary['density_per_km2'] = cluster_summary['n_points'] / cluster_summary['area_km2'].replace(0, np.nan)
cluster_summary['density_per_km2'] = cluster_summary['density_per_km2'].fillna(0).replace(np.inf, 0)

# ---------- Risk labeling by density quantiles ----------
q75 = cluster_summary['density_per_km2'].quantile(0.75)
q25 = cluster_summary['density_per_km2'].quantile(0.25)

def risk_from_density(d):
    if d >= q75:
        return 'high'
    elif d >= q25:
        return 'medium'
    else:
        return 'low'

cluster_summary['risk'] = cluster_summary['density_per_km2'].apply(risk_from_density)

# Map risk back to df (noise -> low)
risk_map = dict(zip(cluster_summary['cluster'], cluster_summary['risk']))
df['risk'] = df['cluster'].map(risk_map).fillna('low')

# ---------- Save outputs ----------
print("Writing outputs...")
if _try_parquet:
    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print("Parquet saved:", OUT_PARQUET)
    except Exception as e:
        print("Parquet write failed, will still write CSV. Error:", e)

df.to_csv(OUT_CSV, index=False)
cluster_summary.to_csv(CLUSTER_SUMMARY_CSV, index=False)
print("CSV saved:", OUT_CSV)
print("Cluster summary saved:", CLUSTER_SUMMARY_CSV)

# ---------- Nearest low-risk finder ----------
low_centroids = [
    (int(row['cluster']), (row['mean_lat'], row['mean_lon']), float(row['density_per_km2']))
    for _, row in cluster_summary.iterrows() if row['risk'] == 'low'
]

def find_nearest_low_risk(user_latlon):
    """
    Return dict {cluster_id, distance_m, centroid, density} or None if no low-risk clusters exist.
    user_latlon: (lat, lon) degrees
    """
    if not low_centroids:
        return None
    dists = [(cid, haversine_distance(user_latlon, latlon), latlon, dens) for cid, latlon, dens in low_centroids]
    nearest = min(dists, key=lambda x: x[1])
    return {'cluster_id': nearest[0], 'distance_m': nearest[1], 'centroid': nearest[2], 'density': nearest[3]}

# ---------- Folium visualization ----------
print("Building folium map (sampled)...")
m = folium.Map(location=[coords_deg[:,0].mean(), coords_deg[:,1].mean()], zoom_start=12)
color_map = {'high':'red', 'medium':'orange', 'low':'green'}

sample = df.sample(n=min(len(df), SAMPLE_FOR_MAP), random_state=1)
for _, row in sample.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=2,
        color=color_map.get(row['risk'], 'gray'),
        fill=True,
        fill_opacity=0.6,
    ).add_to(m)

# centroids
for _, row in cluster_summary.iterrows():
    folium.Marker(
        [row['mean_lat'], row['mean_lon']],
        popup=f"cluster {int(row['cluster'])}<br>n_points={int(row['n_points'])}<br>density={row['density_per_km2']:.1f}/km2<br>risk={row['risk']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# save map
m.save(MAP_OUT)
print("Map saved:", MAP_OUT)

# ---------- Summary prints ----------
print("\nTop clusters by points:\n", cluster_summary.sort_values('n_points', ascending=False).head())
print("\nRisk counts:\n", df['risk'].value_counts())

# Example usage message for the user
center_example = (coords_deg[:,0].mean(), coords_deg[:,1].mean())
print("\nExample: nearest low-risk to center-of-data:", find_nearest_low_risk(center_example))
print("\nTo programmatically query nearest low-risk, import this script or copy `find_nearest_low_risk` into your app.")
