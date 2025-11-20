# backend.py (final merged)
import os
import json
import math
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from math import radians, sin, cos, sqrt, atan2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from shapely.geometry import shape, Point
from collections import defaultdict

# ---------------- Config ----------------
CSV = "crime_clusters_with_labels.csv"   # crime points with cluster labels + risk
CLUSTER_SUMMARY_CSV = "cluster_summary.csv"  # cluster summary (mean_lat/mean_lon/density_per_km2/risk)
GEOJSON = "cluster_polygons.geojson"     # optional polygons
USER_AGENT = "CrimeHotspotApp/1.0 (71762334019@cit.edu.in)"  # change to your contact email
CHICAGO_VIEWBOX = "-87.9401,41.6445,-87.5240,42.0230"
COVERAGE_MARGIN_KM = 10.0
NEARBY_FALLBACK_METERS = 300
EARTH_R = 6371000.0  # meters

# ---------------- HTTP session ----------------
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ---------------- FastAPI app & CORS ----------------
app = FastAPI(title="CrimeRiskAPI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # change to your frontend origin or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------
def haversine_meters(a_lat, a_lon, b_lat, b_lon):
    lat1, lon1, lat2, lon2 = map(radians, [a_lat, a_lon, b_lat, b_lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return EARTH_R * c

def pairwise_nearest_index(user_lat, user_lon, lats, lons):
    lat1 = np.radians(user_lat)
    lon1 = np.radians(user_lon)
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat2 = np.sin(dlat/2.0)**2
    sin_dlon2 = np.sin(dlon/2.0)**2
    a_term = sin_dlat2 + np.cos(lat1)*np.cos(lat2)*sin_dlon2
    c = 2 * np.arctan2(np.sqrt(a_term), np.sqrt(1 - a_term))
    d = EARTH_R * c
    idx = int(np.argmin(d))
    return idx, float(np.min(d))

# ---------------- Load datasets ----------------
if not os.path.exists(CSV):
    raise FileNotFoundError(f"Dataset not found: {CSV} (put it in same folder)")

df = pd.read_csv(CSV)
# ensure required columns exist
if not {'Latitude','Longitude','cluster'}.issubset(df.columns):
    raise ValueError("CSV must contain Latitude, Longitude, and cluster columns")

lats = df['Latitude'].to_numpy()
lons = df['Longitude'].to_numpy()

cluster_summary = None
if os.path.exists(CLUSTER_SUMMARY_CSV):
    cluster_summary = pd.read_csv(CLUSTER_SUMMARY_CSV)
    # prepare low-risk centroids list
    low_centroids = [
        (int(row['cluster']), (float(row['mean_lat']), float(row['mean_lon'])))
        for _, row in cluster_summary.iterrows() if str(row.get('risk','')).lower() == 'low'
    ]
else:
    low_centroids = []

# ---------------- Compute max_radius_m if missing ----------------
if cluster_summary is not None and not cluster_summary.empty:
    if 'max_radius_m' not in cluster_summary.columns:
        print("cluster_summary missing 'max_radius_m' — computing approximate max radii (this may take a bit)...")
        # build cluster->points mapping
        cluster_points = defaultdict(list)
        for lat, lon, cid in df[['Latitude','Longitude','cluster']].to_numpy():
            cluster_points[int(cid)].append((float(lat), float(lon)))

        max_r_list = []
        for _, row in cluster_summary.iterrows():
            cid = int(row['cluster'])
            centroid_lat = float(row['mean_lat'])
            centroid_lon = float(row['mean_lon'])
            pts = cluster_points.get(cid, [])
            if len(pts) == 0:
                max_r_list.append(0.0)
                continue
            pts_arr = np.array(pts)
            lat2 = np.radians(pts_arr[:, 0])
            lon2 = np.radians(pts_arr[:, 1])
            lat1 = np.radians(centroid_lat)
            lon1 = np.radians(centroid_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            sin_dlat2 = np.sin(dlat / 2.0) ** 2
            sin_dlon2 = np.sin(dlon / 2.0) ** 2
            a_term = sin_dlat2 + np.cos(lat1) * np.cos(lat2) * sin_dlon2
            c = 2 * np.arctan2(np.sqrt(a_term), np.sqrt(1.0 - a_term))
            dists = EARTH_R * c
            max_r_list.append(float(dists.max()))
        cluster_summary['max_radius_m'] = max_r_list
        try:
            cluster_summary.to_csv(CLUSTER_SUMMARY_CSV, index=False)
            print(f"Updated {CLUSTER_SUMMARY_CSV} with max_radius_m.")
        except Exception as e:
            print("Warning: could not write cluster_summary.csv:", e)

# ---------------- Load polygons if present ----------------
polygons = {}
if os.path.exists(GEOJSON):
    with open(GEOJSON, 'r') as f:
        gj = json.load(f)
    for feat in gj.get('features', []):
        cid = int(feat['properties']['cluster'])
        polygons[cid] = shape(feat['geometry'])
    print("Loaded polygons for", len(polygons), "clusters")
else:
    print("No cluster polygons found (continuing without polygon containment checks).")

# ---------------- Precompute dataset bbox, centroid, radius ----------------
_dataset_min_lat = float(df['Latitude'].min())
_dataset_max_lat = float(df['Latitude'].max())
_dataset_min_lon = float(df['Longitude'].min())
_dataset_max_lon = float(df['Longitude'].max())
centroid_lat = float(df['Latitude'].mean())
centroid_lon = float(df['Longitude'].mean())

_distances = df.apply(lambda r: haversine_meters(centroid_lat, centroid_lon, r['Latitude'], r['Longitude']), axis=1)
_dataset_radius_m = float(_distances.max())
_dataset_radius_km = _dataset_radius_m / 1000.0
MAX_ACCEPT_DISTANCE_KM = _dataset_radius_km + COVERAGE_MARGIN_KM

# ---------------- Geocode endpoint (Nominatim) ----------------
@app.get("/geocode")
def geocode_place(place: str = Query(..., min_length=1)):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place,
        "format": "json",
        "limit": 1,
        "countrycodes": "us",
        "viewbox": CHICAGO_VIEWBOX,
        "bounded": 1
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = session.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        print("Geocode HTTP error:", code, getattr(e.response, "text", ""))
        if code == 403:
            raise HTTPException(status_code=502, detail="Geocoding provider rejected request (403). Check User-Agent / rate limits.")
        raise HTTPException(status_code=502, detail="Geocoding upstream request failed.")
    except requests.exceptions.RequestException as e:
        print("Geocode request failed:", e)
        raise HTTPException(status_code=502, detail="Geocoding upstream request failed.")

    if not data:
        # fallback (relaxed)
        try:
            resp2 = session.get(url, params={"q": place, "format":"json", "limit":1, "countrycodes":"us"}, headers=headers, timeout=10)
            resp2.raise_for_status()
            data2 = resp2.json()
        except requests.exceptions.RequestException as e:
            print("Geocode fallback failed:", e)
            raise HTTPException(status_code=502, detail="Geocoding fallback failed.")
        if not data2:
            raise HTTPException(status_code=404, detail="Place not found")
        first = data2[0]
    else:
        first = data[0]

    lat = float(first["lat"]); lon = float(first["lon"])
    return {"lat": lat, "lon": lon, "display_name": first.get("display_name","")}

# ---------------- API models ----------------
class LocRequest(BaseModel):
    lat: float
    lon: float

# ---------------- Find nearest low-risk (robust) ----------------
def find_nearest_low_risk(user_latlon, top_k=3, fallback_use_nearest_centroid=True):
    """
    Returns a list of up to top_k candidate centroids (dicts) sorted by distance.
    Fallbacks: explicit low_centroids -> lowest-density clusters -> nearest centroids.
    """
    candidates = []
    if 'low_centroids' in globals() and low_centroids:
        candidates = [(cid, c) for cid, c in low_centroids]

    if not candidates and (cluster_summary is not None) and (not cluster_summary.empty):
        cs_sorted = cluster_summary.sort_values('density_per_km2', ascending=True)
        fallback = cs_sorted.head(max(20, top_k))
        candidates = [(int(r['cluster']), (float(r['mean_lat']), float(r['mean_lon']))) for _, r in fallback.iterrows()]

    if not candidates and fallback_use_nearest_centroid and (cluster_summary is not None) and (not cluster_summary.empty):
        drows = []
        for _, r in cluster_summary.iterrows():
            cid = int(r['cluster'])
            latc = float(r['mean_lat']); lonc = float(r['mean_lon'])
            d = haversine_meters(user_latlon[0], user_latlon[1], latc, lonc)
            drows.append((cid, d, latc, lonc))
        drows.sort(key=lambda x: x[1])
        candidates = [(cid, (latc, lonc)) for cid, d, latc, lonc in drows[:max(top_k, 20)]]

    if not candidates:
        return None

    scored = []
    for cid, (latc, lonc) in candidates:
        d = haversine_meters(user_latlon[0], user_latlon[1], latc, lonc)
        scored.append({'cluster_id': int(cid), 'distance_m': float(d), 'centroid': (float(latc), float(lonc))})
    scored.sort(key=lambda x: x['distance_m'])
    return scored[:top_k]

# ---------------- Helper: centroid containment (optional) ----------------
def get_nearest_centroid_if_within(user_lat, user_lon, buffer_m=50):
    if cluster_summary is None or cluster_summary.empty:
        return None
    best = None
    for _, crow in cluster_summary.iterrows():
        cid = int(crow['cluster'])
        latc = float(crow['mean_lat']); lonc = float(crow['mean_lon'])
        max_r = float(crow.get('max_radius_m', 0.0))
        d = haversine_meters(user_lat, user_lon, latc, lonc)
        if d <= (max_r + buffer_m):
            if best is None or d < best[1]:
                best = (cid, d, max_r, latc, lonc)
    if best is not None:
        cid_near, d_near, max_r, latc_near, lonc_near = best
        return {'cluster_id': int(cid_near), 'distance_m': float(d_near), 'centroid': (latc_near, lonc_near)}
    return None

# ---------------- Main locate endpoint ----------------
@app.post("/locate")
def locate(req: LocRequest):
    user_lat, user_lon = req.lat, req.lon

    # coverage checks
    in_bbox = (_dataset_min_lat <= user_lat <= _dataset_max_lat) and (_dataset_min_lon <= user_lon <= _dataset_max_lon)
    user_to_centroid_km = haversine_meters(user_lat, user_lon, centroid_lat, centroid_lon) / 1000.0
    in_radius = (user_to_centroid_km <= MAX_ACCEPT_DISTANCE_KM)
    in_coverage = in_bbox or in_radius

    if not in_coverage:
        idx, dist_m = pairwise_nearest_index(user_lat, user_lon, lats, lons)
        nearest_row = df.iloc[idx].to_dict()
        fallback = {
            "nearest_point_lat": float(nearest_row['Latitude']),
            "nearest_point_lon": float(nearest_row['Longitude']),
            "nearest_point_distance_km": round(dist_m/1000.0, 3),
            "nearest_point_risk": str(nearest_row.get('risk','unknown')),
            "warning": "User location is outside dataset coverage (Chicago). Results are for demo only and not applicable to local safety."
        }
        return {
            "in_coverage": False,
            "coverage_info": {
                "dataset_bbox": {"min_lat": _dataset_min_lat, "max_lat": _dataset_max_lat,
                                 "min_lon": _dataset_min_lon, "max_lon": _dataset_max_lon},
                "dataset_centroid": {"lat": centroid_lat, "lon": centroid_lon},
                "dataset_radius_km": round(_dataset_radius_km,2),
                "max_accept_distance_km": round(MAX_ACCEPT_DISTANCE_KM,2),
                "user_distance_to_centroid_km": round(user_to_centroid_km,2)
            },
            "demo_nearest_cluster": fallback,
            "message": "Your location is outside the geographic coverage of the dataset (Chicago). No local safety recommendation can be provided."
        }

    # 1) nearest crime point
    idx, dist_m = pairwise_nearest_index(user_lat, user_lon, lats, lons)
    nearest_row = df.iloc[idx].to_dict()
    nearest_info = {
        "nearest_point_index": int(idx),
        "nearest_point_lat": float(nearest_row['Latitude']),
        "nearest_point_lon": float(nearest_row['Longitude']),
        "nearest_point_distance_m": float(dist_m),
        "nearest_point_cluster": int(nearest_row.get('cluster', -1)),
        "nearest_point_risk": str(nearest_row.get('risk', 'low')),
        "primary_type": nearest_row.get('Primary Type', None)
    }

    # 2) polygon containment check
    inside_cluster = None
    if polygons:
        pt = Point(user_lon, user_lat)  # shapely uses (lon,lat)
        for cid, poly in polygons.items():
            if poly.contains(pt):
                inside_cluster = int(cid)
                break

    # 3) centroid containment check (fallback)
    containing_cluster = inside_cluster
    if containing_cluster is None:
        centroid_check = get_nearest_centroid_if_within(user_lat, user_lon, buffer_m=50)
        if centroid_check is not None:
            containing_cluster = int(centroid_check['cluster_id'])

    # 4) nearest-point cluster fallback
    if containing_cluster is None and nearest_info['nearest_point_cluster'] != -1:
        if nearest_info['nearest_point_distance_m'] <= NEARBY_FALLBACK_METERS:
            containing_cluster = int(nearest_info['nearest_point_cluster'])

    # determine if user in high-risk cluster
    user_in_high = False
    if containing_cluster is not None:
        sample = df[df['cluster'] == containing_cluster]
        if not sample.empty:
            user_in_high = (sample.iloc[0].get('risk','low') == 'high')

    # ---------------- Patch A + B: nearest_low_list, message, recommended_action ----------------
    nearest_low_list = find_nearest_low_risk((user_lat, user_lon), top_k=3)
    nearest_low = None
    recommended_action = None
    message = ""

    if user_in_high:
        if nearest_low_list:
            nearest_low = nearest_low_list[0]
            cent_lat, cent_lon = nearest_low['centroid']
            nearest_low['centroid'] = {"lat": float(cent_lat), "lon": float(cent_lon)}
            km = nearest_low['distance_m'] / 1000.0

            if km <= 0.5:
                recommended_action = "reroute"
                action_text = "Consider rerouting."
            elif km <= 2.0:
                recommended_action = "stay_alert"
                action_text = "Remain alert; reroute if possible."
            else:
                recommended_action = "avoid_area"
                action_text = "No nearby safe zone detected; avoid this area if possible."

            inc_type = nearest_info.get("primary_type")
            inc_dist_m = nearest_info.get("nearest_point_distance_m")
            incident_text = ""
            if inc_type:
                incident_text = f" Nearby incident: {inc_type} ({inc_dist_m:.0f} m)."

            cent = nearest_low['centroid']
            cent_text = f"({cent['lat']:.6f}, {cent['lon']:.6f})"

            message = (
                f"You are in a high-risk zone.{incident_text} "
                f"Nearest low-risk area: {km:.2f} km at centroid {cent_text}. "
                f"{action_text}"
            )

        else:
            recommended_action = "avoid_area"
            message = (
                "You are in a high-risk zone. No low-risk centroid could be identified nearby. "
                "Exercise caution and move toward populated or well-lit public areas."
            )
    else:
        recommended_action = "safe"
        message = "You are not in a high-risk zone."

    # ---------------- Response ----------------
    return {
        "in_coverage": True,
        "user_location": {"lat": user_lat, "lon": user_lon},
        "nearest_point": nearest_info,
        "containing_cluster": containing_cluster,
        "in_high_risk": user_in_high,
        "nearest_low": nearest_low,
        "nearest_low_list": nearest_low_list,
        "recommended_action": recommended_action,
        "message": message
    }

# End of backend.py
