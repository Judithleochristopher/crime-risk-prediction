🚨 Crime Risk Prediction \& Safe-Zone Recommendation System



A full-stack machine learning system that analyzes real Chicago crime data, detects high-risk areas using geospatial clustering, identifies nearby crime incidents, and recommends the closest safer zones to the user in real time.



🔍 Project Overview



This project transforms raw crime logs into actionable safety insights by combining geospatial machine learning, density-based clustering, and a full API-driven frontend–backend architecture.



Users can enter any Chicago location and the system will:



Geocode the location (convert place → latitude/longitude)

Identify nearest crime incident within the dataset

Check if the user is inside a high-risk cluster

Recommend the nearest low-risk safe zone

Provide a risk-level explanation + smart action suggestions:



“Reroute”

“Stay alert”

“Avoid area”



All risk assessment uses real geo-coordinates, cluster boundaries, and cluster density analysis.



🧠 Core Machine Learning Techniques

1\. HDBSCAN Clustering (Geospatial Hotspot Detection)



Used to identify crime hotspots based on latitude and longitude

Handles variable density areas

Automatically detects noise (isolated crime incidents)

Uses Haversine distance for accurate Earth-based coordinates



2\. Cluster Summary Generation



For each cluster:

Centroid (mean latitude/longitude)

Maximum radius (distance to furthest point)

Area estimation

Density (points per km²)

Risk label (high / medium / low) via density quantiles



3\. Safe-Zone Recommendation Model



Finds nearest low-risk cluster centroids

Supports fallback strategies if no low-risk clusters exist

Produces distance-based recommended action



🗂 Tech Stack

Backend:

FastAPI

HDBSCAN

NumPy / Pandas

Shapely (polygon containment + geospatial ops)

Uvicorn

Haversine-based geospatial math

Requests (Nominatim API for geocoding)

Frontend:

HTML / CSS / JavaScript

Fetch API for backend communication

Leaflet / Folium (optional map UI if added)



🌍 Dataset



Source: Chicago Open Data (Public Crime Dataset)

Includes crime categories, dates, latitude/longitude

Filtered, cleaned, and clustered into meaningful zones

Large dataset (hundreds of thousands of rows) stored via GitHub Releases, not repository files



🛠 Installation \& Setup

1\. Clone the Repository

git clone https://github.com/Judithleochristopher/crime-risk-prediction.git

cd crime-risk-prediction



2\. Create a Conda Environment

conda create -n crimeenv python=3.10

conda activate crimeenv



3\. Install Dependencies

pip install fastapi uvicorn pandas numpy hdbscan shapely folium requests



4\. Place Required Files



In the project folder, ensure these files exist:



crime\_clusters\_with\_labels.csv

cluster\_summary.csv

Download them from your GitHub Releases if needed.



🚀 Running the Backend (FastAPI)

Run:

uvicorn backend:app --reload



Backend will start at:

http://127.0.0.1:8000



Key Features



Real-time crime hotspot detection

High/medium/low risk classification

Nearest-incident severity awareness (e.g., “ASSAULT, 153 m away”)

Automatic safe-zone recommendations

Smart action suggestions:

1.reroute

2.stay alert

3.avoid area

Full geocoding-to-alert pipeline

Chicago-only coverage with graceful fallback for non-covered inputs



