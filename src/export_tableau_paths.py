import os
import sys
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "src"))

from router import compare_routes, find_nearest_edge
from ensemble import EnsemblePredictor
from stations import STATIONS_RAW

def export_tableau_paths():
    # 1. Load Ensemble Model
    try:
        predictor = EnsemblePredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Pick Source and Destination
    # We will go from K.R. Hospital (H001) to Vijayanagar Police Station (P002)
    # These offer a good cross-city route in Mysore mapped in the data.
    kr_hospital = next(s for s in STATIONS_RAW if s["id"] == "H001")
    vijayanagar = next(s for s in STATIONS_RAW if s["id"] == "P002")

    src_edge = find_nearest_edge(kr_hospital["lat"], kr_hospital["lon"])
    dst_edge = find_nearest_edge(vijayanagar["lat"], vijayanagar["lon"])

    if not src_edge or not dst_edge:
        print("Failed to resolve edges for source or destination.")
        return

    print(f"Routing from {src_edge} to {dst_edge} under heavy traffic conditions...")

    # 3. Define some conditions to ensure the ML route differs from Dijkstra
    conditions = {
        "weather": "rain",
        "time_of_day": "rush",
        "road_hazard": "accident"
    }

    # 4. Generate Routes
    result = compare_routes(src_edge, dst_edge, predictor, conditions=conditions)
    
    dijkstra_coords = result["dijkstra"]["coordinates"]
    ml_coords = result["ml"]["coordinates"]

    if not dijkstra_coords or not ml_coords:
        print("Failed to generate one or both route paths.")
        return

    # 5. Format for Tableau Map (Route_Type, Point_Order, Latitude, Longitude)
    records = []
    
    for i, pt in enumerate(dijkstra_coords):
        records.append({
            "Route_Type": "Standard Shortest Path (Dijkstra)",
            "Point_Order": i + 1,
            "Latitude": pt["lat"],
            "Longitude": pt["lon"],
            "Estimated_Time_sec": result["dijkstra"]["travel_time"]
        })

    for i, pt in enumerate(ml_coords):
        records.append({
            "Route_Type": "AI Quickest Path (ML Ensemble)",
            "Point_Order": i + 1,
            "Latitude": pt["lat"],
            "Longitude": pt["lon"],
            "Estimated_Time_sec": result["ml"]["travel_time"]
        })

    # Save to CSV
    df = pd.DataFrame(records)
    out_path = BASE_DIR / "data" / "visuals" / "tableau_route_paths.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_path, index=False)

    print(f"\n✅ Tableau path map data successfully exported to: {out_path}")
    print(f"   Dijkstra Nodes: {len(dijkstra_coords)}    Estimated Time: {result['dijkstra']['travel_time']}s")
    print(f"   ML Nodes:       {len(ml_coords)}    Estimated Time: {result['ml']['travel_time']}s")

if __name__ == "__main__":
    export_tableau_paths()
