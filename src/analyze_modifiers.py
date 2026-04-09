import os
import sys

# Pre-load torch first before heavy C-ext modules like matplotlib/pandas to prevent WinError 1114 DLL conflicts
try:
    import torch
except ImportError:
    pass

import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import logging
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))
from router import compare_routes, find_nearest_edge
from trainer import load_model

def get_net():
    from router import get_net
    return get_net()

def load_catboost():
    try:
        from catboost_trainer import load_catboost
        return load_catboost()
    except Exception:
        return None

def load_gnn_safely(net):
    try:
        # Pre-load torch before gnn_builder so c10.dll initialises cleanly on Windows
        import torch as _torch  # noqa: F401
        from gnn_builder import load_gnn
        print("Loading GNN model (this might take a few seconds)...", flush=True)
        gnn_model, gnn_edge_ids, gnn_graph = load_gnn(net)
        print("GNN loaded successfully.", flush=True)
        return gnn_model, gnn_edge_ids, gnn_graph
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load GNN: {e}", flush=True)
        return None, [], None

def main():
    lgbm = load_model()
    cb = load_catboost()
    
    net = get_net()
    # Load GNN model if available
    gnn_model, gnn_edge_ids, gnn_graph = load_gnn_safely(net)
        
    from ensemble import EnsemblePredictor
    ensemble = EnsemblePredictor(lgbm_model=lgbm, catboost_model=cb, gnn_model=gnn_model, gnn_graph_data=gnn_graph, gnn_edge_ids=gnn_edge_ids)
    
    # From K.R. Hospital
    hosp_lat, hosp_lon = 12.3131, 76.6496
    
    # Destination Udayagiri
    dest_lat, dest_lon = 12.32493, 76.67711
    
    src_edge = find_nearest_edge(hosp_lat, hosp_lon)
    dst_edge = find_nearest_edge(dest_lat, dest_lon)
    
    modifiers = [
        {"weather": "clear", "time_of_day": "offpeak", "day_type": "weekday", "road_hazard": "none", "_label": "Ideal (Clear, Off-Peak)"},
        {"weather": "clear", "time_of_day": "rush", "day_type": "weekday", "road_hazard": "none", "_label": "Rush Hour"},
        {"weather": "rain", "time_of_day": "offpeak", "day_type": "weekday", "road_hazard": "none", "_label": "Rain (Off-Peak)"},
        {"weather": "rain", "time_of_day": "rush", "day_type": "weekday", "road_hazard": "none", "_label": "Rain + Rush Hour"},
        {"weather": "fog", "time_of_day": "rush", "day_type": "weekday", "road_hazard": "none", "_label": "Fog + Rush Hour"},
        {"weather": "clear", "time_of_day": "offpeak", "day_type": "weekday", "road_hazard": "accident", "_label": "Incident: Accident Avoidance"},
        {"weather": "rain", "time_of_day": "rush", "day_type": "weekday", "road_hazard": "accident", "_label": "Worst Case (Rain+Rush+Accident)"},
    ]

    results = []
    scenario_paths = {}
    
    for mod in modifiers:
        cond = dict(mod)
        label = cond.pop("_label")
        ensemble.refresh_gnn_cache(cond)
        
        # Test routing offline modes
        res = compare_routes(src_edge, dst_edge, ensemble, None, cond)
        
        ml_time = res["ml"].get("travel_time", 0)
        dj_time = res["dijkstra"].get("travel_time", 0)
        
        results.append({
            "Scenario": label,
            "Shortest Path Time (s)": round(dj_time, 1),
            "ML Path Time (s)": round(ml_time, 1)
        })
        
        # Save path details for this scenario
        scenario_paths[label] = {
            "dijkstra": {
                "travel_time": round(dj_time, 1),
                "distance": res["dijkstra"].get("distance", 0),
                "hops": res["dijkstra"].get("hops", 0),
                "edges": res["dijkstra"].get("edges", []),
                "coordinates": res["dijkstra"].get("coordinates", [])
            },
            "ml": {
                "travel_time": round(ml_time, 1),
                "distance": res["ml"].get("distance", 0),
                "hops": res["ml"].get("hops", 0),
                "edges": res["ml"].get("edges", []),
                "coordinates": res["ml"].get("coordinates", [])
            }
        }
        
        print(f"Evaluated {label}: ML Path {ml_time:.1f}s / Dijkstra {dj_time:.1f}s")
        
        # Plotting the routes for this scenario
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Extract coords
        ml_y = [pt['lat'] for pt in res["ml"].get("coordinates", [])]
        ml_x = [pt['lon'] for pt in res["ml"].get("coordinates", [])]
        dj_y = [pt['lat'] for pt in res["dijkstra"].get("coordinates", [])]
        dj_x = [pt['lon'] for pt in res["dijkstra"].get("coordinates", [])]
        
        # Plot paths
        ax.plot(dj_x, dj_y, label='Dijkstra Shortest Path', color='red', linewidth=4, alpha=0.6)
        ax.plot(ml_x, ml_y, label='ML Fastest Path', color='blue', linewidth=4, alpha=0.8, linestyle='--')
        
        # Source/Dest
        ax.scatter([hosp_lon], [hosp_lat], color='green', s=150, label='Source', zorder=5)
        ax.scatter([dest_lon], [dest_lat], color='purple', s=150, label='Destination', zorder=5)
        
        ax.set_title(f"Predicted Paths: {label}")
        ax.legend()
        ax.axis('off')
        
        out_dir = Path(__file__).parent.parent / "data" / "visuals"
        out_dir.mkdir(exist_ok=True, parents=True)
        safe_name = label.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("+", "plus").lower()
        out_img = out_dir / f"path_{safe_name}.png"
        plt.savefig(out_img, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    df = pd.DataFrame(results)
    
    out_dir = Path(__file__).parent.parent / "data" / "visuals"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_png = out_dir / "modifier_impact.png"
    out_csv = out_dir / "modifier_table.csv"
    out_json = out_dir / "modifier_paths.json"
    
    with open(out_json, "w") as f:
        json.dump(scenario_paths, f, indent=2)
    print(f"Saved full path predictions to {out_json}")
    
    df.to_csv(out_csv, index=False)
    print(f"Saved summary data to {out_csv}")
        
    # Plotting
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df["Shortest Path Time (s)"], width, label='Dijkstra Average (Shortest Path)', color='#fc8181')
    ax.bar([i + width/2 for i in x], df["ML Path Time (s)"], width, label='ML Ensemble (Fastest Path)', color='#63b3ed')
    
    ax.set_ylabel('Estimated Travel Time (seconds)')
    ax.set_title('Impact of Environmental Modifiers on Route Travel Time (K.R. Hospital to Udayagiri)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Scenario"], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
