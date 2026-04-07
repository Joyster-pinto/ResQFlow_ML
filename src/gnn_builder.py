"""
ResQFlow ML — GNN Graph Builder & Trainer
Converts the SUMO road network into a PyTorch Geometric graph and trains/loads
the GraphSAGE travel-time prediction model.

Graph structure (LINE GRAPH of SUMO network):
  - GNN node  = SUMO edge (road segment)
  - GNN edge  = consecutive SUMO edges (A's end-node == B's start-node)

Node features:  12 traffic + road features (same as LightGBM)
Node label:     predicted travel time (seconds)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
MODEL_DIR   = BASE_DIR / "models"
GNN_PATH    = MODEL_DIR / "gnn_router.pt"
GRAPH_PATH  = MODEL_DIR / "graph_data.pt"
GNN_META    = MODEL_DIR / "gnn_meta.json"
MODEL_DIR.mkdir(exist_ok=True)

# SUMO network
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
NET_FILE = str(BASE_DIR / "sumo" / "scenarios" / "mysore.net.xml")

from gnn_model import gnn_available, RoadGNN

FEATURE_COLS = [
    "vehicle_count", "mean_speed", "occupancy", "waiting_time",
    "edge_length",   "max_speed",  "lane_count",
    "hour_of_day",   "is_peak_hour",
    "speed_ratio",   "density",    "weather_factor",
]


# ── Road type from speed limit ────────────────────────────────────────────────

def _road_type_code(max_speed_ms: float) -> int:
    """
    Bucket max speed into road type:
    0 = residential (<= 30 km/h)
    1 = local       (<= 50 km/h)
    2 = arterial    (<= 70 km/h)
    3 = highway     (> 70 km/h)
    """
    kmh = max_speed_ms * 3.6
    if kmh <= 30:  return 0
    if kmh <= 50:  return 1
    if kmh <= 70:  return 2
    return 3


# ── Graph construction ────────────────────────────────────────────────────────

def build_road_graph(net, conditions: dict = None):
    """
    Build a PyTorch Geometric Data object from the SUMO network.

    Returns
    -------
    data : torch_geometric.data.Data
    edge_ids : list[str]   — ordered list of SUMO edge IDs (index maps to GNN node)
    """
    if not gnn_available():
        raise RuntimeError("PyTorch / torch_geometric required to build GNN graph.")

    import torch
    from torch_geometric.data import Data

    conditions = conditions or {}
    cond = conditions

    # Extract passenger-accessible edges only
    logger.info("Building road graph from SUMO network...")
    edges_list = [
        e for e in net.getEdges()
        if e.allows("passenger") and e.getLength() > 20
    ]
    logger.info(f"Graph nodes (SUMO edges): {len(edges_list)}")

    # Index: edge_id → node index
    eid_to_idx = {e.getID(): i for i, e in enumerate(edges_list)}
    N = len(edges_list)

    # ── Node feature matrix ──────────────────────────────────────────────────
    # Use realistic feature defaults per edge (no live SUMO data at build time)
    hour = pd.Timestamp.now().hour
    is_peak = 1 if hour in list(range(7, 10)) + list(range(17, 20)) else 0
    weather_factor = {"clear": 0.0, "rain": 0.5, "fog": 1.0}.get(
        cond.get("weather", "clear"), 0.0)

    X = np.zeros((N, 12), dtype=np.float32)
    for i, e in enumerate(edges_list):
        ms  = e.getSpeed()
        el  = e.getLength()
        lc  = e.getLaneNumber()
        vc  = 3.0          # default: moderate traffic
        spd = ms * 0.7     # assume 70% of speed limit
        occ = 0.15
        wt  = 5.0
        sr  = spd / max(ms, 0.1)
        dn  = vc / max(el, 1.0)
        X[i] = [vc, spd, occ, wt, el, ms, lc, hour, is_peak, sr, dn, weather_factor]

    # ── Edge index (GNN edges = consecutive SUMO edges) ──────────────────────
    src_nodes, dst_nodes = [], []
    for e in edges_list:
        to_node = e.getToNode()
        for out_edge in to_node.getOutgoing():
            oid = out_edge.getID()
            if oid in eid_to_idx and oid != e.getID():
                src_nodes.append(eid_to_idx[e.getID()])
                dst_nodes.append(eid_to_idx[oid])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    x_tensor   = torch.tensor(X, dtype=torch.float)

    # Synthetic travel time labels (for training)
    y = torch.tensor(
        [e.getLength() / max(e.getSpeed() * 0.7, 0.1) for e in edges_list],
        dtype=torch.float,
    )

    data = Data(x=x_tensor, edge_index=edge_index, y=y)
    data.edge_ids = [e.getID() for e in edges_list]   # attach for lookup

    logger.info(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    return data, [e.getID() for e in edges_list]


# ── GNN Training ──────────────────────────────────────────────────────────────

def train_gnn(graph_data, synthetic_df: pd.DataFrame = None,
              epochs: int = 150, lr: float = 1e-3) -> "RoadGNN":
    """
    Train the GraphSAGE model on the road graph.
    Uses synthetic travel-time estimates as supervision signal.
    """
    if not gnn_available():
        raise RuntimeError("PyTorch / torch_geometric required.")

    import torch
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training GNN on {device}  ({graph_data.num_nodes} nodes, {epochs} epochs)")

    model = RoadGNN(in_features=12, hidden=128).to(device)
    data  = graph_data.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    best_state = None

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = torch.nn.functional.mse_loss(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == epochs:
            rmse = (loss.item() ** 0.5)
            logger.info(f"  GNN epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}  "
                        f"RMSE={rmse:.3f}s")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    logger.info(f"GNN training complete. Best RMSE: {best_loss**0.5:.3f}s")
    return model


# ── Persistence ───────────────────────────────────────────────────────────────

def save_gnn(model, graph_data, edge_ids):
    if not gnn_available():
        return
    import torch
    torch.save(model.state_dict(), GNN_PATH)
    torch.save({"x": graph_data.x, "edge_index": graph_data.edge_index,
                "y": graph_data.y, "edge_ids": edge_ids}, GRAPH_PATH)
    with open(GNN_META, "w") as f:
        json.dump({"nodes": len(edge_ids), "features": 12,
                   "hidden": 128, "epochs": 150}, f, indent=2)
    logger.info(f"GNN saved → {GNN_PATH}")


def load_gnn(net=None):
    """
    Load saved GNN + graph, or train from scratch if not found.
    Returns (model, edge_ids, graph_data) or (None, [], None) if torch unavailable.
    """
    if not gnn_available():
        logger.warning("GNN unavailable (torch not installed). Skipping.")
        return None, [], None

    import torch
    from torch_geometric.data import Data

    if GNN_PATH.exists() and GRAPH_PATH.exists():
        logger.info("Loading saved GNN model...")
        saved = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
        edge_ids = saved["edge_ids"]
        graph_data = Data(x=saved["x"], edge_index=saved["edge_index"], y=saved["y"])
        graph_data.edge_ids = edge_ids

        model = RoadGNN(in_features=12, hidden=128)
        model.load_state_dict(torch.load(GNN_PATH, map_location="cpu", weights_only=True))
        model.eval()
        logger.info(f"GNN loaded ({len(edge_ids)} road edges).")
        return model, edge_ids, graph_data

    logger.warning("No saved GNN found — training from scratch...")
    if net is None:
        import sumolib
        logger.info("Loading SUMO network for GNN graph construction...")
        net = sumolib.net.readNet(NET_FILE)

    graph_data, edge_ids = build_road_graph(net)
    model = train_gnn(graph_data)
    save_gnn(model, graph_data, edge_ids)
    return model, edge_ids, graph_data


# ── Inference: Pre-compute travel times for all edges ────────────────────────

def precompute_gnn_weights(model, graph_data, edge_ids: list,
                            conditions: dict = None) -> Dict[str, float]:
    """
    Run GNN inference on the full graph and return a dict:
        { edge_id: predicted_travel_time_seconds }

    Called at startup and whenever conditions change significantly.
    This is fast (single forward pass) and the results are cached
    in EnsemblePredictor for use during route computation.
    """
    if not gnn_available() or model is None:
        return {}

    import torch

    cond = conditions or {}
    weather_factor = {"clear": 0.0, "rain": 0.5, "fog": 1.0}.get(
        cond.get("weather", "clear"), 0.0)
    speed_mult = {"clear": 1.0, "rain": 0.75, "fog": 0.60}.get(
        cond.get("weather", "clear"), 1.0)
    # Hazard speed multiplier
    hazard_mult = {"none": 1.0, "accident": 0.50,
                   "construction": 0.55, "flood": 0.35}.get(
        cond.get("road_hazard", "none"), 1.0)
    combined_speed = speed_mult * hazard_mult

    # Adjust feature matrix for current conditions
    x = graph_data.x.clone()
    # mean_speed column (index 1) × combined speed factor
    x[:, 1] = x[:, 1] * combined_speed
    # weather_factor column (index 11)
    x[:, 11] = weather_factor

    device = next(model.parameters()).device
    x = x.to(device)
    ei = graph_data.edge_index.to(device)

    with torch.no_grad():
        preds = model(x, ei).cpu().numpy()

    weights = {}
    for eid, p in zip(edge_ids, preds):
        weights[eid] = float(max(p, 0.1))

    logger.info(f"GNN weights precomputed for {len(weights)} edges "
                f"(conditions: {cond})")
    return weights


if __name__ == "__main__":
    """Run standalone to pre-build/train the GNN graph."""
    if not gnn_available():
        print("❌ PyTorch or torch_geometric not installed.")
        print("   pip install torch torch_geometric")
        raise SystemExit(1)
    import sumolib
    print("Loading SUMO network...")
    net = sumolib.net.readNet(NET_FILE)
    graph_data, edge_ids = build_road_graph(net)
    print(f"Graph built: {len(edge_ids)} nodes")
    model = train_gnn(graph_data, epochs=150)
    save_gnn(model, graph_data, edge_ids)
    print("✅ GNN trained and saved!")
