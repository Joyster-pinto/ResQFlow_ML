"""
ResQFlow ML - Route Comparison Engine
Compares two routing strategies:
  1. Dijkstra / SUMO shortest-path (distance-based)
  2. ML-predicted quickest path (travel-time-based using LightGBM)

Uses TraCI when a live simulation is running, or falls back to sumolib
graph traversal for offline operation.

Supports condition-aware ML inference:
  conditions = {"weather": "rain"|"fog"|"clear",
                "time_of_day": "rush"|"offpeak"|"auto",
                "incident_type": "medical"|"fire"|"police"}
"""

import os
import sys
import math
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import sumolib

logger = logging.getLogger(__name__)

NET_FILE = str(Path(__file__).parent.parent / "sumo" / "scenarios" / "mysore.net.xml")

# Singleton network
_NET = None


def get_net():
    global _NET
    if _NET is None:
        logger.info("Loading SUMO network...")
        _NET = sumolib.net.readNet(NET_FILE)
        logger.info("Network loaded.")
    return _NET


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------

_WEATHER_SPEED_FACTOR = {
    "clear":  1.00,
    "rain":   0.75,   # 25% slower in rain
    "fog":    0.60,   # 40% slower in fog
}

_TIME_PEAK_OVERRIDE = {
    "rush":    1,
    "offpeak": 0,
    "auto":    None,  # use simulation time
}


def _condition_factors(conditions: dict) -> dict:
    """Derive scalar adjustment factors from condition dict."""
    weather      = (conditions or {}).get("weather", "clear")
    time_of_day  = (conditions or {}).get("time_of_day", "auto")

    speed_factor = _WEATHER_SPEED_FACTOR.get(weather, 1.0)
    peak_override = _TIME_PEAK_OVERRIDE.get(time_of_day, None)

    # Weather factor: 0.0=clear, 0.5=rain, 1.0=heavy fog
    weather_factor_scalar = {"clear": 0.0, "rain": 0.5, "fog": 1.0}.get(weather, 0.0)

    return {
        "speed_factor":        speed_factor,
        "peak_override":       peak_override,
        "weather_factor":      weather_factor_scalar,
    }


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _live_edge_features(edge_id: str, traci_module=None, conditions: dict = None) -> dict:
    """Get live TraCI edge features, or return defaults if unavailable.
    Applies condition-based overrides on top of live data.
    """
    factors = _condition_factors(conditions)
    base_speed = 13.89 * factors["speed_factor"]

    defaults = {
        "vehicle_count": 0,
        "mean_speed":    base_speed,
        "occupancy":     0.0,
        "waiting_time":  0.0,
        "hour_of_day":   8,
        "is_peak_hour":  1 if factors["peak_override"] is None else factors["peak_override"],
        "weather_factor": factors["weather_factor"],
    }

    if traci_module is None:
        return defaults

    try:
        import traci as tc
        vcount = tc.edge.getLastStepVehicleNumber(edge_id)
        speed  = tc.edge.getLastStepMeanSpeed(edge_id)
        occ    = tc.edge.getLastStepOccupancy(edge_id)
        wait   = tc.edge.getWaitingTime(edge_id)
        t      = tc.simulation.getTime()
        hour   = int((t % 86400) / 3600)

        # Apply weather speed factor to live speed
        adj_speed = max(speed, 0.5) * factors["speed_factor"]

        # Peak hour: use override if set, otherwise derive from sim time
        if factors["peak_override"] is not None:
            is_peak = factors["peak_override"]
        else:
            is_peak = 1 if hour in list(range(7, 10)) + list(range(17, 20)) else 0

        return {
            "vehicle_count":  vcount,
            "mean_speed":     adj_speed,
            "occupancy":      occ,
            "waiting_time":   wait,
            "hour_of_day":    hour,
            "is_peak_hour":   is_peak,
            "weather_factor": factors["weather_factor"],
        }
    except Exception:
        return defaults


def _build_feature_vector(edge, live: dict) -> np.ndarray:
    """Build the full feature vector for the ML model."""
    max_speed   = edge.getSpeed()
    edge_length = edge.getLength()
    lane_count  = edge.getLaneNumber()
    speed_ratio = live["mean_speed"] / max(max_speed, 0.1)
    density     = live["vehicle_count"] / max(edge_length, 1.0)
    
    return np.array([[
        live["vehicle_count"],
        live["mean_speed"],
        live["occupancy"],
        live["waiting_time"],
        edge_length,
        max_speed,
        lane_count,
        live["hour_of_day"],
        live["is_peak_hour"],
        speed_ratio,
        density,
        live["weather_factor"],   # NEW feature
    ]])


# ---------------------------------------------------------------------------
# Route finders
# ---------------------------------------------------------------------------

def dijkstra_route(src_edge_id: str, dst_edge_id: str, conditions: dict = None) -> dict:
    """
    Find the shortest path by distance using a manual Dijkstra traversal.
    Applies condition speed factor to travel time estimate.
    """
    net = get_net()
    try:
        src = net.getEdge(src_edge_id)
        dst = net.getEdge(dst_edge_id)
    except Exception as e:
        return {"error": str(e), "edges": [], "distance": 0, "travel_time": 0}

    factors = _condition_factors(conditions)

    def length_weight(edge):
        return edge.getLength()

    result_path = _dijkstra_with_weights(net, src, dst, length_weight, max_edges=5000)

    if result_path is None:
        return {"error": "No path found", "edges": [], "distance": 0, "travel_time": 0}

    path, total_dist = result_path
    edges    = [e.getID() for e in path]
    distance = sum(e.getLength() for e in path)
    # Apply speed factor to estimated time
    est_time = sum(
        e.getLength() / max(e.getSpeed() * factors["speed_factor"], 0.1)
        for e in path
    )
    coords   = _path_coordinates(path)

    return {
        "algorithm":   "dijkstra",
        "edges":       edges,
        "distance":    round(distance, 1),
        "travel_time": round(est_time, 1),
        "coordinates": coords,
        "hops":        len(edges),
    }


def ml_route(src_edge_id: str, dst_edge_id: str, model, traci_module=None, conditions: dict = None) -> dict:
    """
    Find the quickest path using ML-predicted travel times as edge weights.
    Conditions influence live feature extraction (weather, peak hour).
    """
    net = get_net()
    try:
        src = net.getEdge(src_edge_id)
        dst = net.getEdge(dst_edge_id)
    except Exception as e:
        return {"error": str(e), "edges": [], "distance": 0, "travel_time": 0}
    
    def ml_edge_weight(edge):
        live = _live_edge_features(edge.getID(), traci_module, conditions)
        feats = _build_feature_vector(edge, live)
        try:
            pred = float(model.predict(feats)[0])
            return max(pred, 0.1)
        except Exception:
            factors = _condition_factors(conditions)
            return edge.getLength() / max(edge.getSpeed() * factors["speed_factor"], 0.1)
    
    result_path = _dijkstra_with_weights(net, src, dst, ml_edge_weight, max_edges=5000)
    
    if result_path is None:
        return {"error": "No ML path found", "edges": [], "distance": 0, "travel_time": 0}
    
    path, total_cost = result_path
    edges    = [e.getID() for e in path]
    distance = sum(e.getLength() for e in path)
    coords   = _path_coordinates(path)
    
    return {
        "algorithm":   "ml_lightgbm",
        "edges":       edges,
        "distance":    round(distance, 1),
        "travel_time": round(total_cost, 1),
        "coordinates": coords,
        "hops":        len(edges),
    }


def _dijkstra_with_weights(net, src_edge, dst_edge, weight_fn, max_edges=500):
    """
    Manual Dijkstra using a callable weight function per edge.
    Searches edge-by-edge (not node-by-node) for SUMO compatibility.
    Limited to max_edges expansions to stay fast.
    """
    import heapq
    import itertools

    counter = itertools.count()   # unique tiebreaker — prevents Edge comparison

    start_cost = weight_fn(src_edge)
    heap = [(start_cost, next(counter), src_edge.getID(), [src_edge])]
    visited = set()
    expansions = 0

    while heap and expansions < max_edges:
        cost, _, eid, path = heapq.heappop(heap)

        if eid in visited:
            continue
        visited.add(eid)
        expansions += 1

        if eid == dst_edge.getID():
            return path, cost

        try:
            current_edge = net.getEdge(eid)
            to_node = current_edge.getToNode()
        except Exception:
            continue

        for out_edge in to_node.getOutgoing():
            nid = out_edge.getID()
            if nid not in visited:
                try:
                    edge_cost = weight_fn(out_edge)
                    heapq.heappush(heap, (cost + edge_cost, next(counter), nid, path + [out_edge]))
                except Exception:
                    pass

    return None


def _path_coordinates(path):
    """Extract lat/lon waypoints from a path of sumolib edges."""
    net = get_net()
    coords = []
    for edge in path:
        try:
            for x, y in edge.getShape():
                lon, lat = net.convertXY2LonLat(x, y)
                coords.append({"lat": round(lat, 6), "lon": round(lon, 6)})
        except Exception:
            continue
    return coords


# ---------------------------------------------------------------------------
# High-level comparison API
# ---------------------------------------------------------------------------

def compare_routes(src_edge_id: str, dst_edge_id: str, model, traci_module=None,
                   conditions: dict = None) -> dict:
    """
    Run both routing algorithms and return a side-by-side comparison.
    conditions are forwarded to both algorithms for consistent feature extraction.
    """
    dijkstra = dijkstra_route(src_edge_id, dst_edge_id, conditions)
    ml       = ml_route(src_edge_id, dst_edge_id, model, traci_module, conditions)
    
    improvement = 0.0
    if dijkstra.get("travel_time", 0) > 0 and ml.get("travel_time", 0) > 0:
        improvement = (
            (dijkstra["travel_time"] - ml["travel_time"]) / dijkstra["travel_time"]
        ) * 100
    
    return {
        "dijkstra":        dijkstra,
        "ml":              ml,
        "improvement_pct": round(improvement, 1),
        "winner":          "ml" if improvement > 0 else "dijkstra",
        "conditions":      conditions or {},
    }


def find_nearest_edge(lat: float, lon: float, radius: float = 300) -> Optional[str]:
    """Find the nearest SUMO edge to a geographic coordinate."""
    net = get_net()
    x, y = net.convertLonLat2XY(lon, lat)

    for r in [radius, 500, 1000, 2000]:
        try:
            edges = net.getNeighboringEdges(x, y, r)
        except Exception:
            edges = []

        if edges:
            try:
                edges.sort(key=lambda e: e[1])
                best = edges[0][0]
            except (TypeError, IndexError):
                best = edges[0]

            try:
                if best.allows("passenger"):
                    return best.getID()
                for item in edges:
                    try:
                        edge = item[0]
                    except (TypeError, IndexError):
                        edge = item
                    if edge.allows("passenger"):
                        return edge.getID()
            except Exception:
                try:
                    return best.getID()
                except Exception:
                    pass

    logger.warning(f"No edge found near lat={lat}, lon={lon}")
    return None
