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
    "auto":    None,  # derive from real clock
}

# Realistic offline traffic volume defaults per time-of-day
# These are applied when SUMO/TraCI is not connected
_TIME_DEFAULT_TRAFFIC = {
    "rush":    {"vehicle_count": 18, "occupancy": 0.55, "waiting_time": 35.0},
    "offpeak": {"vehicle_count":  2, "occupancy": 0.05, "waiting_time":  1.5},
    "auto":    {"vehicle_count":  6, "occupancy": 0.18, "waiting_time":  8.0},
}

# Road hazard modifiers  (multiplied / added onto base features)
_HAZARD_MODIFIERS = {
    # hazard:  (speed_mult, waiting_time_add, vehicle_count_mult)
    "none":         (1.00, 0,    1.0),
    "accident":     (0.50, 90,   2.5),   # half speed, big queue
    "construction": (0.55, 40,   1.8),   # lane restrictions
    "flood":        (0.35, 20,   1.2),   # extreme speed drop
}

# Day-type modifiers on base vehicle count
_DAY_TYPE_TRAFFIC_SCALE = {
    "weekday": 1.0,
    "weekend": 0.45,  # ~55% less traffic on weekends
}


def _condition_factors(conditions: dict) -> dict:
    """Derive scalar adjustment factors from condition dict.
    Handles: weather, time_of_day, day_type, road_hazard.
    """
    cond = conditions or {}
    weather      = cond.get("weather",      "clear")
    time_of_day  = cond.get("time_of_day",  "auto")
    day_type     = cond.get("day_type",     "weekday")
    road_hazard  = cond.get("road_hazard",  "none")

    # ── Weather ──────────────────────────────────────────────────
    weather_speed  = _WEATHER_SPEED_FACTOR.get(weather, 1.0)
    weather_scalar = {"clear": 0.0, "rain": 0.5, "fog": 1.0}.get(weather, 0.0)

    # ── Time of day ───────────────────────────────────────────────
    peak_override   = _TIME_PEAK_OVERRIDE.get(time_of_day, None)
    traffic_defaults = dict(_TIME_DEFAULT_TRAFFIC.get(time_of_day, _TIME_DEFAULT_TRAFFIC["auto"]))

    # Weekend scales down traffic volume regardless of rush/offpeak
    day_scale = _DAY_TYPE_TRAFFIC_SCALE.get(day_type, 1.0)
    # Weekends have no real rush hour
    if day_type == "weekend" and peak_override == 1:
        peak_override = 0
    traffic_defaults["vehicle_count"] = round(traffic_defaults["vehicle_count"] * day_scale)
    traffic_defaults["occupancy"]     = round(traffic_defaults["occupancy"]     * day_scale, 3)
    traffic_defaults["waiting_time"]  = round(traffic_defaults["waiting_time"]  * day_scale, 1)

    # ── Road hazard ───────────────────────────────────────────────
    hazard_speed_mult, hazard_wait_add, hazard_count_mult = \
        _HAZARD_MODIFIERS.get(road_hazard, _HAZARD_MODIFIERS["none"])

    # Combined speed factor: weather × hazard
    combined_speed_factor = weather_speed * hazard_speed_mult

    # Apply hazard to traffic defaults
    traffic_defaults["vehicle_count"] = round(traffic_defaults["vehicle_count"] * hazard_count_mult)
    traffic_defaults["waiting_time"]  = round(traffic_defaults["waiting_time"] + hazard_wait_add, 1)

    return {
        "speed_factor":        combined_speed_factor,
        "weather_speed":       weather_speed,
        "hazard_speed_mult":   hazard_speed_mult,
        "peak_override":       peak_override,
        "weather_factor":      weather_scalar,
        "traffic_defaults":    traffic_defaults,   # realistic offline defaults
    }


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _live_edge_features(edge_id: str, traci_module=None, conditions: dict = None) -> dict:
    """Get live TraCI edge features, or return realistic condition-aware defaults.
    When SUMO/TraCI is not connected, defaults vary meaningfully with conditions
    so that Rush vs Off-Peak, day type, and hazard actually influence routing.
    """
    import datetime
    factors = _condition_factors(conditions)
    td      = factors["traffic_defaults"]

    # Derive hour from real wall clock (used when SUMO not connected)
    now  = datetime.datetime.now()
    hour = now.hour

    # Peak hour: use condition override, or derive from actual clock
    if factors["peak_override"] is not None:
        is_peak = int(factors["peak_override"])
    else:
        is_peak = 1 if hour in list(range(7, 10)) + list(range(17, 20)) else 0

    # Realistic base speed: speed limit × combined factor (varies per edge at call time)
    # We approximate base as 13.89 m/s (50 km/h) scaled by conditions
    base_speed = 13.89 * factors["speed_factor"]

    # ── Offline defaults — now actually vary with conditions ──────
    defaults = {
        "vehicle_count": td["vehicle_count"],
        "mean_speed":    round(base_speed, 3),
        "occupancy":     td["occupancy"],
        "waiting_time":  td["waiting_time"],
        "hour_of_day":   hour,
        "is_peak_hour":  is_peak,
        "weather_factor": factors["weather_factor"],
    }

    if traci_module is None:
        return defaults

    # ── Live TraCI data — apply condition overrides on top ────────
    try:
        import traci as tc
        vcount = tc.edge.getLastStepVehicleNumber(edge_id)
        speed  = tc.edge.getLastStepMeanSpeed(edge_id)
        occ    = tc.edge.getLastStepOccupancy(edge_id)
        wait   = tc.edge.getWaitingTime(edge_id)
        t      = tc.simulation.getTime()
        hour   = int((t % 86400) / 3600)

        # Apply all condition speed factors to live speed
        adj_speed = max(speed, 0.5) * factors["speed_factor"]

        # Apply hazard waiting-time penalty on top of live data
        _, hazard_wait_add, hazard_count_mult = \
            _HAZARD_MODIFIERS.get((conditions or {}).get("road_hazard", "none"),
                                   _HAZARD_MODIFIERS["none"])
        adj_wait  = wait  + hazard_wait_add
        adj_count = round(vcount * hazard_count_mult)

        if factors["peak_override"] is not None:
            is_peak = int(factors["peak_override"])
        else:
            is_peak = 1 if hour in list(range(7, 10)) + list(range(17, 20)) else 0

        return {
            "vehicle_count":  adj_count,
            "mean_speed":     round(adj_speed, 3),
            "occupancy":      occ,
            "waiting_time":   round(adj_wait, 1),
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
