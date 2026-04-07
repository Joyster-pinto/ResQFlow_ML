"""
ResQFlow ML - Route Comparison Engine
Compares two routing strategies:
  1. Dijkstra / SUMO shortest-path (distance-based)
  2. Ensemble ML path (LightGBM + CatBoost + GNN, travel-time-based)

Uses TraCI when a live simulation is running, or falls back to sumolib
graph traversal for offline operation.

Supports condition-aware ML inference:
  conditions = {"weather": "rain"|"fog"|"clear",
                "time_of_day": "rush"|"offpeak"|"auto",
                "day_type": "weekday"|"weekend",
                "road_hazard": "none"|"accident"|"construction"|"flood",
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

def _live_edge_features(edge_id: str, live_vehicles: dict = None, conditions: dict = None) -> dict:
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

    if live_vehicles is None:
        return defaults

    # ── Live Data Calculation from Vehicle Dictionary ────────
    try:
        # Avoid direct TraCI IPC calls to prevent Dijkstra freezing
        import datetime
        now = datetime.datetime.now()
        
        edge_vehs = [v for v in live_vehicles.values() if v["edge"] == edge_id]
        vcount = len(edge_vehs)
        if vcount > 0:
            avg_speed = sum(v["speed"] / 3.6 for v in edge_vehs) / vcount
        else:
            avg_speed = base_speed

        adj_speed = max(avg_speed, 0.5) * factors["speed_factor"]

        # Apply hazard waiting-time penalty on top of live data
        _, hazard_wait_add, hazard_count_mult = \
            _HAZARD_MODIFIERS.get((conditions or {}).get("road_hazard", "none"),
                                   _HAZARD_MODIFIERS["none"])   
        
        # Approximate waiting time based on stopped vehicles
        stopped = sum(1 for v in edge_vehs if v["speed"] < 1.0)
        adj_wait = (stopped * 5.0) + hazard_wait_add

        adj_count = round(vcount * hazard_count_mult)

        return {
            "vehicle_count":  adj_count,
            "mean_speed":     round(adj_speed, 3),
            "occupancy":      min(vcount * 5.0 / 100.0, 1.0), # Approximate length of vehicle = 5m
            "waiting_time":   round(adj_wait, 1),
            "hour_of_day":    defaults["hour_of_day"],
            "is_peak_hour":   defaults["is_peak_hour"],
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
    Find the shortest path by distance using sumolib's built-in shortest-path.
    Falls back to manual Dijkstra if sumolib route unavailable.
    Applies condition speed factor to travel time estimate.
    """
    net = get_net()
    try:
        src = net.getEdge(src_edge_id)
        dst = net.getEdge(dst_edge_id)
    except Exception as e:
        return {"error": str(e), "edges": [], "distance": 0, "travel_time": 0,
                "coordinates": [], "hops": 0}

    factors = _condition_factors(conditions)
    path    = None

    # ── Try sumolib built-in shortest path first (fast, handles large networks) ──
    try:
        edges, cost = net.getShortestPath(src, dst, vClass="passenger")
        if edges:
            path = edges
    except Exception as e:
        logger.debug(f"sumolib getShortestPath failed: {e} — falling back to manual Dijkstra")

    # ── Manual Dijkstra fallback ──────────────────────────────────────────────
    if path is None:
        def length_weight(edge):
            return edge.getLength()
        result = _dijkstra_with_weights(net, src, dst, length_weight)
        if result is None:
            return {"error": "No path found between these points. Try a closer destination.",
                    "edges": [], "distance": 0, "travel_time": 0, "coordinates": [], "hops": 0}
        path, _ = result

    edges    = [e.getID() for e in path]
    distance = sum(e.getLength() for e in path)
    est_time = sum(
        e.getLength() / max(e.getSpeed() * factors["speed_factor"], 0.1)
        for e in path
    )
    coords = _path_coordinates(path)

    logger.info(f"Dijkstra path: {len(edges)} edges, {distance:.0f}m, {est_time:.1f}s")
    return {
        "algorithm":   "dijkstra",
        "edges":       edges,
        "distance":    round(distance, 1),
        "travel_time": round(est_time, 1),
        "coordinates": coords,
        "hops":        len(edges),
    }


def ml_route(src_edge_id: str, dst_edge_id: str, model_or_ensemble,
             live_vehicles=None, conditions: dict = None) -> dict:
    """
    Find the quickest path using ensemble ML travel times as edge weights.
    Accepts either an EnsemblePredictor or a bare LightGBM model.
    Uses predecessor-tracking Dijkstra for memory efficiency on large networks.
    """
    net = get_net()
    try:
        src = net.getEdge(src_edge_id)
        dst = net.getEdge(dst_edge_id)
    except Exception as e:
        return {"error": str(e), "edges": [], "distance": 0, "travel_time": 0,
                "coordinates": [], "hops": 0}

    # Detect whether we have an EnsemblePredictor or a raw model
    try:
        from ensemble import EnsemblePredictor
        _is_ensemble = isinstance(model_or_ensemble, EnsemblePredictor)
    except ImportError:
        _is_ensemble = False

    # Per-request feature/prediction cache (avoids re-computing the same edge)
    _pred_cache: dict = {}

    def ml_edge_weight(edge):
        eid = edge.getID()
        if eid not in _pred_cache:
            live = _live_edge_features(eid, live_vehicles, conditions)
            if _is_ensemble:
                _pred_cache[eid] = model_or_ensemble.predict_edge(edge, live)
            else:
                # Bare LightGBM model (backward-compat)
                feats = _build_feature_vector(edge, live)
                try:
                    p = float(model_or_ensemble.predict(feats)[0])
                    _pred_cache[eid] = max(p, 0.1)
                except Exception:
                    factors = _condition_factors(conditions)
                    _pred_cache[eid] = edge.getLength() / max(
                        edge.getSpeed() * factors["speed_factor"], 0.1)
        return _pred_cache[eid]

    result = _dijkstra_with_weights(net, src, dst, ml_edge_weight)

    if result is None:
        return {"error": "No ML path found between these points.",
                "edges": [], "distance": 0, "travel_time": 0, "coordinates": [], "hops": 0}

    path, total_cost = result
    edges    = [e.getID() for e in path]
    distance = sum(e.getLength() for e in path)
    coords   = _path_coordinates(path)

    algo_label = "ensemble" if _is_ensemble else "ml_lightgbm"
    logger.info(f"{algo_label} path: {len(edges)} edges, {distance:.0f}m, "
                f"{total_cost:.1f}s | cache={len(_pred_cache)}")
    return {
        "algorithm":   algo_label,
        "edges":       edges,
        "distance":    round(distance, 1),
        "travel_time": round(total_cost, 1),
        "coordinates": coords,
        "hops":        len(edges),
    }


def _dijkstra_with_weights(net, src_edge, dst_edge, weight_fn):
    """
    Memory-efficient Dijkstra using predecessor tracking.
    Does NOT store the full path in every heap entry — stores only the
    best-cost and predecessor per node, then reconstructs path at the end.
    This is O(E log E) instead of O(E^2) memory for large networks.
    """
    import heapq

    dst_id  = dst_edge.getID()
    src_id  = src_edge.getID()

    # dist[edge_id] = best known cost to reach this edge
    dist  = {src_id: weight_fn(src_edge)}
    prev  = {src_id: None}           # predecessor edge_id
    heap  = [(dist[src_id], src_id)]

    while heap:
        cost, eid = heapq.heappop(heap)

        # Skip stale heap entries
        if cost > dist.get(eid, float('inf')):
            continue

        if eid == dst_id:
            # Reconstruct path by following predecessors
            path_ids = []
            cur = eid
            while cur is not None:
                path_ids.append(cur)
                cur = prev[cur]
            path_ids.reverse()
            path_edges = []
            for pid in path_ids:
                try:
                    path_edges.append(net.getEdge(pid))
                except Exception:
                    pass
            return path_edges, cost

        try:
            to_node = net.getEdge(eid).getToNode()
        except Exception:
            continue

        for out_edge in to_node.getOutgoing():
            nid = out_edge.getID()
            try:
                edge_cost = weight_fn(out_edge)
                new_cost  = cost + edge_cost
                if new_cost < dist.get(nid, float('inf')):
                    dist[nid] = new_cost
                    prev[nid] = eid
                    heapq.heappush(heap, (new_cost, nid))
            except Exception:
                pass

    logger.warning(f"No path found: {src_id} → {dst_id} (explored {len(dist)} edges)")
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

def _ml_path_cost(edge_ids: list, model_or_ensemble, live_vehicles=None, conditions: dict = None) -> float:
    """
    Compute total ML-estimated travel time for a list of SUMO edge IDs.
    Used to evaluate Dijkstra's path on the same scale as the ML path.
    """
    net = get_net()
    try:
        from ensemble import EnsemblePredictor
        _is_ensemble = isinstance(model_or_ensemble, EnsemblePredictor)
    except ImportError:
        _is_ensemble = False

    total = 0.0
    for eid in edge_ids:
        try:
            edge = net.getEdge(eid)
            live = _live_edge_features(eid, live_vehicles, conditions)
            if _is_ensemble:
                cost = model_or_ensemble.predict_edge(edge, live)
            else:
                feats = _build_feature_vector(edge, live)
                try:
                    cost = float(model_or_ensemble.predict(feats)[0])
                except Exception:
                    factors = _condition_factors(conditions)
                    cost = edge.getLength() / max(
                        edge.getSpeed() * factors["speed_factor"], 0.1)
            total += max(cost, 0.1)
        except Exception:
            pass
    return total


def compare_routes(src_edge_id: str, dst_edge_id: str, model_or_ensemble,
                   live_vehicles=None, conditions: dict = None) -> dict:
    """
    Run both routing algorithms and return a side-by-side comparison.

    FAIR COMPARISON:
      Both paths are scored using the ML model so they are on the same scale.
      - Dijkstra finds the shortest-DISTANCE path, then we ask: "what would
        the ML model estimate this path takes?"
      - ML route finds the path the ML model predicts is fastest.
      - improvement_pct = how much faster ML's chosen path is vs Dijkstra's
        chosen path, both measured in ML-estimated travel time.

    This prevents the old bug where Dijkstra reported free-flow time and
    ML reported congestion-aware time, making Dijkstra always appear faster.
    """
    dijkstra = dijkstra_route(src_edge_id, dst_edge_id, conditions)
    ml       = ml_route(src_edge_id, dst_edge_id, model_or_ensemble,
                        live_vehicles, conditions)

    # Re-score Dijkstra's path using the ML model (fair apples-to-apples)
    dijkstra_ml_cost = _ml_path_cost(
        dijkstra.get("edges", []), model_or_ensemble, live_vehicles, conditions
    )
    ml_cost = ml.get("travel_time", 0)  # already ML-estimated

    improvement = 0.0
    if dijkstra_ml_cost > 0 and ml_cost > 0:
        improvement = (dijkstra_ml_cost - ml_cost) / dijkstra_ml_cost * 100

    # Also update Dijkstra's displayed travel_time to the ML-estimated value
    # so the UI comparison makes sense to the user.
    dijkstra["travel_time_freeflow"] = dijkstra["travel_time"]  # keep original
    dijkstra["travel_time"]          = round(dijkstra_ml_cost, 1)

    logger.info(
        f"Comparison | Dijkstra ML-cost={dijkstra_ml_cost:.1f}s  "
        f"ML-path cost={ml_cost:.1f}s  improvement={improvement:.1f}%"
    )

    return {
        "dijkstra":          dijkstra,
        "ml":                ml,
        "improvement_pct":   round(improvement, 1),
        "winner":            "ml" if improvement > 0 else "dijkstra",
        "conditions":        conditions or {},
        "comparison_basis":  "ml_estimated",   # tells UI both times are ML-estimated
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
