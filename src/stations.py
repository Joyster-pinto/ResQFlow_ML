"""
ResQFlow ML — Station Registry
Defines all dispatch-eligible stations (hospitals, fire, police) in Mysore.
Edge IDs are resolved lazily on first access so the SUMO network only loads once.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Station definitions  (lat / lon from OpenStreetMap)
# ---------------------------------------------------------------------------
STATIONS_RAW = [
    # Hospitals / Medical
    {
        "id":   "H001",
        "name": "K.R. Hospital",
        "type": "hospital",
        "lat":  12.3045,
        "lon":  76.6550,
        "icon": "🏥",
    },
    {
        "id":   "H002",
        "name": "Cheluvamba Hospital",
        "type": "hospital",
        "lat":  12.2958,
        "lon":  76.6429,
        "icon": "🏥",
    },
    {
        "id":   "H003",
        "name": "JSS Hospital",
        "type": "hospital",
        "lat":  12.3077,
        "lon":  76.6247,
        "icon": "🏥",
    },
    {
        "id":   "H004",
        "name": "Columbia Asia Hospital",
        "type": "hospital",
        "lat":  12.3156,
        "lon":  76.6421,
        "icon": "🏥",
    },
    {
        "id":   "H005",
        "name": "Basappa Memorial Hospital",
        "type": "hospital",
        "lat":  12.3104,
        "lon":  76.6585,
        "icon": "🏥",
    },
    # Fire Stations
    {
        "id":   "F001",
        "name": "City Fire Station (Sayyaji Rao Rd)",
        "type": "fire",
        "lat":  12.3089,
        "lon":  76.6520,
        "icon": "🚒",
    },
    {
        "id":   "F002",
        "name": "Vijayanagar Fire Station",
        "type": "fire",
        "lat":  12.3200,
        "lon":  76.6150,
        "icon": "🚒",
    },
    # Police
    {
        "id":   "P001",
        "name": "Mysore City Police HQ",
        "type": "police",
        "lat":  12.3000,
        "lon":  76.6550,
        "icon": "🚔",
    },
    {
        "id":   "P002",
        "name": "Vijayanagar Police Station",
        "type": "police",
        "lat":  12.3180,
        "lon":  76.6090,
        "icon": "🚔",
    },
]

# Resolved cache: id → edge_id
_RESOLVED: dict = {}


def resolve_stations(find_nearest_edge_fn) -> list[dict]:
    """
    Resolve each station's lat/lon to the nearest SUMO edge.
    Caches results so this only runs once per session.
    Returns a list of station dicts with 'edge_id' populated.
    """
    global _RESOLVED

    result = []
    for s in STATIONS_RAW:
        sid = s["id"]
        if sid not in _RESOLVED:
            edge = find_nearest_edge_fn(s["lat"], s["lon"])
            if edge:
                _RESOLVED[sid] = edge
                logger.info(f"Station {sid} ({s['name']}) → edge {edge}")
            else:
                logger.warning(f"Station {sid} ({s['name']}) could not be resolved to an edge")
                _RESOLVED[sid] = None

        result.append({
            **s,
            "edge_id": _RESOLVED[sid],
        })

    return result


def get_station_by_id(station_id: str) -> Optional[dict]:
    """Return raw station dict by ID."""
    for s in STATIONS_RAW:
        if s["id"] == station_id:
            return s
    return None
