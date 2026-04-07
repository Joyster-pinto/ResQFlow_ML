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
        "lat":  12.3131,   # verified: Wikimapia / findlatitudeandlongitude.com
        "lon":  76.6496,
        "icon": "🏥",
    },
    {
        "id":   "H002",
        "name": "Cheluvamba Hospital",
        "type": "hospital",
        "lat":  12.3141,   # verified: Wikimapia / Mapcarta (was 12.2958 — wrong area)
        "lon":  76.6493,
        "icon": "🏥",
    },
    {
        "id":   "H003",
        "name": "JSS Hospital",
        "type": "hospital",
        "lat":  12.2957,   # verified: findlatitudeandlongitude.com (was 12.3077 — wrong)
        "lon":  76.6536,
        "icon": "🏥",
    },
    {
        "id":   "H004",
        "name": "Columbia Asia Hospital",
        "type": "hospital",
        "lat":  12.3514,   # verified: Wikimedia (Bannimantap Ring Road — was 12.3156 — 4 km off)
        "lon":  76.6611,
        "icon": "🏥",
    },
    {
        "id":   "H005",
        "name": "Basappa Memorial Hospital",
        "type": "hospital",
        "lat":  12.3207,   # verified: Mapcarta (was 12.3104 / 76.6585 — wrong area)
        "lon":  76.6204,
        "icon": "🏥",
    },
    # Fire Stations
    {
        "id":   "F001",
        "name": "City Fire Station (Sayyaji Rao Rd)",
        "type": "fire",
        "lat":  12.3128,   # central Mysore fire station near Sayyaji Rao Rd
        "lon":  76.6528,
        "icon": "🚒",
    },
    {
        "id":   "F002",
        "name": "Vijayanagar Fire Station",
        "type": "fire",
        "lat":  12.3358,   # verified: Wikimapia 12°20'08"N 76°36'03"E
        "lon":  76.6008,
        "icon": "🚒",
    },
    # Police
    {
        "id":   "P001",
        "name": "Mysore City Police HQ",
        "type": "police",
        "lat":  12.3025,   # verified: Police Commissioner Office, Mirza Rd, Nazarbad
        "lon":  76.6575,
        "icon": "🚔",
    },
    {
        "id":   "P002",
        "name": "Vijayanagar Police Station",
        "type": "police",
        "lat":  12.3368,   # verified: Mapcarta 12°20'13"N 76°36'6"E
        "lon":  76.6017,
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
