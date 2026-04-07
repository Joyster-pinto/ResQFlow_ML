"""
ResQFlow ML - SUMO TraCI Data Generator
Connects to a running SUMO simulation and collects edge-level traffic data
for training the LightGBM routing model.
"""

import os
import sys
import csv
import time
import random
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add SUMO tools to path
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci
import sumolib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "traffic_samples.csv"

SUMO_CFG = str(Path(__file__).parent.parent / "sumo" / "scenarios" / "mysore.sumocfg")
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo.exe")  # headless for data gen

CSV_HEADERS = [
    "edge_id",
    "step",
    "vehicle_count",
    "mean_speed",
    "occupancy",
    "waiting_time",
    "edge_length",
    "max_speed",
    "lane_count",
    "hour_of_day",
    "is_peak_hour",
    "travel_time",          # target variable (seconds)
    "congestion_level",     # 0=free, 1=moderate, 2=heavy
]

PEAK_HOURS = list(range(7, 10)) + list(range(17, 20))  # 7-9am, 5-8pm


def congestion_level(speed, max_speed):
    """Classify congestion from 0 (free) to 2 (heavy)."""
    if max_speed <= 0:
        return 0
    ratio = speed / max_speed
    if ratio >= 0.7:
        return 0
    elif ratio >= 0.4:
        return 1
    else:
        return 2


def spawn_background_traffic(net, num_vehicles=200):
    """Add random background vehicles to make the simulation realistic."""
    edges = [e for e in net.getEdges() if e.allows("passenger") and e.getLength() > 50]
    if len(edges) < 2:
        return
    
    for i in range(num_vehicles):
        try:
            src = random.choice(edges)
            dst = random.choice(edges)
            if src.getID() == dst.getID():
                continue
            vid = f"bg_{i}"
            traci.vehicle.add(vid, routeID="", typeID="car",
                              departLane="best", departSpeed="max")
            route = traci.simulation.findRoute(src.getID(), dst.getID())
            if route.edges:
                traci.vehicle.setRoute(vid, route.edges)
        except Exception:
            pass


def run_data_collection(steps: int = 1800, port: int = 8813):
    """
    Launches SUMO headless and collects traffic samples for ML training.
    
    Args:
        steps: Number of simulation steps to run (1 step = 1 second)
        port: TraCI port
    """
    logger.info("Loading SUMO network for topology info...")
    net_file = str(Path(__file__).parent.parent / "sumo" / "scenarios" / "mysore.net.xml")
    net = sumolib.net.readNet(net_file)
    all_edges = [e for e in net.getEdges() if e.allows("passenger")]
    
    logger.info(f"Network has {len(all_edges)} driveable edges")
    
    sumo_cmd = [
        SUMO_BINARY,
        "-c", SUMO_CFG,
        "--remote-port", str(port),
        "--no-step-log",
        "--no-warnings",
        "--collision.action", "remove",
        "--time-to-impatience", "30",
    ]
    
    logger.info("Starting SUMO simulation (headless)...")
    traci.start(sumo_cmd, port=port)
    
    samples_written = 0
    write_header = not OUTPUT_CSV.exists()
    
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        
        # Spawn some background traffic
        try:
            spawn_background_traffic(net, num_vehicles=150)
        except Exception as e:
            logger.warning(f"Could not spawn background traffic: {e}")
        
        for step in range(steps):
            traci.simulationStep()
            
            # Simulate time of day (start at 6am = step 0)
            sim_hour = (6 + step // 3600) % 24
            is_peak = 1 if sim_hour in PEAK_HOURS else 0
            
            # Sample a subset of edges every 30 steps to avoid huge files
            if step % 30 == 0:
                sample_edges = random.sample(all_edges, min(100, len(all_edges)))
                
                for edge in sample_edges:
                    eid = edge.getID()
                    try:
                        vcount = traci.edge.getLastStepVehicleNumber(eid)
                        speed  = traci.edge.getLastStepMeanSpeed(eid)
                        occ    = traci.edge.getLastStepOccupancy(eid)
                        wait   = traci.edge.getWaitingTime(eid)
                        length = edge.getLength()
                        mspeed = edge.getSpeed()
                        lanes  = edge.getLaneNumber()
                        
                        eff_speed = max(speed, 0.5)  # avoid division by zero
                        travel_time = length / eff_speed
                        cong = congestion_level(speed, mspeed)
                        
                        writer.writerow({
                            "edge_id":        eid,
                            "step":           step,
                            "vehicle_count":  vcount,
                            "mean_speed":     round(speed, 4),
                            "occupancy":      round(occ, 4),
                            "waiting_time":   round(wait, 4),
                            "edge_length":    round(length, 2),
                            "max_speed":      round(mspeed, 2),
                            "lane_count":     lanes,
                            "hour_of_day":    sim_hour,
                            "is_peak_hour":   is_peak,
                            "travel_time":    round(travel_time, 4),
                            "congestion_level": cong,
                        })
                        samples_written += 1
                    except Exception:
                        continue
            
            if step % 300 == 0:
                logger.info(f"Step {step}/{steps} | Samples: {samples_written} | Vehicles: {traci.vehicle.getIDCount()}")
    
    traci.close()
    logger.info(f"Data collection complete. {samples_written} samples written to {OUTPUT_CSV}")
    return samples_written


if __name__ == "__main__":
    run_data_collection(steps=1800)
