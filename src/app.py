"""
ResQFlow ML - Flask + Socket.IO Application Server
Real-time web dashboard for ML-based emergency vehicle routing.
Station-based dispatch: origin is always a predefined station,
destination is chosen on the map. SUMO is launched automatically.
"""

import os
import sys
import json
import time
import logging
import threading
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
sys.path.append(os.path.join(SUMO_HOME, "tools"))

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from router   import compare_routes, find_nearest_edge, get_net
from trainer  import load_model, train
from stations import resolve_stations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Flask App Setup ──────────────────────────────────────────────────────────
app     = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "resqflow_secret")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global JSON error handlers ────────────────────────────────────────────────
@app.errorhandler(404)
def err_404(e):
    return jsonify({"error": "Not found", "detail": str(e)}), 404

@app.errorhandler(500)
def err_500(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.errorhandler(Exception)
def err_any(e):
    logger.exception("Unhandled exception")
    return jsonify({"error": str(e)}), 500

# ── Global State ─────────────────────────────────────────────────────────────
STATE = {
    "sumo_running":    False,
    "traci_connected": False,
    "sim_step":        0,
    "vehicles":        {},
    "model_loaded":    False,
    "model_meta":      {},
    "active_missions": [],
    "stations":        [],          # cached resolved station list
}

_model      = None
_traci      = None
_sim_thread = None
_sumo_proc  = None                  # subprocess handle for sumo-gui

META_PATH = Path(__file__).parent.parent / "models" / "model_meta.json"


def load_model_state():
    """Load or train the ML model."""
    global _model, STATE
    logger.info("Loading ML model...")
    try:
        _model = load_model()
        STATE["model_loaded"] = True
        if META_PATH.exists():
            with open(META_PATH) as f:
                STATE["model_meta"] = json.load(f)
        logger.info("ML model ready.")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        STATE["model_loaded"] = False


def _preload_stations():
    """Resolve all station edges in the background after startup."""
    try:
        resolved = resolve_stations(find_nearest_edge)
        STATE["stations"] = resolved
        logger.info(f"Stations resolved: {len(resolved)} stations ready.")
    except Exception as e:
        logger.error(f"Station resolution failed: {e}")


# ── SUMO / TraCI Integration ─────────────────────────────────────────────────

def launch_sumo_process(conditions: dict = None) -> bool:
    """
    Launch sumo-gui as a subprocess.
    conditions: {"weather": "rain"|"fog"|"clear", "time_of_day": "rush"|"offpeak"|"auto"}
    Returns True if process started successfully.
    """
    global _sumo_proc, STATE

    conditions = conditions or {}

    # Kill any existing SUMO process
    if _sumo_proc and _sumo_proc.poll() is None:
        logger.info("Terminating existing SUMO process...")
        try:
            _sumo_proc.terminate()
            _sumo_proc.wait(timeout=5)
        except Exception:
            pass

    sumo_binary = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
    sumo_cfg    = str(Path(__file__).parent.parent / "sumo" / "scenarios" / "mysore.sumocfg")

    if not Path(sumo_binary).exists():
        # Try plain name if on PATH
        sumo_binary = "sumo-gui"
        logger.warning(f"sumo-gui.exe not found at {os.path.join(SUMO_HOME, 'bin')} — falling back to PATH")

    # NOTE: Only use valid sumo-gui CLI flags here.
    # Condition effects (weather/time) are handled in the ML feature vector, NOT via SUMO flags.
    cmd = [
        sumo_binary,
        "-c", sumo_cfg,
        "--remote-port", "8813",
        "--start",                  # auto-start simulation
        "--quit-on-end", "false",   # keep GUI open after sim ends
    ]

    logger.info(f"Launching SUMO-GUI: {' '.join(cmd)}")
    try:
        _sumo_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE,  # show SUMO in its own window
        )
        STATE["sumo_running"] = False   # flips to True when TraCI connects
        logger.info(f"SUMO-GUI process started (PID {_sumo_proc.pid})")
        return True
    except FileNotFoundError:
        logger.error(f"sumo-gui binary not found at: {sumo_binary}. Set SUMO_HOME in .env")
        return False
    except Exception as e:
        logger.error(f"SUMO launch failed: {e}")
        return False


def connect_traci(port: int = 8813, delay: float = 8.0):
    """
    Attempt to connect TraCI to a running SUMO instance (non-blocking).
    delay: seconds to wait before first attempt (SUMO needs time to load the network).
    The Mysore network is large — allow at least 8 seconds.
    """
    global _traci, _sim_thread, STATE

    def _do_connect():
        global _traci, _sim_thread, STATE
        import traci

        # Close any stale connection
        try:
            traci.close()
        except Exception:
            pass

        logger.info(f"Waiting {delay}s for SUMO to load the network...")
        socketio.emit("sumo_status", {"phase": "loading", "msg": f"SUMO loading network ({delay:.0f}s)…"})
        time.sleep(delay)

        # Check if the process died during startup
        if _sumo_proc and _sumo_proc.poll() is not None:
            stderr_out = ""
            try:
                stderr_out = _sumo_proc.stderr.read().decode(errors="replace").strip()
            except Exception:
                pass
            err_msg = f"SUMO process exited early (code {_sumo_proc.returncode}). {stderr_out[:300]}"
            logger.error(err_msg)
            STATE["traci_connected"] = False
            STATE["sumo_running"]    = False
            socketio.emit("status", {
                "model_loaded":    STATE["model_loaded"],
                "sumo_running":    False,
                "traci_connected": False,
                "model_meta":      STATE["model_meta"],
                "error":           err_msg,
            })
            return

        # Attempt TraCI connection with limited retries
        MAX_ATTEMPTS = 10
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                traci.init(port=port, numRetries=0)  # one shot per loop iteration
                _traci = traci
                STATE["traci_connected"] = True
                STATE["sumo_running"]    = True
                logger.info(f"TraCI connected on port {port} (attempt {attempt})")
                socketio.emit("status", {
                    "model_loaded":    STATE["model_loaded"],
                    "sumo_running":    True,
                    "traci_connected": True,
                    "model_meta":      STATE["model_meta"],
                })
                simulation_loop()
                return
            except Exception as e:
                # Check if SUMO died between retries
                if _sumo_proc and _sumo_proc.poll() is not None:
                    logger.error(f"SUMO process died during TraCI connect (attempt {attempt})")
                    break
                logger.warning(f"TraCI attempt {attempt}/{MAX_ATTEMPTS}: {e}")
                time.sleep(2)

        # All attempts exhausted
        err_msg = f"Could not connect to SUMO TraCI on port {port} after {MAX_ATTEMPTS} attempts."
        logger.error(err_msg)
        STATE["traci_connected"] = False
        STATE["sumo_running"]    = False
        socketio.emit("status", {
            "model_loaded":    STATE["model_loaded"],
            "sumo_running":    False,
            "traci_connected": False,
            "model_meta":      STATE["model_meta"],
            "error":           err_msg,
        })

    t = threading.Thread(target=_do_connect, daemon=True)
    t.start()
    return True


def simulation_loop():
    """Background thread: advances SUMO one step at a time and streams data."""
    global STATE
    step_interval = float(os.environ.get("SIM_STEP_LENGTH", 1))
    
    while STATE["sumo_running"] and _traci:
        try:
            _traci.simulationStep()
            STATE["sim_step"] += 1
            
            # Collect vehicle positions
            vehicles = {}
            for vid in _traci.vehicle.getIDList():
                try:
                    x, y    = _traci.vehicle.getPosition(vid)
                    net     = get_net()
                    lon, lat = net.convertXY2LonLat(x, y)
                    speed    = _traci.vehicle.getSpeed(vid)
                    edge_id  = _traci.vehicle.getRoadID(vid)
                    vtype    = _traci.vehicle.getTypeID(vid)
                    vehicles[vid] = {
                        "lat":     round(lat, 6),
                        "lon":     round(lon, 6),
                        "speed":   round(speed * 3.6, 1),  # m/s → km/h
                        "edge":    edge_id,
                        "type":    vtype,
                    }
                except Exception:
                    pass
            
            STATE["vehicles"] = vehicles
            
            # Push update to all connected clients every 2 steps
            if STATE["sim_step"] % 2 == 0:
                socketio.emit("sim_update", {
                    "step":     STATE["sim_step"],
                    "vehicles": vehicles,
                    "count":    len(vehicles),
                })
            
            time.sleep(step_interval * 0.1)  # 10x realtime speed
            
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
            STATE["sumo_running"]    = False
            STATE["traci_connected"] = False
            break
    
    logger.info("Simulation loop ended.")


# ── REST API Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    net = get_net()
    return jsonify({
        "sumo_running":    STATE["sumo_running"],
        "traci_connected": STATE["traci_connected"],
        "sim_step":        STATE["sim_step"],
        "model_loaded":    STATE["model_loaded"],
        "model_meta":      STATE["model_meta"],
        "vehicle_count":   len(STATE["vehicles"]),
        "net_loaded":      net is not None,
    })


@app.route("/api/stations")
def api_stations():
    """Return the resolved list of dispatch stations."""
    if not STATE["stations"]:
        # Resolve on-demand if not yet done
        _preload_stations()
    return jsonify(STATE["stations"])


@app.route("/api/launch_sumo", methods=["POST"])
def api_launch_sumo():
    """
    Launch sumo-gui with optional condition parameters, then auto-connect TraCI.
    Body: { "weather": "clear"|"rain"|"fog",
            "time_of_day": "auto"|"rush"|"offpeak",
            "incident_type": "medical"|"fire"|"police" }
    """
    data       = request.get_json(force=True, silent=True) or {}
    conditions = {
        "weather":      data.get("weather",      "clear"),
        "time_of_day":  data.get("time_of_day",  "auto"),
        "incident_type": data.get("incident_type", "medical"),
    }

    ok = launch_sumo_process(conditions)
    if not ok:
        return jsonify({"ok": False, "error": "sumo-gui binary not found. Check SUMO_HOME in .env"}), 500

    # Store conditions in STATE for ML feature override
    STATE["conditions"] = conditions

    # Auto-connect TraCI after SUMO starts (3-second delay)
    connect_traci(port=8813, delay=3.0)

    return jsonify({
        "ok":      True,
        "message": "SUMO launching… TraCI will connect automatically in ~3 seconds.",
        "conditions": conditions,
    })


@app.route("/api/connect_sumo", methods=["POST"])
def api_connect_sumo():
    """Manual TraCI connect (fallback for already-running SUMO)."""
    data = request.get_json(force=True, silent=True) or {}
    port = data.get("port", 8813)
    connect_traci(port, delay=0)
    return jsonify({"ok": True, "message": f"Connecting to SUMO on port {port}…"})


@app.route("/api/route", methods=["POST"])
def api_route():
    """Compute and compare routes from a station to a destination."""
    try:
        data = request.get_json(force=True, silent=True) or {}

        src_edge = data.get("src_edge")
        dst_edge = data.get("dst_edge")

        # Resolve lat/lon → edge if provided
        if not src_edge and data.get("src_lat") is not None:
            src_edge = find_nearest_edge(float(data["src_lat"]), float(data["src_lon"]))
        if not dst_edge and data.get("dst_lat") is not None:
            dst_edge = find_nearest_edge(float(data["dst_lat"]), float(data["dst_lon"]))

        if not src_edge or not dst_edge:
            return jsonify({"error": "Could not resolve source/destination to road edges. Try clicking closer to a road."}), 400

        # Merge session conditions with per-request overrides
        conditions = {**STATE.get("conditions", {}), **data.get("conditions", {})}

        logger.info(f"Routing: {src_edge} → {dst_edge} | conditions={conditions}")
        result = compare_routes(src_edge, dst_edge, _model, _traci, conditions)

        # Store as active mission
        mission = {
            "id":          f"M{len(STATE['active_missions'])+1:03d}",
            "src_edge":    src_edge,
            "dst_edge":    dst_edge,
            "result":      result,
            "conditions":  conditions,
            "timestamp":   time.time(),
        }
        STATE["active_missions"].append(mission)
        socketio.emit("new_mission", mission)
        return jsonify(result)

    except Exception as e:
        logger.exception("Route computation error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """Trigger ML model re-training."""
    def _train_bg():
        global _model, STATE
        try:
            socketio.emit("training_status", {"status": "started"})
            model, meta = train()
            _model = model
            STATE["model_meta"]  = meta
            STATE["model_loaded"] = True
            socketio.emit("training_status", {"status": "complete", "meta": meta})
        except Exception as e:
            socketio.emit("training_status", {"status": "error", "message": str(e)})
    
    threading.Thread(target=_train_bg, daemon=True).start()
    return jsonify({"ok": True, "message": "Training started in background"})


@app.route("/api/missions")
def api_missions():
    return jsonify(STATE["active_missions"][-20:])  # last 20


@app.route("/api/network/info")
def api_network_info():
    """Return network bounding box and center for map initialization."""
    net = get_net()
    bbox = net.getBoundary()  # (x_min, y_min, x_max, y_max) in XY
    
    lon_min, lat_min = net.convertXY2LonLat(bbox[0], bbox[1])
    lon_max, lat_max = net.convertXY2LonLat(bbox[2], bbox[3])
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2
    
    return jsonify({
        "center": {"lat": round(center_lat, 6), "lon": round(center_lon, 6)},
        "bounds": {
            "sw": {"lat": round(lat_min, 6), "lon": round(lon_min, 6)},
            "ne": {"lat": round(lat_max, 6), "lon": round(lon_max, 6)},
        },
        "edges": net.getEdgeCount() if hasattr(net, "getEdgeCount") else "N/A",
    })


@app.route("/api/vehicles")
def api_vehicles():
    return jsonify(STATE["vehicles"])


@app.route("/api/sample_edges")
def api_sample_edges():
    """Return a pair of nearby valid edge IDs and their coordinates for demo routing."""
    import random
    net = get_net()
    all_edges = [e for e in net.getEdges() if e.allows("passenger") and e.getLength() > 100]
    if len(all_edges) < 2:
        return jsonify([])

    seed = random.choice(all_edges)
    seed_shape = seed.getShape()
    if not seed_shape:
        seed = random.choice(all_edges)
        seed_shape = seed.getShape()
    sx, sy = seed_shape[len(seed_shape) // 2]

    nearby = net.getNeighboringEdges(sx, sy, 2000)
    nearby_edges = []
    for item in nearby:
        try:
            e = item[0]
        except (TypeError, IndexError):
            e = item
        if e.getID() != seed.getID() and e.allows("passenger") and e.getLength() > 50:
            nearby_edges.append(e)

    if not nearby_edges:
        nearby_edges = [e for e in all_edges if e.getID() != seed.getID()]

    dst = random.choice(nearby_edges)

    result = []
    for edge in [seed, dst]:
        shape = edge.getShape()
        if shape:
            mid = shape[len(shape) // 2]
            lon, lat = net.convertXY2LonLat(mid[0], mid[1])
            result.append({"edge_id": edge.getID(), "lat": round(lat, 6), "lon": round(lon, 6)})

    return jsonify(result)


# ── Socket.IO Events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit("status", {
        "model_loaded": STATE["model_loaded"],
        "sumo_running": STATE["sumo_running"],
        "model_meta":   STATE["model_meta"],
    })


@socketio.on("disconnect")
def on_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on("ping_sim")
def on_ping():
    emit("pong_sim", {"step": STATE["sim_step"], "vehicles": len(STATE["vehicles"])})


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model_state()

    # Pre-resolve station edges in background (non-blocking)
    threading.Thread(target=_preload_stations, daemon=True).start()

    host  = os.environ.get("FLASK_HOST", "0.0.0.0")
    port  = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    
    logger.info(f"Starting ResQFlow ML server on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)
