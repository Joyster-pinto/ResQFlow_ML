"""
ResQFlow ML — Master Entry Point
Run this file to start the complete system:
  1. Trains the ML model (if not already trained)
  2. Starts the Flask + Socket.IO web server
  
Usage:
    python run.py                  # Start server (trains if needed)
    python run.py --train          # Force re-train before starting
    python run.py --collect-data   # Run SUMO data collection only
    python run.py --port 5000      # Custom port
"""

import sys
import os
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "data" / "resqflow.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)

# Ensure data dir exists
(BASE_DIR / "data").mkdir(exist_ok=True)
(BASE_DIR / "models").mkdir(exist_ok=True)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          ResQFlow ML — AI Traffic Routing System         ║
║     SUMO + TraCI + LightGBM + Flask + Socket.IO         ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="ResQFlow ML Server")
    parser.add_argument("--train",        action="store_true", help="Force model re-training")
    parser.add_argument("--collect-data", action="store_true", help="Run SUMO data collection only")
    parser.add_argument("--port",         type=int, default=5000, help="Web server port")
    parser.add_argument("--host",         default="0.0.0.0",      help="Web server host")
    parser.add_argument("--steps",        type=int, default=1800,  help="Data collection steps")
    args = parser.parse_args()

    # ── Data Collection Mode ──────────────────────────────────────────────
    if args.collect_data:
        logger.info("Starting SUMO data collection...")
        from data_generator import run_data_collection
        n = run_data_collection(steps=args.steps)
        logger.info(f"Collected {n} samples. Now run: python run.py --train")
        return

    # ── Training ──────────────────────────────────────────────────────────
    model_path = BASE_DIR / "models" / "lgbm_router.pkl"

    if args.train or not model_path.exists():
        logger.info("Training LightGBM model...")
        from trainer import train
        model, meta = train(synthetic_fallback=True)
        logger.info(f"Model trained. MAE={meta['mae']}s, R²={meta['r2']}")
    else:
        logger.info("Using existing model (pass --train to retrain).")

    # ── Server ────────────────────────────────────────────────────────────
    logger.info(f"Starting Flask server on http://{args.host}:{args.port}")
    logger.info("Open your browser at: http://localhost:%d", args.port)

    # Set env vars for app.py
    os.environ["FLASK_HOST"] = args.host
    os.environ["FLASK_PORT"] = str(args.port)

    from app import app, socketio, load_model_state
    load_model_state()
    socketio.run(app, host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
