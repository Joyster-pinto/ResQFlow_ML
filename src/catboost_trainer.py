"""
ResQFlow ML — CatBoost Travel-Time Trainer
CatBoost gradient-boosted model that handles categorical road features natively.

Uses the same 12 traffic features as LightGBM, plus:
  - road_type_code  (0=residential, 1=local, 2=arterial, 3=highway)
    → derived from max_speed, treated as categorical by CatBoost

CatBoost natively handles categoricals without one-hot encoding, giving it
an edge when road type significantly influences travel patterns.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "traffic_samples.csv"
MODEL_DIR = BASE_DIR / "models"
CB_PATH   = MODEL_DIR / "catboost_router.cbm"
CB_META   = MODEL_DIR / "catboost_meta.json"
MODEL_DIR.mkdir(exist_ok=True)

# ── Optional CatBoost import ──────────────────────────────────────────────────
try:
    from catboost import CatBoostRegressor
    _CATBOOST = True
except ImportError:
    _CATBOOST = False
    logger.warning("catboost not installed. Run: pip install catboost")


def catboost_available() -> bool:
    return _CATBOOST


FEATURE_COLS_CB = [
    "vehicle_count",
    "mean_speed",
    "occupancy",
    "waiting_time",
    "edge_length",
    "max_speed",
    "lane_count",
    "hour_of_day",
    "is_peak_hour",
    "speed_ratio",
    "density",
    "weather_factor",
    "road_type_code",     # categorical — NEW
]
TARGET_COL      = "travel_time"
CATEGORICAL_COLS = ["road_type_code"]


def _road_type_code(max_speed_ms: float) -> int:
    kmh = max_speed_ms * 3.6
    if kmh <= 30:  return 0   # residential
    if kmh <= 50:  return 1   # local
    if kmh <= 70:  return 2   # arterial
    return 3                  # highway


def _add_road_type(df: pd.DataFrame) -> pd.DataFrame:
    """Derive road_type_code from max_speed column."""
    df = df.copy()
    if "road_type_code" not in df.columns:
        df["road_type_code"] = df["max_speed"].apply(_road_type_code).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speed_ratio"] = df["mean_speed"] / (df["max_speed"].clip(lower=0.1))
    df["density"]     = df["vehicle_count"] / (df["edge_length"].clip(lower=1.0))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = _add_road_type(df)
    return df


def train_catboost(df: pd.DataFrame = None):
    """
    Train CatBoost regressor on traffic data.
    If no data provided, generates synthetic bootstrap data.
    """
    if not catboost_available():
        raise RuntimeError("catboost not installed. pip install catboost")

    if df is None:
        # Import synthetic generator from trainer module
        sys.path.insert(0, str(Path(__file__).parent))
        from trainer import generate_synthetic_data
        logger.warning("No data provided — generating 10,000 synthetic samples.")
        df = generate_synthetic_data(10_000)

    df = engineer_features(df)

    X    = df[FEATURE_COLS_CB]
    y    = df[TARGET_COL]

    # Filter outliers
    q99 = y.quantile(0.99)
    X, y = X[y <= q99], y[y <= q99]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_feature_indices = [FEATURE_COLS_CB.index(c) for c in CATEGORICAL_COLS]

    logger.info(f"Training CatBoost on {len(X_train):,} samples…")
    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        eval_metric="MAE",
        cat_features=cat_feature_indices,
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2  = r2_score(y_val, y_pred)
    logger.info(f"CatBoost validation  MAE={mae:.4f}s  R²={r2:.4f}")

    model.save_model(str(CB_PATH))
    meta = {
        "mae": round(mae, 4), "r2": round(r2, 4),
        "n_samples": len(X), "features": FEATURE_COLS_CB,
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with open(CB_META, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"CatBoost saved → {CB_PATH}")
    return model, meta


def load_catboost():
    """Load saved CatBoost model, or train if not found."""
    if not catboost_available():
        logger.warning("CatBoost unavailable — skipping.")
        return None

    if CB_PATH.exists():
        logger.info("Loading saved CatBoost model…")
        model = CatBoostRegressor()
        model.load_model(str(CB_PATH))
        logger.info("CatBoost model loaded.")
        return model

    logger.warning("No CatBoost model found — training from scratch…")
    model, _ = train_catboost()
    return model


def predict_catboost(model, edge, live: dict) -> float:
    """
    Predict travel time for a single edge using CatBoost.
    Builds full feature vector including road_type_code.
    """
    if model is None:
        return None

    max_speed   = edge.getSpeed()
    edge_length = edge.getLength()
    lane_count  = edge.getLaneNumber()
    speed_ratio = live["mean_speed"] / max(max_speed, 0.1)
    density     = live["vehicle_count"] / max(edge_length, 1.0)
    road_type   = _road_type_code(max_speed)

    feats = [[
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
        live["weather_factor"],
        road_type,
    ]]

    try:
        pred = float(model.predict(feats)[0])
        return max(pred, 0.1)
    except Exception as e:
        logger.debug(f"CatBoost predict error: {e}")
        return None


if __name__ == "__main__":
    if not catboost_available():
        print("❌ CatBoost not installed. Run: pip install catboost")
        raise SystemExit(1)
    model, meta = train_catboost()
    print(f"\n✅ CatBoost training complete!")
    print(f"   MAE: {meta['mae']}s   R²: {meta['r2']}")
