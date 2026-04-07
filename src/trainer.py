"""
ResQFlow ML - LightGBM Model Trainer
Trains a gradient-boosted model to predict edge travel time given real-time
traffic features. The trained model is used at routing time to pick the
ML-optimal path vs the standard shortest-path.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "traffic_samples.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "lgbm_router.pkl"
META_PATH  = MODEL_DIR / "model_meta.json"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "vehicle_count",
    "mean_speed",
    "occupancy",
    "waiting_time",
    "edge_length",
    "max_speed",
    "lane_count",
    "hour_of_day",
    "is_peak_hour",
    "speed_ratio",         # engineered
    "density",             # engineered: vehicles / edge_length
    "weather_factor",      # 0.0=clear, 0.5=rain, 1.0=fog
]
TARGET_COL = "travel_time"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model accuracy."""
    df = df.copy()
    df["speed_ratio"] = df["mean_speed"] / (df["max_speed"].clip(lower=0.1))
    df["density"]     = df["vehicle_count"] / (df["edge_length"].clip(lower=1.0))
    # Handle infinities / nans
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. "
            "Run data_generator.py first."
        )
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df):,} training samples from {DATA_PATH}")
    return df


def train(synthetic_fallback: bool = True):
    """
    Train the LightGBM travel-time prediction model.
    
    If not enough real SUMO data exists and synthetic_fallback=True,
    generates synthetic data to bootstrap the model.
    """
    try:
        df = load_data()
    except FileNotFoundError:
        if synthetic_fallback:
            logger.warning("No training data found — generating synthetic bootstrap data.")
            df = generate_synthetic_data(n=10000)
        else:
            raise
    
    min_samples = int(os.environ.get("MIN_TRAINING_SAMPLES", 500))
    if len(df) < min_samples:
        logger.warning(f"Only {len(df)} samples; augmenting with synthetic data.")
        synthetic = generate_synthetic_data(n=max(min_samples - len(df), 2000))
        df = pd.concat([df, synthetic], ignore_index=True)
    
    df = engineer_features(df)
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    # Remove extreme outliers (> 99th percentile travel time)
    q99 = y.quantile(0.99)
    mask = y <= q99
    X, y = X[mask], y[mask]
    logger.info(f"Training set size after filtering: {len(X):,}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    params = {
        "objective":        "regression",
        "metric":           "mae",
        "num_leaves":       63,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 20,
        "n_estimators":     500,
        "n_jobs":           -1,
        "random_state":     42,
        "verbose":          -1,
    }
    
    logger.info("Training LightGBM model...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )
    
    y_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, y_pred)
    r2   = r2_score(y_val, y_pred)
    logger.info(f"Validation MAE: {mae:.4f}s | R²: {r2:.4f}")
    
    # Save model + metadata
    joblib.dump(model, MODEL_PATH)
    
    import json
    meta = {
        "mae":          round(mae, 4),
        "r2":           round(r2, 4),
        "n_samples":    len(X),
        "features":     FEATURE_COLS,
        "trained_at":   pd.Timestamp.now().isoformat(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Model saved to {MODEL_PATH}")
    return model, meta


def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic traffic data for bootstrapping the model.
    Simulates Mysore city road network characteristics.
    """
    rng = np.random.default_rng(42)
    
    edge_lengths = rng.exponential(scale=150, size=n).clip(20, 2000)
    max_speeds   = rng.choice([8.33, 13.89, 16.67, 22.22], size=n)   # 30/50/60/80 km/h
    lane_counts  = rng.choice([1, 2, 3], size=n, p=[0.5, 0.35, 0.15])
    hour_of_day  = rng.integers(0, 24, size=n)
    is_peak      = ((hour_of_day >= 7) & (hour_of_day <= 9) |
                    (hour_of_day >= 17) & (hour_of_day <= 19)).astype(int)
    
    # Traffic density higher during peak
    base_density = rng.exponential(scale=0.02, size=n)
    vehicle_count = (base_density * edge_lengths * (1 + is_peak)).astype(int).clip(0, 50)
    
    # Speed degrades with congestion
    congestion_factor = 1 - (vehicle_count / vehicle_count.max()) * 0.6
    mean_speed = (max_speeds * congestion_factor * rng.uniform(0.6, 1.0, size=n)).clip(0.5, max_speeds)
    
    occupancy   = (vehicle_count / (lane_counts * edge_lengths / 8)).clip(0, 1)
    waiting_time = (vehicle_count * rng.exponential(scale=3, size=n)).clip(0, 300)

    # Synthetic weather factor: mostly clear, occasionally rain/fog
    weather_factor = rng.choice([0.0, 0.5, 1.0], size=n, p=[0.7, 0.2, 0.1])
    # Weather slows traffic — adjust travel time accordingly
    speed_reduction = 1 - weather_factor * 0.4
    travel_time  = edge_lengths / (mean_speed * speed_reduction).clip(0.5)

    return pd.DataFrame({
        "edge_id":         [f"synth_{i}" for i in range(n)],
        "step":            rng.integers(0, 3600, size=n),
        "vehicle_count":   vehicle_count,
        "mean_speed":      mean_speed.round(4),
        "occupancy":       occupancy.round(4),
        "waiting_time":    waiting_time.round(4),
        "edge_length":     edge_lengths.round(2),
        "max_speed":       max_speeds,
        "lane_count":      lane_counts,
        "hour_of_day":     hour_of_day,
        "is_peak_hour":    is_peak,
        "weather_factor":  weather_factor.round(1),
        "travel_time":     travel_time.round(4),
        "congestion_level": np.where(mean_speed / max_speeds >= 0.7, 0,
                             np.where(mean_speed / max_speeds >= 0.4, 1, 2)),
    })


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        logger.warning("No trained model found. Training with synthetic data...")
        model, _ = train(synthetic_fallback=True)
        return model
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    model, meta = train()
    print(f"\n✅ Training complete!")
    print(f"   MAE:  {meta['mae']} seconds")
    print(f"   R²:   {meta['r2']}")
    print(f"   N:    {meta['n_samples']:,} samples")
