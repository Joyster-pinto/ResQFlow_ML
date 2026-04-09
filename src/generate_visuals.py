import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "traffic_samples.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "visuals"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def generate_model_summary_csv():
    # Load model metadata
    cb_meta_path = MODEL_DIR / "catboost_meta.json"
    lgbm_meta_path = MODEL_DIR / "model_meta.json"

    records = []
    
    if cb_meta_path.exists():
        with open(cb_meta_path, "r") as f:
            meta = json.load(f)
            records.append({
                "Model": "CatBoost",
                "MAE": meta.get("mae", 0),
                "R2": meta.get("r2", 0),
                "Samples": meta.get("n_samples", 0),
                "Number of Features": len(meta.get("features", []))
            })
            
    if lgbm_meta_path.exists():
        with open(lgbm_meta_path, "r") as f:
            meta = json.load(f)
            records.append({
                "Model": "LightGBM",
                "MAE": meta.get("mae", 0),
                "R2": meta.get("r2", 0),
                "Samples": meta.get("n_samples", 0),
                "Number of Features": len(meta.get("features", []))
            })

    if records:
        df = pd.DataFrame(records)
        out_path = OUTPUT_DIR / "model_comparison_tableau.csv"
        df.to_csv(out_path, index=False)
        print(f"Exported model summary for Tableau/PowerBI to {out_path}")
        
        # Also plot model comparison
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Model", y="MAE", palette="viridis")
        plt.title("Model Comparison by Mean Absolute Error (MAE)")
        plt.ylabel("Mean Absolute Error (s)")
        plt.savefig(OUTPUT_DIR / "model_mae_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

def generate_data_visuals():
    if not DATA_PATH.exists():
        print(f"Data not found at {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # 1. Travel Time Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["travel_time"], bins=50, kde=True, color="blue")
    plt.title("Distribution of Travel Times")
    plt.xlabel("Travel Time (s)")
    plt.xlim(0, df["travel_time"].quantile(0.99)) # Cut off extreme outliers
    plt.savefig(OUTPUT_DIR / "travel_time_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Matrix")
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Vehicle Count vs Travel Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sample(min(2000, len(df))), x="vehicle_count", y="travel_time", alpha=0.5)
    plt.title("Vehicle Count vs. Travel Time")
    plt.xlabel("Vehicle Count on Edge")
    plt.ylabel("Travel Time (s)")
    plt.savefig(OUTPUT_DIR / "vehicle_vs_travel_time.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Generate a cleaned sample CSV for Tableau
    sample_out = OUTPUT_DIR / "tableau_traffic_data_sample.csv"
    df.sample(min(5000, len(df))).to_csv(sample_out, index=False)
    print(f"Exported traffic data sample for Tableau/PowerBI to {sample_out}")

if __name__ == "__main__":
    generate_model_summary_csv()
    generate_data_visuals()
    print("Visualizations generated successfully.")
