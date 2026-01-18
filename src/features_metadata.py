import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/covid_misinfo_raw.csv"
OUT_PATH = "data/processed/covid_misinfo_metadata_features.csv"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleanup
    df = df.copy()

    # Ensure numeric types
    for col in ["time", "friends", "followers", "label"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "friends", "followers", "label"])

    # ---------- Feature engineering ----------
    # Log transforms (handle zeros safely)
    df["log_followers"] = np.log1p(df["followers"])
    df["log_friends"] = np.log1p(df["friends"])

    # Ratio features (avoid divide-by-zero)
    df["followers_per_friend"] = df["followers"] / (df["friends"] + 1.0)
    df["friends_per_follower"] = df["friends"] / (df["followers"] + 1.0)

    # Interaction feature
    df["followers_x_friends"] = df["followers"] * df["friends"]

    # Time feature (simple + safe):
    # "time" here is numeric; we don't assume it's a real timestamp.
    # We use a coarse binning so the model can learn "early vs late" patterns.
    df["time_bin"] = pd.qcut(df["time"], q=10, duplicates="drop").astype(str)

    # One-hot encode the time bins
    df = pd.get_dummies(df, columns=["time_bin"], drop_first=True)

    # Keep only columns needed for modeling + id/graph_id for traceability
    keep_cols = ["id", "graph_id", "label", "time", "friends", "followers",
                 "log_followers", "log_friends",
                 "followers_per_friend", "friends_per_follower",
                 "followers_x_friends"]

    # Add the time_bin dummy columns
    keep_cols += [c for c in df.columns if c.startswith("time_bin_")]

    df = df[keep_cols]

    return df

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    needed = {"id", "time", "friends", "followers", "label", "graph_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw dataset: {missing}")

    feat_df = add_features(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    feat_df.to_csv(OUT_PATH, index=False)

    print(f"Saved metadata features to: {OUT_PATH}")
    print("Shape:", feat_df.shape)
    print("Columns:", feat_df.columns.tolist())

if __name__ == "__main__":
    main()
