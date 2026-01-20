import os
import re
import pandas as pd

TRUE_PATH = "data_text/raw/True.csv"
FAKE_PATH = "data_text/raw/Fake.csv"
OUT_PATH = "data_text/processed/isot_text_processed.csv"

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    if not os.path.exists(TRUE_PATH):
        raise FileNotFoundError(f"Missing file: {TRUE_PATH}")
    if not os.path.exists(FAKE_PATH):
        raise FileNotFoundError(f"Missing file: {FAKE_PATH}")

    true_df = pd.read_csv(TRUE_PATH)
    fake_df = pd.read_csv(FAKE_PATH)

    true_df["label"] = 0  # real
    fake_df["label"] = 1  # fake

    df = pd.concat([true_df, fake_df], ignore_index=True)

    if "title" not in df.columns or "text" not in df.columns:
        raise ValueError(f"Expected columns not found. Columns: {df.columns.tolist()}")

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["full_text"] = (df["title"] + ". " + df["text"]).str.strip()

    df = df[df["full_text"].str.len() > 0].copy()
    df["clean_text"] = df["full_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    out = df[["clean_text", "label"]].copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved processed text dataset to: {OUT_PATH}")
    print("Shape:", out.shape)
    print("Label counts:\n", out["label"].value_counts())

if __name__ == "__main__":
    main()
