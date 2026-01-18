import os
import pandas as pd

BASE_DIR = "data/raw"

records = []

def process_folder(base_path, label):
    for root, dirs, files in os.walk(base_path):
        if "nodes.csv" in files:
            nodes_path = os.path.join(root, "nodes.csv")
            try:
                df = pd.read_csv(nodes_path)
                df["label"] = label
                df["graph_id"] = os.path.basename(root)
                records.append(df)
            except Exception as e:
                print(f"Skipping {nodes_path}: {e}")

# Label mapping
process_folder(os.path.join(BASE_DIR, "Non_Conspiracy_Graphs"), label=0)
process_folder(os.path.join(BASE_DIR, "Other_Graphs"), label=1)

master_df = pd.concat(records, ignore_index=True)

output_path = os.path.join(BASE_DIR, "covid_misinfo_raw.csv")
master_df.to_csv(output_path, index=False)

print("Saved master dataset to:", output_path)
print("Shape:", master_df.shape)
print("Columns:", master_df.columns.tolist())


# Save processed version (baseline clean copy)
processed_path = "data/processed/covid_misinfo_processed.csv"
master_df.to_csv(processed_path, index=False)
print("Saved processed dataset to:", processed_path)
