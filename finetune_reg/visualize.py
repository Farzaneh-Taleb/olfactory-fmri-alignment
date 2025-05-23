import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
base_dir='/proj/rep-learning-robotics/users/x_farzt'
# import sys
# parent_dir = f'{base_dir}/MoLFormer_fMRI'
# sys.path.append(parent_dir)

# === Config ===
data_dir = f"{base_dir}/finetuned_reg_metrics"  # change this
pattern = os.path.join(data_dir, "mean_mse_*.csv")

# === Helper: extract metadata from filename ===
def parse_filename(filename):
    base = os.path.basename(filename)
    match = re.match(r"mean_mse_(.*?)_(\d+)_(\d+)_(\d+)_(.*?)_(\d+)\.csv", base)
    if not match:
        return None
    model_name, n_fold, num_epochs, subject, behavior_embedding, unfreeze_last_n = match.groups()
    return {
        "file": filename,
        "model_name": model_name,
        "n_fold": int(n_fold),
        "subject": int(subject),
        "behavior_embedding": behavior_embedding,
        "unfreeze_last_n": int(unfreeze_last_n),
    }

# === Aggregate and group ===
all_dfs = []
for file in glob.glob(pattern):
    meta = parse_filename(file)
    if not meta:
        continue
    df = pd.read_csv(file)
    df["epoch"] = df["epoch"].astype(int)
    for key, val in meta.items():
        if key != "file":
            df[key] = val
    all_dfs.append(df)

all_data = pd.concat(all_dfs, ignore_index=True)

# === Plot per group ===
group_keys = ["model_name", "subject", "behavior_embedding", "n_fold"]
grouped = all_data.groupby(group_keys)

for key, group in grouped:
    avg_mse = group.groupby("epoch")["mean_mse"].mean()
    plt.figure()
    plt.plot(avg_mse.index, avg_mse.values, marker='o')
    plt.title(f"MSE over Epochs\nmodel={key[0]}, subject={key[1]}, embedding={key[2]}, folds={key[3]}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean MSE (avg over folds)")
    plt.grid(True)
    plt.tight_layout()
    # Optional: save plot
    # fname = f"mse_plot_{key[0]}_sub{key[1]}_emb{key[2]}_folds{key[3]}.png"
    # plt.savefig(fname)
    plt.show()
