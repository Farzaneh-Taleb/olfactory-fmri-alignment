import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
import csv
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = '/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment'
sys.path.insert(0, project_root)

from utils.config import BASE_DIR, SEED

def extract_hparams_from_filename(filename):
    """
    Extract hyperparameters from a filename like:
      mean_mse_{model_name}_{n_fold}_{epochs}_{subject}_{embedding}_{unfreeze}_{lr}_{bs}_{wd}.csv
    Returns a tuple of ALL seven hyperparameters.
    """
    stem = Path(filename).stem
    if not stem.startswith("mean_mse_"):
        return None
    stem = stem[len("mean_mse_"):]

    # split on '_' and peel off last 8 fields:
    parts = stem.split("_")
    if len(parts) < 9:
        print(f"Skipping malformed filename: {filename}")
        return None

    # destructure:
    *model_name_parts, n_fold, epochs, subject, embedding, unfreeze, lr, bs, wd = parts
    model_name = "_".join(model_name_parts)

    try:
        return (
            model_name,
            int(n_fold),
            int(epochs),
            int(subject),
            embedding,
            int(unfreeze),
            float(lr),
            int(bs),
            float(wd),
        )
    except ValueError as e:
        print(f"Error parsing values in {filename}: {e}")
        return None

def select_top5_hyperparams(csv_files):
    """
    Group files by the full 9‐tuple of hyperparameters and pick top 5 by mean best‐MSE.
    """
    grouped = defaultdict(list)
    for f in csv_files:
        h = extract_hparams_from_filename(f)
        if h:
            grouped[h].append(f)

    results = []
    for hparams, files in grouped.items():
        try:
            df_all = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
            # best MSE per fold, then average across folds:
            best_per_fold = df_all.groupby("fold")["mean_mse"].min()
            mean_best_mse = best_per_fold.mean()
            results.append((hparams, mean_best_mse, len(files)))
        except Exception as e:
            print(f"Failed reading {files}: {e}")

    if not results:
        print("No valid configurations found.")
        return []

    # sort ascending by mean_best_mse
    results.sort(key=lambda x: x[1])

    print("\nTop 5 hyperparameter configurations (including lr, bs, wd):")
    for rank, (h, score, count) in enumerate(results[:5], 1):
        (model, n_fold, epochs, subj, emb, unfreeze, lr, bs, wd) = h
        print(f"{rank:>2}. model={model}, folds={n_fold}, epochs={epochs}, subject={subj}, "
              f"emb=[{emb}], unfreeze={unfreeze}, lr={lr}, bs={bs}, wd={wd}  →  AvgBestMSE={score:.6f}  "
              f"({count} files)")
    return results[:5]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",         required=True)
    parser.add_argument("--subject",            required=True)
    parser.add_argument("--behavior_embedding", required=True)
    parser.add_argument("--unfreeze_last_n",    required=True)
    parser.add_argument("--metrics_dir",        required=True)
    parser.add_argument("--save_dir",           required=True)
    args = parser.parse_args()

    pattern = (
        f"{args.metrics_dir}/mean_mse_{args.model_name}_*_*_{args.subject}_"
        f"{args.behavior_embedding}_{args.unfreeze_last_n}_*_*_*.csv"
    )
    print(f"Searching for: {pattern}")
    csv_files = glob.glob(pattern)
    print(f"Found {len(csv_files)} files.")

    top5 = select_top5_hyperparams(csv_files)
    if not top5:
        exit(1)

    # write out a CSV with all 9 params + score + file-count
    out_path = (
        f"{args.save_dir}/top5_hparams_{args.model_name}_"
        f"{args.subject}_{args.behavior_embedding}_{args.unfreeze_last_n}.csv"
    )
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "Rank","Model","n_fold","epochs","subject",
            "embedding","unfreeze_last_n","lr","batch_size","weight_decay",
            "avg_best_mse","num_files"
        ])
        for rank, (h, score, count) in enumerate(top5, 1):
            writer.writerow([
                rank, *h, f"{score:.6f}", count
            ])

    print(f"\nTop 5 saved to: {out_path}")
