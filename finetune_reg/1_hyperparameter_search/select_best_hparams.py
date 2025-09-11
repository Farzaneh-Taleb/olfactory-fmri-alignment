import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import csv
import sys
from utils.arg_parser import create_hparm_parser
from utils.config import BASE_DIR
from utils.helpers import get_descriptors
parser = create_hparm_parser()
def load_tracking_csv(metrics_dir: str, model_name: str, ds: str,run_id: str) -> Path:
    """
    New pipeline appends all runs of a (model, ds) into a single CSV:
      mse_tracking_model-{model}_ds-{ds}.csv
    """
    p = Path(metrics_dir) / f"mse_tracking_model-{model_name}_ds-{ds}_runid-{run_id}.csv"
    if not p.exists():
        print(f"[ERROR] Tracking file not found: {p}", file=sys.stderr)
        sys.exit(1)
    return p

def select_top5_from_tracking(df: pd.DataFrame,
                              subject: int,
                              unfreeze_last_n: int,
                              behavior_embeddings: str | None) -> list[tuple]:
    """
    From the consolidated tracking DataFrame, pick top-5 hyperparams by:
      1) For each (fold, hyperparams) take the minimum mean_mse over epochs.
      2) Average across folds -> AvgBestMSE.
    Hyperparams considered: (learning_rate, batch_size, weight_decay, num_train_epochs).
    We pre-filter by subject (participant_id) and unfreeze_last_n. If the column
    'behavior_embeddings' exists we also filter by exact match to behavior_embedding.
    Returns list of tuples: (hyper_tuple, avg_best_mse, n_folds)
    where hyper_tuple = (lr, bs, wd, epochs)
    """
    # Required columns check
    needed = {"fold", "mean_mse", "participant_id", "unfreeze_last_n",
              "learning_rate", "batch_size", "weight_decay", "num_train_epochs"}
    if not needed.issubset(df.columns):
        missing = sorted(list(needed - set(df.columns)))
        print(f"[ERROR] Tracking CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    beh_val = (
        json.dumps(behavior_embeddings)   # '["intensity","pleasantness","sweet"]'
        .replace('"', "'")                # -> "['intensity','pleasantness','sweet']"
        if isinstance(behavior_embeddings, (list, tuple))
        else str(behavior_embeddings or "")
    )
    q = (
    (df["participant_id"] == int(subject)) &
    ((df["unfreeze_last_n"] == unfreeze_last_n) | (df["unfreeze_last_n"].isna() & (unfreeze_last_n is None))) &
    (df["behavior_embeddings"] == beh_val)
) 

    print(df["unfreeze_last_n"].unique(),"uuu")
    print(unfreeze_last_n,"unfreeze_last_nnn")

    dff = df.loc[q].copy()
    if dff.empty:
        print("[ERROR] No rows after filtering by subject/unfreeze (and embedding if available).", file=sys.stderr)
        sys.exit(2)

    # Define hyperparam key
    hyper_cols = ["learning_rate", "batch_size", "weight_decay", "num_train_epochs"]
    dff["__hyper__"] = list(zip(*[dff[c] for c in hyper_cols]))

    # Step 1: per (hyper, fold) min over epochs
    per_fold_min = (
        dff.groupby(["__hyper__", "fold"], as_index=False)["mean_mse"]
        .min()
        .rename(columns={"mean_mse": "best_mse"})
    )

    # Step 2: average across folds
    agg = (
        per_fold_min.groupby("__hyper__", as_index=False)
        .agg(avg_best_mse=("best_mse", "mean"),
             n_folds=("best_mse", "size"))
    )

    # Sort ascending by avg_best_mse
    agg = agg.sort_values("avg_best_mse", ascending=True)

    # Convert to list of tuples
    results = [ (row["__hyper__"], float(row["avg_best_mse"]), int(row["n_folds"])) 
                for _, row in agg.iterrows() ]

    return results[:5]

if __name__ == "__main__":
    
    
    args = parser.parse_args()
    unfreeze_last_n=args.unfreeze_last_n
    if unfreeze_last_n in (None, "", "None"):
    # nothing to unfreeze
        print("No layers will be unfrozen.")
    else:
        unfreeze_last_n = int(unfreeze_last_n)
    metrics_dir =f'{BASE_DIR}/{args.out_dir}_finetune_metrics_{args.run_id}'
    # Load the consolidated tracking CSV
    tracking_path = load_tracking_csv(metrics_dir, args.model, args.ds,args.run_id)
    ds = args.ds
    behavior_embeddings = args.behavior_embeddings or get_descriptors(ds)
    print(f"Reading tracking CSV: {tracking_path}")
    df = pd.read_csv(tracking_path)
    df["unfreeze_last_n"] = df["unfreeze_last_n"].replace("", np.nan)

    # Select top-5
    top5 = select_top5_from_tracking(
        df, subject=args.participant_id,
        unfreeze_last_n=unfreeze_last_n,
        behavior_embeddings=behavior_embeddings
    )
    if not top5:
        print("[ERROR] No valid configurations found.")
        sys.exit(3)

    # Pretty print
    print("\nTop 5 hyperparameter configurations (lr, bs, wd, epochs):")
    for i, (hyper, score, nfolds) in enumerate(top5, 1):
        lr, bs, wd, epochs = hyper
        print(f"{i:>2}. lr={lr}, bs={bs}, wd={wd}, epochs={epochs}  →  AvgBestMSE={score:.6f}  (folds={nfolds})")

    # Save CSV
    out_name = (
        f"top5_hparams_{args.model}_{args.ds}_"
        f"subj-{args.participant_id}_emb-{behavior_embeddings}_unf-{unfreeze_last_n}_runid-{args.run_id}.csv"
    )
    out_path = Path(args.save_dir) /args.run_id/ out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow([
            "Rank","model","ds","participant_id","behavior_embeddings","unfreeze_last_n",
            "learning_rate","batch_size","weight_decay","num_train_epochs",
            "avg_best_mse","n_fold","run_id"
        ])
        for rank, (hyper, score, nfolds) in enumerate(top5, 1):
            lr, bs, wd, epochs = hyper
            w.writerow([
                rank, args.model, args.ds, args.participant_id,behavior_embeddings,
                unfreeze_last_n, lr, bs, wd, epochs, f"{score:.6f}", nfolds,args.run_id
            ])


    print(f"\nTop 5 saved to: {out_path}")
