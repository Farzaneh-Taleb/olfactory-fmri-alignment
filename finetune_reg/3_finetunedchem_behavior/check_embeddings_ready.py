#!/usr/bin/env python
import os
import sys
import argparse

# --- Make 'utils' importable like in your other scripts ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from utils.model_config import MODELS, LAYERS_END
from utils.data_loader import load_fold_cids, load_finetuned_model_embeddings


def parse_int_or_none(x: str | None):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"none", "null", "nan"}:
        return None
    return int(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--participant_id", required=True, type=int)
    p.add_argument("--model", required=True)
    p.add_argument("--n_fold", required=True, type=int)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ds", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--unfreeze_last_n", required=False)  # may be "None"
    p.add_argument("--behavior_embeddings", required=False, default="")
    args = p.parse_args()

    unfreeze_last_n = parse_int_or_none(args.unfreeze_last_n)
    behavior_val = args.behavior_embeddings or ""  # pass through as saved

    # model index and layer range
    try:
        m = MODELS.index(args.model)
    except ValueError:
        print(f"[CHECK] Model not found in MODELS: {args.model}", flush=True)
        return 2
    last_layer = LAYERS_END[m]

    missing = []
    for i_fold in range(args.n_fold):
        train_cids, test_cids = load_fold_cids(args.n_fold, i_fold, args.ds)
        # Probe representative layers (0 and last)
        for layer in sorted({0, last_layer}):
            for cids, split in [(train_cids, "train"), (test_cids, "test")]:
                try:
                    _ = load_finetuned_model_embeddings(
                        ds=args.ds,
                        model_name=args.model,
                        cids=cids,
                        layer=layer,
                        out_dir=args.out_dir,
                        run_id=args.run_id,
                        i_fold=i_fold,
                        embed_type="can",
                        participant_id=args.participant_id,
                        behavior_embeddings=behavior_val,
                        unfreeze_last_n=unfreeze_last_n,
                    )
                except FileNotFoundError:
                    missing.append(f"fold={i_fold} layer={layer} split={split}")
                except Exception as e:
                    missing.append(
                        f"fold={i_fold} layer={layer} split={split} err={type(e).__name__}"
                    )

    if missing:
        print("[CHECK] Missing embeddings:", *missing, sep="\n  ")
        return 2

    print("[CHECK] Embeddings found for sentinel files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
