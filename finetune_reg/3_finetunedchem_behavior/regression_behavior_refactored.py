import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import json
from utils.config import BASE_DIR, SEED
from utils.helpers import set_seeds, get_descriptors
from utils.regression import compute_correlation
from utils.model_config import MODELS, LAYERS_END
from utils.arg_parser import create_behavior_parser,create_regression_behavior_parser
from utils.data_loader import (
    load_behavior_embeddings,
    load_finetuned_model_embeddings,
    load_fold_cids,
)

# parser = create_behavior_parser("chem_exploration")


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    return [x]


def main():
    set_seeds(seed=SEED)
    
    parser = create_regression_behavior_parser()
    args = parser.parse_args()

    model_name     = args.model
    m              = MODELS.index(model_name)
    participant_id = args.participant_id
    n_components   = args.n_components
    n_fold         = args.n_fold
    out_dir        = args.out_dir
    z_score        = bool(args.z_score)
    ds             = args.ds
    run_id         = getattr(args, "run_id", os.environ.get("RUN_ID", "UNKNOWN"))
    embed_type     = "can"
    behavior_embeddings = args.behavior_embeddings or get_descriptors(ds)
    beh_val = (
        json.dumps(behavior_embeddings)   # '["intensity","pleasantness","sweet"]'
        .replace('"', "'")                # -> "['intensity','pleasantness','sweet']"
        if isinstance(behavior_embeddings, (list, tuple))
        else str(behavior_embeddings or "")
    )
    unfreeze_last_n: int = args.unfreeze_last_n

    # behavior columns
    embed_cols = get_descriptors(ds)

    # where to write metrics
    out_base = Path(BASE_DIR) / f"{out_dir}_behaviortuned_metrics_{run_id}"
    out_base.mkdir(parents=True, exist_ok=True)
    out_file = out_base / f"metrics_model-{model_name}_ds-{ds}_runid-{run_id}_unfreeze-{unfreeze_last_n}_behembd-{beh_val}.csv"

    # IMPORTANT: embeddings were saved with 0-based layer indices -> iterate 0..LAYERS_END[m] inclusive
    for layer in range(0, LAYERS_END[m] + 1):
        # Collect per-fold data for this layer
        train_embeddings_list, test_embeddings_list = [], []
        train_behaviors_list, test_behaviors_list   = [], []

        for i_fold in range(n_fold):
            print(f"[Fold {i_fold}/{n_fold}] layer={layer}")

            train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)

            # Load behavior (keeps CID order)
            train_behavior = load_behavior_embeddings(
                ds, train_cids, participant_id, embed_cols, group_by_cid=True
            )
            test_behavior = load_behavior_embeddings(
                ds, test_cids, participant_id, embed_cols, group_by_cid=True
            )

            # Load FINETUNED model embeddings FROM SAVED CSV (keeps CID order)
            train_emb = load_finetuned_model_embeddings(
                ds=ds,
                model_name=model_name,
                cids=train_cids,
                layer=layer,
                out_dir=out_dir,
                run_id=run_id,
                i_fold=i_fold,
                embed_type=embed_type,
                participant_id=participant_id,
                behavior_embeddings=beh_val,
                unfreeze_last_n=unfreeze_last_n
            )
            test_emb = load_finetuned_model_embeddings(
                ds=ds,
                model_name=model_name,
                cids=test_cids,
                layer=layer,
                out_dir=out_dir,
                run_id=run_id,
                i_fold=i_fold,
                embed_type=embed_type,
                participant_id=participant_id,
                 behavior_embeddings=beh_val,
                unfreeze_last_n=unfreeze_last_n

            )

            train_embeddings_list.append(train_emb)
            test_embeddings_list.append(test_emb)
            train_behaviors_list.append(train_behavior)
            test_behaviors_list.append(test_behavior)

        # Compute correlations aggregating across folds (per layer)
        metrics = compute_correlation(
            train_embeddings_list,
            train_behaviors_list,
            test_embeddings_list,
            test_behaviors_list,
            n_components=n_components,
        )

        # Annotate and save
        metrics = metrics.assign(
            model=model_name,
            ds=ds,
            participant_id=participant_id,
            layer=layer,
            n_fold=n_fold,
            n_components=n_components,
            z_score=z_score,
            target=get_descriptors(ds),
            behavior_embeddings = beh_val,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=run_id,
            unfreeze_last_n=unfreeze_last_n
        )

        write_header = not out_file.exists()
        metrics.to_csv(out_file, mode="a", index=False, header=write_header)
        print("****")


if __name__ == "__main__":
    main()
