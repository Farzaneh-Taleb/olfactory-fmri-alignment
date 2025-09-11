from __future__ import annotations
import argparse
from pathlib import Path
import os, sys, glob
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
project_root = '/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment-NEW'
sys.path.insert(0, project_root)
from utils.config import BASE_DIR, SEED
from utils.model_config import INPUT_TYPES_CAN
from utils.helpers import set_seeds, _load_text_for_cids,get_descriptors,extract_representations,get_latest_checkpoint,build_models_dir,build_embeds_dir
from utils.data_loader import load_fold_cids
from utils.arg_parser import create_extract_rep_parser




def main():
    set_seeds(seed=SEED)
    parser = create_extract_rep_parser()
    args= parser.parse_args()

    model_name: str = args.model
    participant_id: int = args.participant_id
    n_fold: int = args.n_fold
    ds: str = args.ds

    behavior_embeddings = args.behavior_embeddings or get_descriptors(ds)
    beh_val = (
        json.dumps(behavior_embeddings)   # '["intensity","pleasantness","sweet"]'
        .replace('"', "'")                # -> "['intensity','pleasantness','sweet']"
        if isinstance(behavior_embeddings, (list, tuple))
        else str(behavior_embeddings or "")
    )
    unfreeze_last_n = args.unfreeze_last_n
    print(unfreeze_last_n,"unfreeze_last_n")
    out_dir_name: str = args.out_dir
    run_id: str = args.run_id
    embed_type = 'can'

    input_type: str = INPUT_TYPES_CAN.get(model_name)

    models_dir = build_models_dir(out_dir_name, run_id)
    embeds_dir = build_embeds_dir(out_dir_name, run_id)
    num_train_epochs = 40
    print(embeds_dir)
    (embeds_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Expect EXACT run_id in filename
    # sel_csv = (
    #     Path(BASE_DIR)
    #     / "best_hparam_selection_logs"
    #     / run_id
    #     / f"top5_hparams_{model_name}_{ds}_subj-{participant_id}_emb-{behavior_embeddings}_unf-{unfreeze_last_n}_runid-{run_id}.csv"
    # )

    # if not sel_csv.exists():
    #     # Optional: fallback to latest same spec (without run_id) if needed
    #     pattern = (
    #         Path(BASE_DIR) / "best_hparam_selection_logs" /
    #         f"top5_hparams_{model_name}_{ds}_subj-{participant_id}_emb-{behavior_embeddings}_unf-{unfreeze_last_n}_runid-*.csv"
    #     )
    #     matches = sorted(glob.glob(str(pattern)), key=os.path.getmtime)
    #     if not matches:
    #         raise FileNotFoundError(f"Top-5 hparam CSV not found: {sel_csv} nor any matching {pattern}")
    #     sel_csv = Path(matches[-1])
    #     print(f"[WARN] Exact run_id CSV not found; falling back to latest: {sel_csv}", flush=True)

    # top5 = pd.read_csv(sel_csv)
    # if top5.empty:
    #     raise RuntimeError(f"No rows in: {sel_csv}")

    # best = top5.sort_values("Rank").iloc[0]
    # lr = float(best["learning_rate"])
    # batch_size = int(best["batch_size"])
    # weight_decay = float(best["weight_decay"])
    # num_train_epochs = int(best["num_train_epochs"])

    # One accumulating CSV per (model_name, ds, run_id)
    per_model_ds_csv = embeds_dir / f"reps_{model_name}_ds-{ds}_runid-{run_id}_unfreeze-{unfreeze_last_n}_behavior_embeddings-{beh_val}.csv"

    for i_fold in range(n_fold):
        # print(
        #     f"[Extract] model={model_name} fold={i_fold}/{n_fold} "
        #     f"epochs={num_train_epochs} subj={participant_id} emb={behavior_embeddings} "
        #     f"unfreeze={unfreeze_last_n} ds={ds} lr={lr} bs={batch_size} wd={weight_decay}",
        #     flush=True,
        # )

        train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)
        cids = list(train_cids) + list(test_cids)
        model_dir = models_dir / (
            f"model_{model_name}_{n_fold}_{num_train_epochs}_{participant_id}_"
            f"{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
        )
        ckpt = get_latest_checkpoint(model_dir)
        model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

        extract_representations(
            cids=cids,
            participant_id=participant_id,
            input_type=input_type,
            out_csv=per_model_ds_csv,
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            n_fold=n_fold,
            i_fold=i_fold,
            subject=participant_id,
            behavior_embeddings=behavior_embeddings,
            unfreeze_last_n=unfreeze_last_n,
            ds=ds,
            token_index=0,
            embed_type=embed_type
        )
        print("extracted fold",i_fold)
    print(f"All done! Extracted representations saved to: {per_model_ds_csv}", flush=True)
if __name__ == "__main__":
    main()
