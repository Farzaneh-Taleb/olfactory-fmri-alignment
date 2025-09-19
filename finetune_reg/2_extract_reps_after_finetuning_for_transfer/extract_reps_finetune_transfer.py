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
from utils.helpers import set_seeds, _load_text_for_cids,get_descriptors,extract_representations,get_latest_checkpoint,build_models_dir,build_embeds_dir,build_transfer_embeds_dir
from utils.data_loader import load_fold_cids
from utils.arg_parser import create_extract_rep_parser


def main():
    set_seeds(seed=SEED)
    parser = create_extract_rep_parser()
    args= parser.parse_args()

    model_name: str = args.model
    participants = [1,2,3]
    participant_source: int = args.participant_id
    participants_dest =  [s for s in participants if s != participant_source]


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
    embeds_dir = build_transfer_embeds_dir(out_dir_name, run_id)
    num_train_epochs = 40
    print(embeds_dir)
    (embeds_dir / "logs").mkdir(parents=True, exist_ok=True)
    per_model_ds_csv = embeds_dir / f"reps_{model_name}_ds-{ds}_runid-{run_id}_unfreeze-{unfreeze_last_n}_behaviorembeddings-{beh_val}_nfold_{n_fold}.csv"




    
    for participant_dest in participants_dest:
    
        for i_fold in range(n_fold):
        
            model_dir = models_dir / (
                f"model_{model_name}_{n_fold}_{participant_source}_"
                f"{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
            )
            ckpt = get_latest_checkpoint(model_dir)
            model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

            train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)
            cids = list(train_cids) + list(test_cids)
            extract_representations(
                cids=cids,
                participant_source=participant_source,
                input_type=input_type,
                out_csv=per_model_ds_csv,
                tokenizer=tokenizer,
                model=model,
                model_name=model_name,
                n_fold=n_fold,
                i_fold=i_fold,
                participant_dest=participant_dest,
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
