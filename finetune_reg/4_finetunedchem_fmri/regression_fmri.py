import sys
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# --- Project imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from utils.config import BASE_DIR, SEED
from utils.helpers import set_seeds, read_fmri,get_descriptors  # read_fmri(BASE_DIR, subj, roi, tr) -> (fmri, cids)
from utils.regression import compute_correlation
from utils.model_config import MODELS, LAYERS_END, ROIS, P_VALUES
from utils.arg_parser import create_fmri_parser, parse_common_args
from utils.data_loader import (
    load_finetuned_model_embeddings,
    load_fold_cids,
  # if you keep this in helpers instead, import from there
    slice_fmri_by_cids
)

# # ---------- helpers ----------
# def _as_list(x):
#     if x is None:
#         return []
#     if isinstance(x, (list, tuple)):
#         return list(x)
#     if isinstance(x, str):
#         return [s.strip() for s in x.split(",") if s.strip()]
#     return [x]

def load_fmri_data(subject: int, roi: str, tr: int, z_score: bool = False):
    """
    Returns:
        fmri: np.ndarray [n_cids x n_voxels]
        cids: np.ndarray[int] aligned with fmri rows (int64)
    """
    fmri, cids = read_fmri(BASE_DIR, subject, roi, tr)
    cids = np.asarray(cids, dtype=np.int64)
    if z_score:
        fmri = stats.zscore(fmri, axis=0, nan_policy="omit")
        fmri = np.nan_to_num(fmri, nan=0.0)
    return fmri, cids

# ---------- main ----------
def main():
    set_seeds(seed=SEED)

    parser = create_fmri_parser("chem_exploration")
    args = parser.parse_args()
    args = parse_common_args(args)

    model_name_path = args.model
    model_path = model_name_path.split('/')[0]
    model_name = model_name_path.split('/')[1]
    m              = MODELS.index(model_name)
    participant_id = args.participant_id
    roi            = args.roi
    n_components   = args.n_components
    n_fold         = args.n_fold
    out_dir        = args.out_dir
    tr             = args.tr
    z_score        = bool(args.z_score)
    ds             = args.ds

    run_id = getattr(args, "run_id", os.environ.get("RUN_ID", "UNKNOWN"))
    unfreeze_last_n = getattr(args, "unfreeze_last_n", None)

    # behavior_embeddings is used in the **file pattern** for finetuned embeddings
    behavior_embeddings = args.behavior_embeddings or get_descriptors(ds)
    # Serialize to match filenames used when saving finetuned embeddings
    beh_val = (
        json.dumps(behavior_embeddings)   # '["intensity","pleasantness","sweet"]'
        .replace('"', "'")                # -> "['intensity','pleasantness','sweet']"
        if isinstance(behavior_embeddings, (list, tuple))
        else str(behavior_embeddings or "")
    )

    # TR handling: if TR=-1, use peak TR from P_VALUES for this ROI/subject
    i_roi = ROIS.index(roi)
    tr_orig = tr
    if tr == -1:
        tr = P_VALUES[i_roi][participant_id - 1]

    # Output
    out_base = Path(BASE_DIR) / f"{out_dir}_fmrifinetuned_metrics_{run_id}"
    out_base.mkdir(parents=True, exist_ok=True)
    out_file = out_base /  f"metrics_model-{model_name}_ds-{ds}_unfreeze-{unfreeze_last_n}_behembd-{beh_val}.csv"

    # Iterate **0..LAYERS_END[m] inclusive** (finetuned embeddings were saved 0-based)
    for layer in range(1, LAYERS_END[m] + 1):
        print(f"[Layer {layer}] ROI={roi}, subj={participant_id}, TR={tr} (arg TR={tr_orig})")

        # Load fMRI once per layer (same TR/ROI/subject); then slice per fold CIDs
        fmri_data, all_cids = load_fmri_data(participant_id, roi, tr, z_score=z_score)

        train_embeddings_list, test_embeddings_list = [], []
        train_fmri_list,       test_fmri_list       = [], []

        for i_fold in range(n_fold):
            print(f"  └─ Fold {i_fold}/{n_fold-1}")

            train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)

            # fMRI slices (keeps CID order by masking)
            fmri_train = slice_fmri_by_cids(fmri_data, all_cids, train_cids)
            fmri_test  = slice_fmri_by_cids(fmri_data, all_cids, test_cids)

            # Load FINETUNED embeddings (CSV) exactly like your behavior script
            train_emb = load_finetuned_model_embeddings(
                ds=ds,
                model_name=model_name,
                cids=train_cids,
                layer=layer,
                out_dir=out_dir,
                run_id=run_id,
                i_fold=i_fold,
                embed_type="can",
                participant_id=participant_id,
                behavior_embeddings=beh_val,
                unfreeze_last_n=unfreeze_last_n,
                n_fold=n_fold
            )
            test_emb = load_finetuned_model_embeddings(
                ds=ds,
                model_name=model_name,
                cids=test_cids,
                layer=layer,
                out_dir=out_dir,
                run_id=run_id,
                i_fold=i_fold,
                embed_type="can",
                participant_id=participant_id,
                behavior_embeddings=beh_val,
                unfreeze_last_n=unfreeze_last_n,
                n_fold=n_fold,
                
                
            )

            train_embeddings_list.append(train_emb)
            test_embeddings_list.append(test_emb)
            train_fmri_list.append(fmri_train)
            test_fmri_list.append(fmri_test)

        # Compute correlations aggregating across folds (per layer)
        metrics = compute_correlation(
            train_embeddings_list,  # X_train per fold
            train_fmri_list,        # Y_train per fold (fMRI)
            test_embeddings_list,   # X_test per fold
            test_fmri_list,         # Y_test per fold (fMRI)
            n_components=n_components,
            z_score=z_score
        )
        # Suppose test_fmri_list is a list of (TRs, V) arrays
        Y = np.vstack(test_fmri_list)   # shape = (N_total_TRs, N_voxels)
        targets = np.arange(Y.shape[1])  # 0 .. V-1
        # Annotate and append
        metrics = metrics.assign(
            model=model_name,
            ds=ds,
            participant_id=participant_id,
            roi=roi,
            layer=layer,
            n_fold=n_fold,
            n_components=n_components,
            z_score=z_score,
            target =  targets,
            # behavior_embeddings=get_descriptors(ds),
            behavior_embeddings = beh_val,
            unfreeze_last_n=unfreeze_last_n,
            tr=tr_orig,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=run_id,
        )

        write_header = not out_file.exists()
        metrics.to_csv(out_file, mode="a", index=False, header=write_header)
        print("  ✓ saved")

    print(f"Done. Appended metrics to: {out_file}")

if __name__ == "__main__":
    main()
