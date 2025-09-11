import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from utils.config import BASE_DIR, SEED
from utils.helpers import *
from utils.regression import compute_correlation
from utils.model_config import MODELS, LAYERS_END, ROIS, P_VALUES
from utils.arg_parser import create_fmri_parser, parse_common_args
from utils.data_loader import load_model_embeddings, load_fold_cids,slice_fmri_by_cids
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pathlib import Path

parser = create_fmri_parser('chem_exploration')


def load_fmri_data(subject, roi, tr, z_score=False):
    """
    Returns:
        fmri: np.ndarray [n_cids x n_voxels]
        cids: np.ndarray[int] aligned with fmri rows
    """
    fmri, cids = read_fmri(BASE_DIR, subject, roi, tr)  # from utils.helpers
    cids = np.asarray(cids, dtype=np.int64)  # <- keep as integers

    if z_score:
        fmri = stats.zscore(fmri, axis=0, nan_policy='omit')
        fmri = np.nan_to_num(fmri, nan=0.0)

    return fmri, cids




def main():
    set_seeds(seed=SEED)
    args = parser.parse_args()
    args = parse_common_args(args)

    model_name = args.model
    m = MODELS.index(model_name)
    participant_id = args.participant_id
    roi = args.roi
    n_components = args.n_components
    n_fold = args.n_fold
    out_dir = args.out_dir
    tr = args.tr
    z_score = bool(args.z_score)
    ds = args.ds  # use dataset from args for consistency with your other script
    run_id = args.run_id
    # If TR=-1, choose the peak TR from P_VALUES for this ROI/subject
    i_roi = ROIS.index(roi)
    tr_orig = tr
    if tr == -1:
        tr = P_VALUES[i_roi][participant_id - 1]

    # Output path setup
    out_base = Path(BASE_DIR) / f"{out_dir}_fmri_metrics_{run_id}"
    out_base.mkdir(parents=True, exist_ok=True)
    out_file = out_base / f"metricsfmri_model-{model_name}_ds-{ds}_runid-{run_id}.csv"

    # Loop over layers
    for layer in range(1, LAYERS_END[m] + 1):
        print(f"Processing layer {layer}")
        train_embeddings_list, test_embeddings_list = [], []
        train_fmri_list,       test_fmri_list       = [], []


        # Load fMRI once per layer (same TR/ROI/subject)
        fmri_data, cids = load_fmri_data(participant_id, roi, tr, z_score=z_score)

        for i_fold in range(n_fold):
            print(f"Fold {i_fold}, Layer {layer}")

            # CIDs per fold
            train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)

            # fMRI slices
            fmri_train = slice_fmri_by_cids(fmri_data, cids, train_cids)
            fmri_test = slice_fmri_by_cids(fmri_data, cids, test_cids)

            # Embeddings (X)
            train_emb = load_model_embeddings(ds, model_name, train_cids, layer, embed_type='can')
            test_emb = load_model_embeddings(ds, model_name, test_cids, layer, embed_type='can')

            train_embeddings_list.append(train_emb)
            test_embeddings_list.append(test_emb)
            train_fmri_list.append(fmri_train)
            test_fmri_list.append(fmri_test)

            # Compute correlations: X=embeddings, Y=fMRI
        metrics = compute_correlation(
            train_embeddings_list, train_fmri_list, test_embeddings_list, test_fmri_list,
            n_components=n_components,z_score=z_score
        )
        # Attach metadata
        metrics = metrics.assign(
            model=model_name,
            ds=ds,
            participant_id=participant_id,
            roi=roi,
            layer=layer,
            n_fold=n_fold,
            i_fold=i_fold,
            n_components=n_components,
            z_score=z_score,
            tr=tr_orig,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=os.environ.get("RUN_ID", "UNKNOWN")
        )
        write_header = not out_file.exists()
        metrics.to_csv(out_file, mode="a", index=False, header=write_header)
        print("****")


if __name__ == "__main__":
    main()