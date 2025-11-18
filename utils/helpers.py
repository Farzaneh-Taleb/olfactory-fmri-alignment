import pandas as pd
import numpy as np
import ast
import nibabel as nib
import scipy.io as sio
import os
import random
import torch
import pubchempy as pcp
from scipy.io import loadmat
import selfies as sf
import sys
import os, glob, re
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset
from .data_loader import load_behavior_embeddings, slice_fmri_by_cids, load_fmri_data
from datasets import Dataset as HFDataset
from .config import BASE_DIR
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .config import SEED
from scipy import stats

mid_dir='data'
from transformers import (
    AutoTokenizer,
    AutoModel
)
from pathlib import Path
def set_seeds(seed=SEED):
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If using GPUs, set seed for CUDA operations
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def create_nc_mask(BASE_DIR,subject,roi,threshold=0.3):

    #loading noise_ceiling
    noise_ceiling = pd.read_csv(f'{BASE_DIR}/fmri/noise_ceiling_session/noise_ceiling_session_{subject}_{roi}.csv')
    noise_ceiling.rename(columns={'fmri':'nc'}, inplace=True)
    noise_ceiling['nc_corrected'] = np.sqrt((2/(1+np.sqrt(1/noise_ceiling['nc']**2))))
    noise_ceiling['nc_corrected'] = np.where(np.isinf(noise_ceiling['nc_corrected']), 0, noise_ceiling['nc_corrected'])
    noise_ceiling['nc_corrected'] = np.nan_to_num(noise_ceiling['nc_corrected'])

    noise_ceiling_avg = noise_ceiling.groupby(['voxel','subject','roi']).mean()

    # noise_ceiling_avg= pd.merge(df,noise_ceiling_avg,on=['subject','roi','voxel'])
    noise_ceiling_avg.sort_values(by=['voxel','subject','roi'], inplace=True)
    noise_ceiling_avg['nc_selected']= noise_ceiling_avg['nc_corrected']>=threshold
    mask_1d= noise_ceiling_avg['nc_selected'].values.tolist()

    mask_1d=np.array(mask_1d)
    # print(mask_1d.shape,"nc shape")
    _,mask_3d = to3d(BASE_DIR,mask_1d,subject,roi)
    return  mask_3d



def mask_fmri(BASE_DIR,fmri,nc_mask_array,subject,roi,nc_mask,func_mask):
    path_ROI = f'{BASE_DIR}/fmri/supportings/S{subject}/'
    # Load the NIfTI mask file
    #load the whole brain mask
    maskfile = f'rw{roi}.nii'
    mask_img = nib.load(path_ROI + maskfile)
    mask_data = mask_img.get_fdata()>0.1


    #load functional mask
    maskfile_func = 'ARC3_fanatgw3_pos.nii'
    mask_img_func = nib.load(path_ROI + maskfile_func)
    mask_img_func=mask_img_func.get_fdata()>0

    #load anatomical mask
    maskfile_anat = 'ARC3_anatgw.nii'
    mask_img_anat = nib.load(path_ROI + maskfile_anat)
    mask_img_anat=mask_img_anat.get_fdata()>0.1

    #count the number of true values in the mask
    print(np.sum(mask_data),np.sum(nc_mask_array),np.sum(mask_img_func),np.sum(mask_img_anat))



    #intersection of all masks
    if func_mask and nc_mask:
        result_reduce = np.logical_and.reduce([mask_data,mask_img_anat, mask_img_func, nc_mask_array])
    elif func_mask:
        result_reduce = np.logical_and.reduce([mask_data,mask_img_anat, mask_img_func])
    elif nc_mask:
        result_reduce = np.logical_and.reduce([mask_data, mask_img_anat, nc_mask_array]) 
    else:
        result_reduce = np.logical_and.reduce([mask_data, mask_img_anat])
    _,mask_1d = from3d(BASE_DIR,result_reduce,subject,roi)


    #read fmri data
    # masked_fmri = fmri[:,mask_1d]

       # Identify voxels that contain NaN values in any trial
    # nan_mask = np.isnan(fmri,axis=0)
     # True if voxel has NaN in any CID

    # Compute mean for each voxel across all CIDs
    mean_values_voxel = fmri.mean(axis=0)
    zero_mean_voxels = abs(mean_values_voxel) == 0  # Near-zero mean voxels

    # Combine NaN and zero-mean voxels
    # print(len(nan_mask),len(zero_mean_voxels),len(mask_1d))
    voxels_to_remove = zero_mean_voxels | ~mask_1d  # Logical OR to combine both conditions
    masked_fmri = fmri[:, ~voxels_to_remove]  # Drop the columns (voxels) to remove

    #index of ~voxels_to_remove
    indices = np.where(~voxels_to_remove)[0]
    indices = pd.DataFrame(indices)





    # print(mask_1d)
    # print(fmri.shape,np.sum(voxels_to_remove),masked_fmri.shape)
    return indices, masked_fmri


def df_to_cid_voxel_array(df,tr):
    # Load the data
   
    values=f'TR_{tr}'
    # Create fMRI array (CIDs as rows, voxels as columns)
    fmri_array = df.pivot(index='CID', columns='voxel', values=values)
    fmri_array_sorted = fmri_array.sort_index().sort_index(axis=1)
    fmri_array_cleaned = fmri_array_sorted.to_numpy()

    fmri_array_cleaned = np.nan_to_num(fmri_array_cleaned)



    return fmri_array_cleaned

def read_fmri_avgSingleTrial_peak(BASE_DIR, subject_id, selected_roi):
    # Load the data
    parent_input_sagar_original = f'{BASE_DIR}/fmri/average_of_singletrial/fmri_{subject_id}_{selected_roi}.csv'
    df = pd.read_csv(parent_input_sagar_original)
    # if dro_voxels is not None:
    #     #drop voxels that are not in the mask
    #     df = df[~df['voxel'].isin(dro_voxels)]
    # df = df.sort_values(by=['CID','voxel'])

    # Create fMRI array (CIDs as rows, voxels as columns)
    fmri_array = df.pivot(index='CID', columns='voxel', values='fmri')




    # Convert the cleaned DataFrame to a numpy array
    fmri_array_cleaned = fmri_array.to_numpy()

    #replace nan with 0
    fmri_array_cleaned = np.nan_to_num(fmri_array_cleaned)



    return fmri_array_cleaned


def read_fmri_avgSingleTrial_dropna(BASE_DIR, subject_id, selected_roi,dro_voxels=None):
    # Load the data
    parent_input_sagar_original = f'{BASE_DIR}/fmri/average_of_singletrial/fmri_{subject_id}_{selected_roi}.csv'
    df = pd.read_csv(parent_input_sagar_original)


    # Create fMRI array (CIDs as rows, voxels as columns)
    fmri_array = df.pivot(index='CID', columns='voxel', values='fmri')




    # Convert the cleaned DataFrame to a numpy array
    fmri_array_cleaned = fmri_array.to_numpy()

    #replace nan with 0
    fmri_array_cleaned = np.nan_to_num(fmri_array_cleaned)



    return fmri_array_cleaned



def prepare_dataset(ds):
    if 'y' in ds.columns:
        ds['y'] = ds['y'].apply(ast.literal_eval)
    ds['embeddings'] = ds['embeddings'].apply(ast.literal_eval)
    return ds


def read_pom(BASE_DIR, CIDs):
    embedding_open_pom = '/alignment_olfaction_datasets/curated_datasets/embeddings/pom/sagar_pom_embeddings_Apr17.csv'
    ds_pom = pd.read_csv(BASE_DIR + embedding_open_pom)
    ds_pom = prepare_dataset(ds_pom)

    ds_pom = ds_pom.sort_values(by='CID')
    ds_pom = ds_pom[['CID', 'embeddings', 'y', 'subject']]
    # ds_pom = ds_pom[ds_pom['subject'] == subject]
    ds_pom = ds_pom[ds_pom['CID'].isin(CIDs)]

    #filter based on unique CIDs
    ds_pom = ds_pom.drop_duplicates(subset=['CID'])

    pom_embeddings = ds_pom['embeddings']
    pom_embeddings = np.array([np.array(x) for x in pom_embeddings])

    return pom_embeddings


def npy2nii( data_array,roi, subject,dir='' , title='', save=True):
    mask_img, newVOL = to3d(data_array, roi, subject)
    new_img = nib.Nifti1Image(newVOL, affine=mask_img.affine)

    # Save the new NIfTI image
    # output_file_path = 'path/to/output_file.nii'  # Output NIfTI file path
    # nib.save(new_img, f"{BASE_DIR}/fmri/metrics_avgsingletrial/subj_{subject}_{roi}_{model}_{layer}_{title}_metrics.nii")
    if save:
        nib.save(new_img, f"{dir}/{roi}+{subject}_{title}.nii")

    return new_img

def from3d(BASE_DIR,newVOL, subject, roi):
    # path_ROI = f'../../../../T5 EVO/fmri/supportings/S{subject}/'
    path_ROI = f'{BASE_DIR}/fmri/supportings/S{subject}/'
    maskfile = f'rw{roi}.nii'
    # Load the NIfTI mask file
    mask_img = nib.load(path_ROI + maskfile)
    mask_data = mask_img.get_fdata()
    mask_data[np.isnan(mask_data)] = 0
    # Convert to a boolean array (logical)
    mask_data = mask_data.astype(bool)
    voxel_inds = np.flatnonzero(mask_data)
    # Now let's reverse the process and extract the values from newVOL
    data_array_reversed = np.zeros(len(voxel_inds),dtype=bool)

    for idx, voxel_idx in enumerate(voxel_inds):
        xx, yy, zz = np.unravel_index(voxel_idx, newVOL.shape)
        data_array_reversed[idx] = newVOL[xx, yy, zz]


    return mask_img, data_array_reversed

def to3d(BASE_DIR,data_array, subject,roi):
    path_ROI = f'{BASE_DIR}/fmri/supportings/S{subject}/'
    maskfile = f'rw{roi}.nii'
    # Load the NIfTI mask file
    mask_img = nib.load(path_ROI + maskfile)
    mask_data = mask_img.get_fdata()
    mask_data[np.isnan(mask_data)] = 0
    # Convert to a boolean array (logical)
    mask_data = mask_data.astype(bool)
    voxel_inds = np.flatnonzero(mask_data)
    # Initialize new volume with zeros (same shape as mask)
    print(data_array.shape, voxel_inds.shape, subject, roi)
    if data_array.shape[0] != voxel_inds.shape[0]:
        raise ValueError("Data array and mask must have the same shape.", data_array.shape, voxel_inds.shape, subject,
                         roi)
    newVOL = np.zeros(mask_data.shape)
    print(newVOL.shape, data_array.shape)
    # Iterate over voxel indices
    for ii in range(len(voxel_inds)):
        # Convert linear index to 3D subscripts
        xx, yy, zz = np.unravel_index(voxel_inds[ii], mask_data.shape)
        # print(data_array[ii][0][0])
        # Assign data value to new volume
        newVOL[xx, yy, zz] = data_array[ii]
    return mask_img, newVOL


def f_test(X, y, estimator):
    # Predicted values
    y_pred = estimator.predict(X)

    # Residual Sum of Squares (RSS)


    # Total Sum of Squares (SST)
    y_mean = np.mean(y, axis=0)
    # SST = np.sum((y - y_mean) ** 2)
    SSE = np.sum((y - y_pred) ** 2)
    SSM = np.sum((y_pred - y_mean) ** 2)
    SST = np.sum((y - y_mean) ** 2)

    # Regression Sum of Squares (SSR)
    # SSR = SST - SSE

    dfm = X.shape[1] - 1  # Degrees of freedom for the model
    dfe = X.shape[0] - X.shape[1]  # Degrees of freedom for the error
    dft = X.shape[0] - 1  # Total degrees of freedom

    msm = SSM / dfm  # Mean Squares for the model
    mse = SSE / dfe  # Mean Squares for the error
    mst = SST / dft  # Mean Squares for the total


    # Degrees of freedom
    # n = len(y)  # Number of samples
    # k = X.shape[1]  # Number of predictors (features)

    # F-statistic
    F =msm/mse

    # P-value
    from scipy.stats import f
    p_value = 1 - f.cdf(F, dfm, dfe)

    return F, p_value

def find_overlap(BASE_DIR, ds, subject_source):
    subjects = [1, 2, 3]
    subjects.remove(subject_source)
    smiles_df_1 = pd.read_csv(f"{BASE_DIR}/embeddings{ds}/CIDs_smiles_selfies_{subject_source}{ds}.csv")
    # Get the CIDs from the source DataFrame
    cids_1 = smiles_df_1['CIDs']
    cids_rest = []

    # Collect CIDs from the remaining subjects
    for subject in subjects:
        smiles_df_dest = pd.read_csv(f"{BASE_DIR}/embeddings{ds}/CIDs_smiles_selfies_{subject}{ds}.csv")
        cids_2 = smiles_df_dest['CIDs']
        cids_rest.append(set(cids_2))

    # Find overlapping indices
    overlapping_indices = [idx for idx, cid in enumerate(cids_1) if all(cid in cids_set for cids_set in cids_rest)]

    return overlapping_indices,cids_1



def common_cids_per_ds(BASE_DIR, ds):
    """
    Return a sorted list of CIDs that appear in ALL subjects for this dataset (ds).
    Looks under: {BASE_DIR}/embeddings/{ds}/
    """
    emb_dir = os.path.join(BASE_DIR, "datasets", ds)

    # Try to find per-subject CSVs
    cid_sets = []

  
    combined = os.path.join(emb_dir, f"{ds}_data.csv")
    
    df = pd.read_csv(combined)
    cid_col ="cid"
    pid_col = "participant_id"
    df = df[[pid_col, cid_col]]
    for _, g in df.groupby("participant_id"):
        cids=set(g["cid"].tolist())
        print(len(cids))
        cid_sets.append(cids)
    

    if not cid_sets:
        return []
    
    filtered = []
    common = set.intersection(*cid_sets)
    for cid in common:
        try:
            filtered.append(int(cid))
        except (ValueError, TypeError):
            continue
    # return as sorted strings (or map to int if you prefer)
    return sorted(filtered, key=lambda x: int(x))



def save_fold_indices(BASE_DIR, n_fold,ds):
    
    common_cids = common_cids_per_ds(BASE_DIR,ds)
    print("ds",ds,len(common_cids))
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=SEED)
    
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(common_cids)):
        # map indices -> actual CID values
        train_c = [str(common_cids[i]) for i in train_idx]
        test_c  = [str(common_cids[i]) for i in test_idx]

        
        for cid in train_c:
            rows.append({
                "cid": cid,
                "set": "train",
                "n_fold": n_fold,
                "fold_idx": fold_idx,
                "ds": ds
            })
        for cid in test_c:
            rows.append({
                "cid": cid,
                "set": "test",
                "n_fold": n_fold,
                "fold_idx": fold_idx,
                "ds": ds
            })
    
    all_folds_df = pd.DataFrame(rows)
    all_folds_df.to_csv(
        f"{BASE_DIR}/folds/fold_indices_ds-{ds}_nfold-{n_fold}.csv", index=False)


def get_descriptors(ds):
    if ds =='bierling2025':
        return ['intensity','pleasantness','familiar','edible', 'warm','sour', 'cold','sweet','fruit','spices','bakery','garlic', 'fish', 
                    'burnt', 'decayed', 'grass', 'wood', 'chemical','flower', 'musky', 'sweaty', 'ammonia']
    elif ds == 'keller2016':
        return['intensive', 'pleasant','familiar','edible','bakery','sweet','fruit','fish','garlic','spices','cold','sour',
               'burnt','acid','warm','musky','sweaty','ammonia','decayed','wood','grass','flower','chemical']
    elif ds== 'sagar2023_v1':
        pass
    elif ds == 'sagar2023_v2':
        pass
    elif ds == 'sagar2023':
        return [ 'intensity', 'pleasantness', 'fishy', 'burnt', 'sour', 'decayed', 'musky',
    'fruity', 'sweaty', 'cool', 'floral', 'sweet', 'warm', 'bakery', 'spicy']
    else:
        raise ValueError("Unsupported dataset: {}".format(ds))
    
    






def _load_text_for_cids(ds, cids,participant_id, input_type: str) -> pd.Series:
    """
    Returns a Series of molecule strings in the order of `cids`.
    input_type ∈ {'smiles','selfies'}.
    """
    df = pd.read_csv(f"{BASE_DIR}/DATASETS/datasets/{ds}/{ds}_data.csv")
    df = df[df["participant_id"]==participant_id].copy()
    df = df[df["cid"].isin(cids)].copy()

    # enforce provided CID order
    df = df.set_index("cid").loc[cids].reset_index()
    
    return df[input_type].reset_index(drop=True)


def build_hf_text_dataset_for_cids(
    ds: str,
    participant_id,
    behavior_embeddings,   # whatever your loader accepts (indices or names)
    cids,
    input_type,
    finetune_by='beh'
    #add optional tr for fmri
    ,roi=None
    ,tr=None
    
):
    """
    Returns (hf_dataset, num_targets) where hf_dataset has:
      - a text column named exactly input_type ('smiles' or 'selfies')
      - columns prop0..propK-1 (behavior targets)
    Row order matches `cids`.
    """
    # y: your existing loader (handles selection/aggregation)

    if finetune_by=='beh':
        y_np = load_behavior_embeddings(
            ds=ds,
            cids=cids,
            participant_id=participant_id,
            embed_cols=behavior_embeddings,
            group_by_cid=True,
        )
    elif finetune_by=='fmri':
        fmri_data, all_cids = load_fmri_data(participant_id, roi, tr, z_score=False)        

            # fMRI slices (keeps CID order by masking)
        y_np = slice_fmri_by_cids(fmri_data, all_cids, cids)
        
    else:
        raise ValueError("finetune_by must be 'beh' or 'fmri'", finetune_by)
        
    columns = [f"prop{i}" for i in range(y_np.shape[1])]
    y_df = pd.DataFrame(y_np, columns=columns)

    # rename behavior columns to prop0..propK-1
    # x: molecule strings (Series aligned to cids order)
    texts = _load_text_for_cids( ds, cids,participant_id, input_type=input_type)

    # assemble HF dataset (text + props only)
    df = pd.DataFrame({input_type: texts}).reset_index(drop=True)
    df = pd.concat([df, y_df.reset_index(drop=True)], axis=1)
    ds_hf = HFDataset.from_pandas(df)
    return ds_hf, y_np.shape[1]



@torch.no_grad()
def extract_representations(
    *, cids, participant_dest, input_type, out_csv: Path,
    tokenizer: AutoTokenizer, model: AutoModel, model_name: str,
    n_fold: int, i_fold: int,
    behavior_embeddings: str, unfreeze_last_n: int, ds: str,
     token_index: int = 0,embed_type: str='can',participant_source:int = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    texts_df = _load_text_for_cids(ds, cids, participant_dest, input_type=input_type)
    
    texts  = texts_df.astype(str).tolist() 
    assert len(texts) == len(cids), f"Length mismatch: texts={len(texts)} vs cids={len(cids)}"

    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to(device)
    hidden_states = model(**inputs, output_hidden_states=True).hidden_states

    rows = []

    if participant_source is None:
        participant_source= participant_dest
    for layer_idx, hs in enumerate(hidden_states):
        arr = hs[:, token_index, :].detach().cpu().numpy()  # [N, D]
        for mol_idx, vec in enumerate(arr):
            row = {
                "layer": layer_idx,
                "cid": str(cids[mol_idx]),
                "model": model_name,
                "participant_id": participant_dest,
                "n_fold": n_fold,
                "i_fold": i_fold,
                "behavior_embeddings": behavior_embeddings,
                "unfreeze_last_n": unfreeze_last_n,
                "ds": ds,
                "participant_source_id": participant_source,
            }
            for d, val in enumerate(vec):
                row[f"{embed_type}_e{d}"] = float(val)
            rows.append(row)
    append_csv(pd.DataFrame(rows), out_csv)

def get_latest_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    candidates = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not candidates:
        raise FileNotFoundError(f"No 'checkpoint-*' subdirs under: {path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def build_models_dir(out_dir: str, run_id: str) -> Path:
    return Path(BASE_DIR) / f"{out_dir}_finetune_models_{run_id}"

def build_embeds_dir(out_dir: str, run_id: str) -> Path:
    # keep run_id to avoid collisions across runs
    return Path(BASE_DIR) / f"{out_dir}_fembeddings_{run_id}"

def build_transfer_embeds_dir(out_dir: str, run_id: str) -> Path:
    # keep run_id to avoid collisions across runs
    return Path(BASE_DIR) / f"{out_dir}_transferred_fembeddings_{run_id}"
def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    df.to_csv(path, mode=("a" if exists else "w"), header=not exists, index=False)


# # --- Progressive unfreezing helpers -----------------------------------------
# def _get_encoder_layers(model):
#     """
#     Try to locate the list/ModuleList of encoder blocks for common HF models.
#     Returns (layers, total_layers). layers can be an iterable of blocks or None.
#     """
#     enc = getattr(model, model.base_model_prefix, None)
#     for attr in ["encoder", "transformer", "layers"]:
#         enc = getattr(enc, attr, enc)
#     layers = None
#     for attr in ["layer", "layers", "h"]:
#         layers = getattr(enc, attr, None) if enc is not None else None
#         if layers is not None:
#             break
#     if hasattr(layers, "__len__"):
#         return layers, len(layers)
#     return None, 0


# finetune_utils/progressive_unfreeze.py
from typing import Optional, Tuple, List
from transformers import TrainerCallback
import math
import torch

def _is_norm_or_bias(name: str) -> bool:
    nl = name.lower()
    return (
        nl.endswith(".bias")
        or "norm" in nl
        or "layernorm" in nl
        or "rmsnorm" in nl
        or "layer_norm" in nl
    )

def _get_encoder_layers(model) -> Tuple[Optional[List[torch.nn.Module]], int]:
    """
    Try to find the stack of encoder blocks across common HF backbones.
    Returns (layers_list_like, total_layers).
    """
    enc = getattr(model, getattr(model, "base_model_prefix", ""), None)
    if enc is None:
        # fallbacks for some models
        for attr in ["roberta", "bert", "backbone", "transformer", "model"]:
            enc = getattr(model, attr, None)
            if enc is not None:
                break
    # dive into encoder containers
    for attr in ["encoder", "transformer", "layers"]:
        enc = getattr(enc, attr, enc)
    layers = None
    for attr in ["layer", "layers", "h", "blocks"]:
        layers = getattr(enc, attr, None) if enc is not None else None
        if hasattr(layers, "__len__"):
            break
    total = len(layers) if hasattr(layers, "__len__") else 0
    return layers, total

def freeze_all_but_head(model):
    """
    Freeze everything except classification/regression heads and all norms/biases
    (norms/bias kept trainable for stability).
    """
    for name, p in model.named_parameters():
        if any(k in name for k in ["classifier", "regression", "score", "lm_head"]):
            p.requires_grad = True
        elif _is_norm_or_bias(name):
            p.requires_grad = True
        else:
            p.requires_grad = False

def apply_unfreeze_last_n(model, last_n):
    """
    Make the last `last_n` encoder blocks trainable (plus norms/bias + heads always trainable).
    If last_n is None or 0 -> only head + norms/bias are trainable.
    If last_n >= 1e6 -> unfreeze all blocks.
    """
    layers, total = _get_encoder_layers(model)
    freeze_all_but_head(model)  # baseline: head + norms/bias trainable

    if layers is None or total == 0 or not last_n:
        return

    # sentinel for "all"
    if isinstance(last_n, (int, float)) and last_n >= 1_000_000:
        keep_start = 0
    else:
        keep_start = max(0, total - int(last_n))

    for i, block in enumerate(layers):
        req_grad = i >= keep_start
        for n, p in block.named_parameters(recurse=True):
            if _is_norm_or_bias(n):
                p.requires_grad = True
            else:
                p.requires_grad = req_grad

    # (optional) if fully unfrozen, also unfreeze embeddings fully
    if keep_start == 0:
        for name, p in model.named_parameters():
            if "embeddings" in name and not _is_norm_or_bias(name):
                p.requires_grad = True

class ProgressiveUnfreezeCallback(TrainerCallback):
    """
    Unfreeze schedule like [(0,0), (1,2), (3,4), (5,"all")].
    - "all" is normalized to a large sentinel.
    - Plan is clamped to num_train_epochs on train begin.
    - Applies epoch-0 state immediately on train begin.
    - Rebuilds optimizer & scheduler whenever trainable params change.
    """
    def __init__(self, schedule):
        norm = []
        for ep, n in schedule:
            if isinstance(n, str) and n.lower() == "all":
                n = 1_000_000
            norm.append((int(ep), None if n in (None, "", "None", 0) else int(n)))
        self.schedule = sorted(norm, key=lambda x: x[0])

        self._active_schedule = self.schedule
        self._applied_epoch = None
        self._last_n_applied = None
        self.trainer = None  # set externally: prog_cb.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        # hard reset in case the same instance is reused
        self._applied_epoch = None
        self._last_n_applied = None
        self._active_schedule = self.schedule

        # Clamp milestones to trial's epoch budget (epochs = 0..n_ep-1)
        n_ep = int(args.num_train_epochs)
        self._active_schedule = [(ep, n) for ep, n in self.schedule if ep < n_ep]

        # Apply initial (epoch 0) state right away
        last_n = self._last_n_for_epoch(0)
        self._apply_and_refresh(last_n, current_epoch=0)
        return control

    def _last_n_for_epoch(self, epoch: int):
        last_n = None
        for ep, n in self._active_schedule:
            if epoch >= ep:
                last_n = n
            else:
                break
        return last_n

    def _rebuild_optim_and_sched(self):
        tr = self.trainer
        if tr is None:
            return
        args = tr.args
        state = tr.state

        # Rebuild optimizer so newly-unfrozen params are included
        tr.optimizer = None
        tr.create_optimizer()

        # Rebuild LR scheduler with remaining steps
        train_loader = tr.get_train_dataloader()
        gas = max(1, args.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(len(train_loader) / gas))
        cur_ep = int(state.epoch or 0)
        epochs_left = max(0, int(args.num_train_epochs) - cur_ep)
        remaining_steps = max(1, steps_per_epoch * epochs_left)

        tr.lr_scheduler = None
        tr.create_scheduler(num_training_steps=remaining_steps)

    def _apply_and_refresh(self, last_n, current_epoch: int):
        # Only re-apply if change
        if self._applied_epoch is None or self._last_n_applied != last_n:
            tr = self.trainer
            if tr is None:
                return
            apply_unfreeze_last_n(tr.model, last_n)
            self._applied_epoch = current_epoch
            self._last_n_applied = last_n

            # Report trainable stats
            m = tr.model
            tot = sum(p.numel() for p in m.parameters())
            trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"[ProgressiveUnfreeze] epoch={current_epoch} -> last_n={last_n} "
                  f"trainable={trn}/{tot} ({trn/tot:.2%})")

            # Ensure optimizer/scheduler include new params
            self._rebuild_optim_and_sched()

    def on_epoch_begin(self, args, state, control, **kwargs):
        cur_epoch = int(state.epoch or 0)
        last_n = self._last_n_for_epoch(cur_epoch)
        self._apply_and_refresh(last_n, current_epoch=cur_epoch)
        return control


