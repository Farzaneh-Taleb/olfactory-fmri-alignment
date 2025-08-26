import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
"""
Common data loading utilities for regression scripts.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import BASE_DIR
def load_embeddings(model_name, layer, z_score=False):
    """
    Load and optionally standardize embeddings.
    
    Args:
        BASE_DIR: Base directory path
        model_name: Name of the model
        subject: Subject identifier
        layer: Layer number
        z_score: Whether to apply z-score normalization
        
    Returns:
        numpy.ndarray: Loaded embeddings
    """
    embeddings = np.load(f'{BASE_DIR}/embeddings/embeddings_{model_name}.npy')
    
    if z_score:
        print("zscored")
        embeddings = np.nan_to_num(embeddings, nan=0)
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    return embeddings


#reviewed
def load_behavior_embeddings(ds,cids,participant_id, embed_cols, group_by_cid=True):
    """
    Load behavior embeddings with optional filtering.

    Args:
        subject (int|str): Subject identifier (matches values in 'subject' column)
        behavior_embeddings (str|None): Comma-separated embedding column names to select.
                                        If None, selects all numeric columns except ['cid', 'subject'].
        group_by_cid (bool): If True, aggregate duplicate CIDs by mean (within the subject).

    Returns:
        numpy.ndarray: Array of shape (n_rows, n_features) with the selected embeddings.
    """
    # --- Load CSV as DataFrame ---
    df_behavior = pd.read_csv(f"{BASE_DIR}/DATASETS/datasets/{ds}/{ds}_data.csv")
    df_behavior = df_behavior[df_behavior["participant_id"] == participant_id].copy()

    # --- Sort by cid first ---
    df_behavior = df_behavior.sort_values(by="cid").reset_index(drop=True)
    # --- Filter rows for the given subject ---
    print(cids)
    print(df_behavior["cid"].values.tolist())
    if cids is not None:
        df_behavior = df_behavior[df_behavior["cid"].isin(cids)].copy()

    # --- Decide which embedding columns to use ---
    
   

    # --- Optional: group by cid (mean across duplicates within this subject) ---
    if group_by_cid:
        # Keep only the needed columns for aggregation: cid + embeddings
        agg_df = df_behavior[["cid"] + embed_cols].groupby("cid", as_index=False).mean(numeric_only=True)
        # After grouping, drop 'cid' before converting to numpy
        behavior = agg_df[embed_cols].to_numpy()
    else:
        # No grouping: just take embeddings in current row order
        behavior = df_behavior[embed_cols].to_numpy()

    print("Behavior shape:", behavior.shape, flush=True)
    return behavior



# def load_model_embeddings(ds,model_name,cids,layer,embed_type):
#     """
#     Load  embeddings with optional filtering.

#     Args:
#         BASE_DIR (str): Base directory path
#         ds (str): Dataset identifier
#     Returns:
#         numpy.ndarray: Array of shape (n_rows, n_features) with the selected embeddings.
#     """
#     # --- Load CSV as DataFrame ---
#     df_embeddings = pd.read_csv(f"{BASE_DIR}/DATASETS/embeddings/{ds}_{model_name}_embeddings.csv")
    
#     #get only embed_type columns iso or can

#     # --- Sort by cid first ---
#     df_embeddings = df_embeddings.sort_values(by="cid").reset_index(drop=True)
#     df_embeddings = df_embeddings[df_embeddings["layer"] == layer].copy()
    
    

#     # --- Filter rows for the given subject ---
#     if cids is not None:
#         df_embeddings = df_embeddings[df_embeddings["cid"].isin(cids)].copy()

   

#     print("Embeddings shape:", df_embeddings.shape, flush=True)
#     return df_embeddings


def load_model_embeddings(ds, model_name, cids, layer, embed_type):
    """
    Load embeddings for a given dataset/model, filtered by layer and (optionally) a CID list.

    Args:
        ds (str): Dataset identifier
        model_name (str): Model name used in the embeddings file
        cids (list|None): List of CIDs to keep (order preserved). If None, keep all.
        layer (int): Layer index to select
        embed_type (str): "iso" (isomeric) or "can" (canonical)

    Returns:
        numpy.ndarray: Array of shape (n_rows, n_features) with the selected embeddings,
                       in the same order as `cids` if provided.
    """
    # --- Load CSV as DataFrame ---
    df = pd.read_csv(f"{BASE_DIR}/DATASETS/embeddings/{ds}_{model_name}_embeddings.csv")

    
    embed_type = str(embed_type).lower().strip()
    if embed_type not in {"iso", "can"}:
        raise ValueError("embed_type must be 'iso' or 'can'.")

    # --- Pick the right block of columns ---
    prefix = "iso_e" if embed_type == "iso" else "can_e"
    emb_cols = [c for c in df.columns if c.startswith(prefix)]
    if not emb_cols:
        raise ValueError(f"No embedding columns found with prefix '{prefix}'. "
                         f"Columns present: {list(df.columns)[:12]}...")

    # --- Filter by layer ---
    df = df[df["layer"] == layer].copy()

    # --- Optional: filter and preserve order of CIDs ---
    # Keep only requested CIDs, in the order given by `cids`
    # (drop any CIDs not present in the file)
    df = df.set_index("cid")
    present = [cid for cid in cids if cid in df.index]
    if not present:
        raise ValueError("None of the requested CIDs are present in the embeddings file.")
    df = df.loc[present]
    
    df = df.sort_values(by="cid")

    # --- Extract and return as numpy array ---
    arr = df[emb_cols].to_numpy(dtype=float)
    print("Embeddings shape:", arr.shape, flush=True)
    return arr


def load_fold_cids(n_fold,i_fold, ds):
    """
    Load train and test CIDs for a specific fold from pre-created fold indices.
    
    Args:
        BASE_DIR: Base directory path
        out_dir: Output directory name
        i_fold: Fold index
        ds: Dataset suffix (empty string for main dataset)
        
    Returns:
        tuple: (train_cids, test_cids) as numpy arrays
    """
    fold_file = f"{BASE_DIR}/DATASETS/folds/fold_indices_ds-{ds}_nfold-{n_fold}.csv"
    
   
    
    fold_df = pd.read_csv(fold_file)

    fold_df = fold_df[fold_df['fold_idx']==i_fold]
    train_cids = fold_df[fold_df['set']=='train'][ "cid"].astype(int).tolist()
    test_cids = fold_df[fold_df['set']=='test'][ "cid"].astype(int).tolist()
    
    return train_cids, test_cids


# def get_all_cids(BASE_DIR, subject):
#     """
#     Get all CIDs for a subject.
    
#     Args:
#         BASE_DIR: Base directory path
#         subject: Subject identifier
        
#     Returns:
#         list: All CIDs for the subject
#     """
#     smiles_df = pd.read_csv(f"{BASE_DIR}/embeddings/CIDs_smiles_selfies_{subject}.csv")
#     return smiles_df["CIDs"].values.tolist()


def split_train_test_indices(all_cids, train_cids, test_cids):
    """
    Split indices into training and test sets using pre-defined CID splits.
    
    Args:
        all_cids: List of all CIDs
        train_cids: Training CIDs from fold indices
        test_cids: Test CIDs from fold indices
        
    Returns:
        tuple: (train_indices, test_indices)
    """
    indices_train = np.where(np.isin(all_cids, train_cids))[0]
    indices_test = np.where(np.isin(all_cids, test_cids))[0]
    
    return indices_train, indices_test


# def prepare_fold_data(BASE_DIR, n_fold, model_name, subject, layer, behavior, out_dir, 
#                      z_score=False, ds=""):
#     """
#     Prepare data for all folds using pre-created fold indices.
    
#     Args:
#         BASE_DIR: Base directory path
#         n_fold: Number of folds
#         model_name: Name of the model
#         subject: Subject identifier
#         layer: Layer number
#         behavior: Behavior data
#         out_dir: Output directory name (used to locate fold indices)
#         z_score: Whether to apply z-score normalization
#         ds: Dataset suffix (empty string for main dataset)
        
#     Returns:
#         tuple: (embeddings_train, embeddings_test, targets_train, targets_test)
#     """
#     embeddings = load_embeddings(BASE_DIR, model_name, subject, layer, z_score)
#     all_cids = get_all_cids(BASE_DIR, subject)
    
#     embeddings_train = []
#     embeddings_test = []
#     targets_train = []
#     targets_test = []
    
#     for i_fold in range(n_fold):
#         train_cids, test_cids = load_fold_cids(BASE_DIR, out_dir, i_fold, ds)
        
#         indices_train, indices_test = split_train_test_indices(all_cids, train_cids, test_cids)
        
#         embedding_train = embeddings[indices_train]
#         embedding_test = embeddings[indices_test]
#         target_train = behavior[indices_train]
#         target_test = behavior[indices_test]
        
#         embeddings_train.append(embedding_train)
#         embeddings_test.append(embedding_test)
#         targets_train.append(target_train)
#         targets_test.append(target_test)
    
#     return embeddings_train, embeddings_test, targets_train, targets_test