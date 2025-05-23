import pandas as pd
import numpy as np
base_dir = '/cfs/klemming/projects/supr/olfactory_alignment'
model_name="MoLFormer-XL-both-10pct"
n_fold=5
num_train_epochs=40

subject_dest=1
subject_source=3
behavior_embeddings="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"
unfreeze_last_n=0
i_folds=[0,1,2,3,4]
for i_fold in i_folds:
    cids_dest = pd.read_csv(f"{base_dir}/read_orig_avg/finetuned_reg_metrics/test_CIDs_{model_name}_{n_fold}_{num_train_epochs}_{subject_dest}_{behavior_embeddings}_{unfreeze_last_n}_{i_fold}.csv")
    cids_dest = cids_dest.values
    # all_cids, _ = read_CIDs(base_dir,subject)
    smiles_df = pd.read_csv(f"{base_dir}/embeddings/CIDs_smiles_selfies_{subject_dest}.csv")
    all_cids = smiles_df["CIDs"].values.tolist()
    indices_test = np.where(np.isin(all_cids, cids_dest.flatten()))[0]


    cids_source = pd.read_csv(f"{base_dir}/read_orig_avg/finetuned_reg_metrics/test_CIDs_{model_name}_{n_fold}_{num_train_epochs}_{subject_source}_{behavior_embeddings}_{unfreeze_last_n}_{i_fold}.csv")
    cids_source= cids_source.values
    # all_cids, _ = read_CIDs(base_dir,subject)
    smiles_df = pd.read_csv(f"{base_dir}/embeddings/CIDs_smiles_selfies_{subject_source}.csv")
    all_cids = smiles_df["CIDs"].values.tolist()
    indices_train = np.where(~np.isin(all_cids, cids_source.flatten()))[0]

    cids_dest_array = [int(cid) for cid in cids_dest]
    cids_source_array = [int(cid) for cid in cids_source]
    #find overlap between indices_test and indices_source
    overlapping_cids = set(cids_dest_array).intersection(set(cids_source_array))
    cids_dest_filtered = [cid for cid in cids_dest_array if cid not in overlapping_cids]


    #remove whatever from cids_test whatever is in overlapping_cids

    print(i_fold,overlapping_cids)
