# parameter grid (single source of truth)
datasets=("sagar2023")
folds=(5)                                   # Python loops i_fold in range(n_fold)
subjects=(1 2 3)
nums_train_epochs=(2)
n_components=("None" 30)                    # parser should coerce "None" -> None
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("")  # "" => use get_descriptors(ds)
unfreeze_layers=(0)
z_scores=(True)
OUT_DIR="Aug25"
PYTHON_EXEC=/cfs/klemming/projects/supr/olfactory_alignment/conda-dirs/envs/fmri_proj/bin/python