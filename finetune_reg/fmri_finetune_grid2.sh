#!/bin/bash
# grid.sh – defines the experiment grid
unset -v datasets subjects n_folds models behavior_embeddings unfreeze_layers \
         lrs weight_decays batch_sizes OUT_DIR EPOCHS RUN_ID embed_type \
         z_scores n_components

# Force array semantics (global) so we don't get scalars from env
declare -ag datasets subjects n_folds models behavior_embeddings unfreeze_layers \
             lrs weight_decays batch_sizes z_scores n_components
# Datasets to iterate over
datasets=(
  "sagar2023"
)

# Subjects, folds, and model/hparam grid
subjects=(1 2 3)
n_folds=(5)
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "ChemBERT_ChEMBL_pretrained" "SELFormer")
behavior_embeddings=("" "intensity" "pleasantness")       # empty string => use default behavior columns inside your script
unfreeze_layers=(1 2 3 4 5 "None")
OUT_DIR="Sep10"
EPOCHS=40
RUN_ID="02"
embed_type="can"
z_scores=(True)
n_components=("None" 0.9)   