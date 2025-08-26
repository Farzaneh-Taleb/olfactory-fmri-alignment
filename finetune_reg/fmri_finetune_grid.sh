#!/bin/bash
# grid.sh – defines the experiment grid

# Datasets to iterate over
datasets=(
  "sagar2023"
)

# Subjects, folds, and model/hparam grid
subjects=(1 2 3)
n_folds=(5)
models=(
  "ibm/MoLFormer-XL-both-10pct"
  "seyonec/ChemBERTa-zinc-base-v1"
  "HUBioDataLab/SELFormer"
  "jonghyunlee/ChemBERT_ChEMBL_pretrained"
)
behavior_embeddings=("" "intensity" "pleasantness")       # empty string => use default behavior columns inside your script
unfreeze_layers=(1 2 -1)
lrs=(1e-5 2e-5 3e-5 4e-5 5e-5)
weight_decays=(0.0)
batch_sizes=(16)
OUT_DIR="Aug26"
EPOCHS=20