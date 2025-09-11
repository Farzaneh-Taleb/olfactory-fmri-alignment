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
behavior_embeddings=("" "pleasantness")       # empty string => use default behavior columns inside your script

unfreeze_layers=(1 2  "None")
lrs=(1e-5)
weight_decays=(0.0)
batch_sizes=(16)
OUT_DIR="Sep11"
EPOCHS=40
RUN_ID="01"
embed_type="can"