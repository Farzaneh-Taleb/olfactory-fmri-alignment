#!/bin/bash
# FMRI grid (minimal: no nc_mask, func_mask, read_style, num_train_epochs, unfreeze_layers)

datasets=(sagar2023)

subjects=(1 2 3)
folds=(5)
n_components=("None" 0.9)

models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")

rois=("PirF" "PirT" "AMY" "OFC")
trs=(0 1 2 3 4 5 -1)
z_scores=(True)
OUT_DIR="Sep9"
