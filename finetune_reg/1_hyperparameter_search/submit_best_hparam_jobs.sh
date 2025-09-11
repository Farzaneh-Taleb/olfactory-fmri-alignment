#!/bin/bash
# set -euo pipefail

GRID_FILE=""/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/finetune_reg/fmri_finetune_grid.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$GRID_FILE"

: "${OUT_DIR:?Set OUT_DIR in the grid file or export OUT_DIR before submitting}"

num_datasets=${#datasets[@]}
num_models=${#models[@]}
num_subjects=${#subjects[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_embed=${#behavior_embeddings[@]}

total_jobs=$(( num_datasets * num_models * num_subjects * num_unfreeze * num_embed ))
echo "Submitting $total_jobs jobs for best hyperparameter selection..."

# One RUN_ID for the entire array


sbatch \
  --export=ALL,OUT_DIR="$OUT_DIR",RUN_ID="$RUN_ID" \
  --array=0-$((total_jobs - 1)) \
  pdc_select_best_hparam_jobs.sh
