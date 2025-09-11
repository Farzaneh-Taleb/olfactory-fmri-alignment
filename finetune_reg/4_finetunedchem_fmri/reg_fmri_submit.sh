#!/bin/bash
# set -euo pipefail

# Path to your grid.sh (the one you shared)
GRID_FILE="/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid2.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }

# Load to compute total count (no need to duplicate values)
# shellcheck source=/dev/null
source "$GRID_FILE"

# FMRI axes (not in grid.sh)
rois=("PirF" "PirT" "AMY" "OFC")
trs=(0 1 2 3 4 5 -1)

# Sizes
num_ds=${#datasets[@]}
num_subjects=${#subjects[@]}
num_folds=${#n_folds[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_ncomp=${#n_components[@]}
num_rois=${#rois[@]}
num_trs=${#trs[@]}
num_z=${#z_scores[@]}

total_jobs=$(( num_ds * num_subjects * num_folds * num_models * num_behaviors * num_unfreeze * num_ncomp * num_rois * num_trs * num_z ))
echo "Total job combinations: $total_jobs"

# Chunking to respect Slurm MaxArraySize (often 1001)
CHUNK_SIZE=1000
SLEEP_BETWEEN_BATCHES=5
SLEEP_BETWEEN_RETRIES=600

declare -a starts=()
for start in $(seq 0 $CHUNK_SIZE $((total_jobs - 1))); do
  starts+=("$start")
done

# for start in $(seq 0 $CHUNK_SIZE 2); do
#   starts+=("$start")
# done

echo "Submitting ${#starts[@]} batch(es), chunk size=$CHUNK_SIZE"

while ((${#starts[@]} > 0)); do
  echo "New pass: ${#starts[@]} batch(es) to submit"
  declare -a failed=()

  for start in "${starts[@]}"; do
    end=$(( start + CHUNK_SIZE - 1 ))
    (( end >= total_jobs )) && end=$(( total_jobs - 1 ))
    count=$(( end - start + 1 ))
    echo "  -> submitting offset=$start count=$count"

    sbatch \
      --export=ALL,GRID_FILE="$GRID_FILE",OFFSET="$start",COUNT="$count" \
      --array=0-$((count-1)) \
      pdc_regresion_fmri_run_job.sh || failed+=("$start")

    sleep "$SLEEP_BETWEEN_BATCHES"
  done

  starts=("${failed[@]}")
  if ((${#starts[@]} > 0)); then
    echo "Some batches failed. Retrying after $SLEEP_BETWEEN_RETRIES s..."
    sleep "$SLEEP_BETWEEN_RETRIES"
  else
    echo "All batches submitted successfully!"
  fi
done
