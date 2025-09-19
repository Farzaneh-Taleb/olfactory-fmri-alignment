#!/bin/bash
# set -euo pipefail

GRID_FILE="/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$GRID_FILE"

num_folds_vals=${#n_folds[@]}
num_subjects=${#subjects[@]}
(( num_folds_vals > 0 && num_subjects > 0 )) || { echo "n_folds or subjects array is empty"; exit 1; }

jobs_per_combo=$(( num_folds_vals * num_subjects ))
echo "Submitting arrays per (model, behavior_embedding, unfreeze_last_n, dataset) combo."
echo "Each array size: folds(${num_folds_vals}) × subjects(${num_subjects}) = ${jobs_per_combo}"

# Helper to sanitize strings for job names
sanitize() {
  local s="$1"
  [ -z "$s" ] && s="none"
  echo "$s" | sed 's/[^A-Za-z0-9._-]/_/g'
}

for model in "${models[@]}"; do
  for beh in "${behavior_embeddings[@]}"; do
    for unf in "${unfreeze_layers[@]}"; do
      for ds in "${datasets[@]}"; do
        model_safe="$(sanitize "$model")"
        beh_safe="$(sanitize "$beh")"
        unf_safe="$(sanitize "$unf")"
        ds_safe="$(sanitize "$ds")"

        job_name="finetune_${model_safe}_${beh_safe}_unf${unf_safe}_${ds_safe}"

        echo "  -> $job_name (array size: ${jobs_per_combo}, concurrency: 1)"

        sbatch \
          --job-name="$job_name" \
          --export=ALL,OUT_DIR="${OUT_DIR}",RUN_ID="${RUN_ID}",model="${model}",behavior_embedding="${beh}",unfreeze_last_n="${unf}",ds="${ds}" \
          --array=0-$((jobs_per_combo-1))%1 \
          --dependency=singleton \
          pdc_extract_reps_finetune_transfer_run_job.sh

      done
    done
  done
done
echo "All jobs submitted."