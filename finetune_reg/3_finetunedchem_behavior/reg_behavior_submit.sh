#!/bin/bash
# pdc_regression_behavior_submit.sh
# Submits one SLURM array per (model, behavior_embedding, unfreeze_last_n, dataset, z_score, n_components) combo.
# Each array spans (folds × subjects), with optional concurrency %1 and singleton to avoid overlap.

set -euo pipefail

GRID_FILE="/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid2.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$GRID_FILE"

num_folds_vals=${#n_folds[@]}
num_subjects=${#subjects[@]}
(( num_folds_vals > 0 && num_subjects > 0 )) || { echo "n_folds or subjects array is empty"; exit 1; }

jobs_per_array=$(( num_folds_vals * num_subjects ))
echo "Submitting arrays per (model, behavior_embedding, unfreeze_last_n, dataset, z_score, n_components)"
echo "Each array size: folds(${num_folds_vals}) × subjects(${num_subjects}) = ${jobs_per_array}"

# Helper to sanitize strings for job names
sanitize() {
  local s="$1"
  [ -z "$s" ] && s="none"
  echo "$s" | sed 's/[^A-Za-z0-9._-]/_/g'
}

mkdir -p logs

# One RUN_ID for the whole campaign (use existing if exported)
: "${RUN_ID:=rg_$(date +%Y%m%dT%H%M%S)_$RANDOM}"
export RUN_ID

for model in "${models[@]}"; do
  for beh in "${behavior_embeddings[@]}"; do
    for unf in "${unfreeze_layers[@]}"; do
      for ds in "${datasets[@]}"; do
        for zsc in "${z_scores[@]}"; do
          for ncomp in "${n_components[@]}"; do

            model_safe="$(sanitize "$model")"
            beh_safe="$(sanitize "$beh")"
            unf_safe="$(sanitize "$unf")"
            ds_safe="$(sanitize "$ds")"
            zsc_safe="$(sanitize "$zsc")"
            ncomp_safe="$(sanitize "$ncomp")"

            job_name="reg_${model_safe}_${beh_safe}_unf${unf_safe}_${ds_safe}_z${zsc_safe}_nc${ncomp_safe}"

            echo " -> $job_name (array size: ${jobs_per_array}, concurrency: 1)"
            sbatch \
              --job-name="$job_name" \
              --export=ALL,OUT_DIR="${OUT_DIR}",RUN_ID="${RUN_ID}",model="${model}",behavior_embeddings="${beh}",unfreeze_last_n="${unf}",ds="${ds}",z_score="${zsc}",n_components="${ncomp}" \
              --array=0-$((jobs_per_array-1))%1 \
              --dependency=singleton \
              pdc_regression_behavior_run_job.sh

          done
        done
      done
    done
  done
done

echo "All jobs submitted. RUN_ID=${RUN_ID}"
