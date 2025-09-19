#!/bin/bash
# set -euo pipefail

source "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"

TASKS_PER_JOB=1
chunk_size=1000
sleep_between_batches=5
sleep_between_retries=600
dry_run=false  # set to true for testing without sbatch

# --- sum_n_folds: i_fold axis cardinality across all n_folds entries ---
sum_n_folds=0
for nf in "${n_folds[@]}"; do
  sum_n_folds=$(( sum_n_folds + nf ))
done

# --- Total configs includes i_fold as an axis via sum_n_folds ---
total_configs=$(( ${#datasets[@]} * ${#subjects[@]} * sum_n_folds * ${#models[@]} \
                * ${#behavior_embeddings[@]} * ${#unfreeze_layers[@]} \
                * ${#lrs[@]} * ${#weight_decays[@]} * ${#batch_sizes[@]} ))

total_tasks=$(( (total_configs + TASKS_PER_JOB - 1) / TASKS_PER_JOB ))

echo "[$(date)] Total configs: $total_configs"
echo "[$(date)] Bundling $TASKS_PER_JOB per job => $total_tasks tasks"
echo "[$(date)] RUN_ID=${RUN_ID}"

declare -a batches_to_submit=()
for start in $(seq 0 $chunk_size $((total_tasks - 1))); do
  batches_to_submit+=($start)
done

attempt=1
while [ ${#batches_to_submit[@]} -gt 0 ]; do
  echo "[$(date)] Attempt $attempt: ${#batches_to_submit[@]} batches to submit"

  failed_batches=()
  for start in "${batches_to_submit[@]}"; do
    end=$((start + chunk_size - 1))
    [ $end -ge $((total_tasks - 1)) ] && end=$((total_tasks - 1))
    count=$((end - start + 1))

    echo "[$(date)] Submitting batch: offset=$start count=$count (RUN_ID=$RUN_ID)"

    if [ "$dry_run" = true ]; then
      echo "DRY-RUN: sbatch --export=ALL,OFFSET=$start,TASKS_PER_JOB=$TASKS_PER_JOB,RUN_ID=$RUN_ID --array=0-$((count-1)) pdc_reg_finetune_run_job.sh"
    else
      sbatch --export=ALL,OFFSET=$start,TASKS_PER_JOB=$TASKS_PER_JOB,RUN_ID=$RUN_ID \
             --array=0-$((count-1)) pdc_reg_finetune_run_job.sh
      submit_status=$?
      if [ $submit_status -ne 0 ]; then
        echo "[$(date)] Batch offset=$start FAILED (status=$submit_status), will retry."
        failed_batches+=($start)
      else
        echo "[$(date)] Batch offset=$start submitted successfully."
      fi
      sleep "$sleep_between_batches"
    fi
  done

  batches_to_submit=("${failed_batches[@]}")
  if [ ${#batches_to_submit[@]} -gt 0 ]; then
    echo "[$(date)] Some batches failed (${#batches_to_submit[@]}). Retrying in $sleep_between_retries seconds..."
    sleep "$sleep_between_retries"
    attempt=$((attempt+1))
  else
    echo "[$(date)] All batches submitted successfully!"
  fi
done
