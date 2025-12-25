#!/bin/bash
# pdc_reg_finetune_submit_chunks.sh
set -euo pipefail

source "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"

# Harmonize flag
finetune_by="${finetune_by:-${finetyne_by:-beh}}"

# ---- Tunables (adjust to your cluster policy) ----
TASKS_PER_JOB=${TASKS_PER_JOB:-1}
chunk_size=${chunk_size:-1000}                 # tasks per array submission
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-1000}   # --array ...%CONC
MAX_ACTIVE_ARRAYS=${MAX_ACTIVE_ARRAYS:-1000}   # how many array jobs you keep in queue at once
POLL_SECS=${POLL_SECS:-30}
sleep_between_batches=${sleep_between_batches:-3}
sleep_between_retries=${sleep_between_retries:-600}
dry_run=${dry_run:-false}

# (new) Defaults for search mode exposed to run script/Python
USE_GRID_SEARCH=${USE_GRID_SEARCH:-1}          # deterministic grid (recommended for 125x15)
OPTUNA_TRIALS=${OPTUNA_TRIALS:-0}              # 0 means disabled
export USE_GRID_SEARCH OPTUNA_TRIALS

# ---- Helper: current number of your jobs (all states) ----
my_jobs_count() {
  squeue -h -u "$USER" | wc -l | awk '{print $1}'
}

# ---- i_fold sum ----
sum_n_folds=0
for nf in "${n_folds[@]}"; do sum_n_folds=$((sum_n_folds + nf)); done

# fmri axes
roi_factor=1; tr_factor=1
if [ "${finetune_by}" = "fmri" ]; then
  roi_factor=${#rois[@]}
  tr_factor=${#trs[@]}
fi

# ---- total configs (each config == one TASKS_PER_JOB slot) ----
total_configs=$(( ${#datasets[@]} * ${#subjects[@]} * sum_n_folds * ${#models[@]} \
                * ${#behavior_embeddings[@]} * ${#unfreeze_layers[@]} \
                * ${#lrs[@]} * ${#weight_decays[@]} * ${#batch_sizes[@]} \
                * roi_factor * tr_factor ))

total_tasks=$(( (total_configs + TASKS_PER_JOB - 1) / TASKS_PER_JOB ))
# total_tasks=1

echo "[$(date)] Total configs: $total_configs"
echo "[$(date)] Bundling $TASKS_PER_JOB per job => $total_tasks tasks"
echo "[$(date)] RUN_ID=${RUN_ID}  finetune_by=${finetune_by}"
if [ "${finetune_by}" = "fmri" ]; then
  echo "[$(date)] ROI levels=${#rois[@]} TR levels=${#trs[@]}"
fi
echo "[$(date)] Throttle: MAX_ACTIVE_ARRAYS=$MAX_ACTIVE_ARRAYS, ARRAY_CONCURRENCY=$ARRAY_CONCURRENCY"
echo "[$(date)] Search: USE_GRID_SEARCH=${USE_GRID_SEARCH}, OPTUNA_TRIALS=${OPTUNA_TRIALS}"

# ---- Build batch offsets (each becomes one array job) ----
declare -a pending_offsets=()
for start in $(seq 0 $chunk_size $((total_tasks - 1))); do
  pending_offsets+=("$start")
done

attempt=1
while [ ${#pending_offsets[@]} -gt 0 ]; do
  echo "[$(date)] Attempt $attempt: ${#pending_offsets[@]} arrays left to submit"

  # Throttle: wait until our active arrays < MAX_ACTIVE_ARRAYS
  while true; do
    active=$(my_jobs_count || echo 999999)
    if [[ "$active" =~ ^[0-9]+$ ]] && [ "$active" -lt "$MAX_ACTIVE_ARRAYS" ]; then
      break
    fi
    echo "[$(date)] Queue full ($active >= $MAX_ACTIVE_ARRAYS). Sleeping $POLL_SECS s…"
    sleep "$POLL_SECS"
  done

  failed_next_round=()
  headroom=$(( MAX_ACTIVE_ARRAYS - active ))
  [ $headroom -lt 1 ] && headroom=1
  to_submit_count=$(( headroom < ${#pending_offsets[@]} ? headroom : ${#pending_offsets[@]} ))

  for ((i=0; i<to_submit_count; i++)); do
    start="${pending_offsets[$i]}"
    end=$((start + chunk_size - 1)); [ $end -ge $((total_tasks - 1)) ] && end=$((total_tasks - 1))
    count=$((end - start + 1))

    echo "[$(date)] Submitting array: OFFSET=$start COUNT=$count (RUN_ID=$RUN_ID) conc=%${ARRAY_CONCURRENCY}"

    if [ "$dry_run" = true ]; then
      echo "DRY-RUN: sbatch --export=ALL,OFFSET=$start,TASKS_PER_JOB=$TASKS_PER_JOB,RUN_ID=$RUN_ID,USE_GRID_SEARCH=$USE_GRID_SEARCH,OPTUNA_TRIALS=$OPTUNA_TRIALS --array=0-$((count-1))%${ARRAY_CONCURRENCY} pdc_reg_finetune_run_job.sh"
    else
      sbatch --export=ALL,OFFSET="$start",TASKS_PER_JOB="$TASKS_PER_JOB",RUN_ID="$RUN_ID",USE_GRID_SEARCH="$USE_GRID_SEARCH",OPTUNA_TRIALS="$OPTUNA_TRIALS" \
             --array=0-$((count-1))%${ARRAY_CONCURRENCY} pdc_reg_finetune_run_job.sh
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "[$(date)] ❌ Submission FAILED for offset=$start (rc=$rc). Will retry."
        failed_next_round+=("$start")
      else
        echo "[$(date)] ✅ Submitted offset=$start successfully."
      fi
      sleep "$sleep_between_batches"
    fi
  done

  pending_offsets=("${pending_offsets[@]:$to_submit_count}")

  if [ ${#failed_next_round[@]} -gt 0 ]; then
    pending_offsets+=("${failed_next_round[@]}")
    echo "[$(date)] ${#failed_next_round[@]} arrays failed; retrying them in $sleep_between_retries s…"
    sleep "$sleep_between_retries"
  fi

  attempt=$((attempt+1))
done

echo "[$(date)] All arrays submitted under throttle. Done."
