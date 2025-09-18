#!/bin/bash
# set -euo pipefail

# ------------------- CONFIG -------------------
GRID_FILE="/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }

# Load arrays: datasets, subjects, n_folds, models, behavior_embeddings, unfreeze_layers, n_components, z_scores
# shellcheck source=/dev/null
source "$GRID_FILE"

# fMRI axes (define here unless moved into GRID_FILE)
rois=("PirF" "PirT" "AMY" "OFC")
trs=(0 1 2 3 4 5 -1)

# Chunking / concurrency
: "${CHUNK_SIZE:=1000}"          # how many (subject×fold×roi×tr) indices per array
: "${ARRAY_CONCURRENCY:=0}"      # 0 = no %limit, else e.g. 10 => %10
: "${SLEEP_BETWEEN_SUBMITS:=2}"  # seconds between sbatch calls
: "${RETRY_SLEEP:=600}"          # seconds before retrying failed arrays in a pass
# ---------------------------------------------

# Sanity checks
num_subjects=${#subjects[@]}
num_folds=${#n_folds[@]}
num_rois=${#rois[@]}
num_trs=${#trs[@]}
(( num_subjects > 0 )) || { echo "subjects array is empty"; exit 1; }
(( num_folds  > 0 )) || { echo "n_folds array is empty"; exit 1; }
(( num_rois   > 0 )) || { echo "rois array is empty"; exit 1; }
(( num_trs    > 0 )) || { echo "trs array is empty"; exit 1; }

# Helper: sanitize for job names
sanitize() {
  local s="$1"
  [ -z "$s" ] && s="none"
  echo "$s" | sed 's/[^A-Za-z0-9._-]/_/g'
}

mkdir -p logs

# Dimensions
num_ds=${#datasets[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_ncomp=${#n_components[@]}
num_z=${#z_scores[@]}

echo "Granularity: per (model, behavior, unfreeze, dataset, z, ncomp)."
echo "Each array spans subjects × folds × rois × trs (chunked to CHUNK_SIZE=$CHUNK_SIZE)."
echo "ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-0} (0 means no limit)."

# Precompute subject×fold×roi×tr total
total_sfrt=$(( num_subjects * num_folds * num_rois * num_trs ))
echo "Subjects=${num_subjects}, n_folds=${num_folds}, rois=${num_rois}, trs=${num_trs}, S×F×R×T per array=${total_sfrt}"

# Collect all submissions so we can retry failures per pass
declare -a SUBMIT_CMDS=()
declare -a SUBMIT_DESCR=()

for model in "${models[@]}"; do
  for beh in "${behavior_embeddings[@]}"; do
    for unf in "${unfreeze_layers[@]}"; do
      for ds in "${datasets[@]}"; do
        for zsc in "${z_scores[@]}"; do
          for ncomp_idx in "${!n_components[@]}"; do
            ncomp="${n_components[$ncomp_idx]}"

            # Descriptive job base name (no roi/tr here because array spans them)
            model_safe="$(sanitize "$model")"
            beh_safe="$(sanitize "$beh")"
            unf_safe="$(sanitize "$unf")"
            ds_safe="$(sanitize "$ds")"
            zsc_safe="$(sanitize "$zsc")"
            ncomp_safe="$(sanitize "$ncomp")"

            job_base="fmri_${model_safe}_${beh_safe}_unf${unf_safe}_${ds_safe}_z${zsc_safe}_nc${ncomp_safe}"

            # Chunk S×F×R×T space
            for start in $(seq 0 $CHUNK_SIZE $(( total_sfrt > 0 ? total_sfrt - 1 : 0 ))); do
              end=$(( start + CHUNK_SIZE - 1 ))
              (( end >= total_sfrt )) && end=$(( total_sfrt - 1 ))
              count=$(( end - start + 1 ))
              (( count <= 0 )) && continue

              # Array spec with optional concurrency limiter
              if (( ARRAY_CONCURRENCY > 0 )); then
                array_spec="0-$((count-1))%$ARRAY_CONCURRENCY"
              else
                array_spec="0-$((count-1))"
              fi

              # Export per-combo constants; job script decodes subject×fold×roi×tr from OFFSET+TASK_ID
              cmd=(sbatch
                --job-name="${job_base}_ofs${start}_cnt${count}"
                --export=ALL,GRID_FILE="$GRID_FILE",OFFSET="$start",COUNT="$count",\
MODEL="$model",BEH_EMB="$beh",UNFREEZE_LAST_N="$unf",DS="$ds",Z_SCORE="$zsc",NCOMP_IDX="$ncomp_idx"
                --array="$array_spec"
                pdc_regresion_fmri_run_job.sh
              )

              SUBMIT_CMDS+=("$(printf '%q ' "${cmd[@]}")")
              SUBMIT_DESCR+=("$job_base (OFFSET=$start, COUNT=$count, array=$array_spec)")
            done

          done
        done
      done
    done
  done
done

echo "Prepared ${#SUBMIT_CMDS[@]} array submission(s). Submitting in passes with retry on failures."

# Submit with retry passes
pending_idx=($(seq 0 $(( ${#SUBMIT_CMDS[@]} - 1 )) ))

pass=1
while ((${#pending_idx[@]} > 0)); do
  echo "=== PASS $pass: attempting ${#pending_idx[@]} submission(s) ==="
  failed_idx=()

  for i in "${pending_idx[@]}"; do
    echo " -> ${SUBMIT_DESCR[$i]}"
    eval "${SUBMIT_CMDS[$i]}" || failed_idx+=("$i")
    sleep "$SLEEP_BETWEEN_SUBMITS"
  done

  if ((${#failed_idx[@]} > 0)); then
    echo "Some submissions failed (${#failed_idx[@]}). Retrying after ${RETRY_SLEEP}s..."
    pending_idx=("${failed_idx[@]}")
    sleep "$RETRY_SLEEP"
    ((pass++))
  else
    echo "All arrays submitted successfully."
    break
  fi
done
