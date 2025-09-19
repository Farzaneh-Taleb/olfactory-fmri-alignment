#!/usr/bin/env bash
# set -euo pipefail

# ============ CONFIG: set your directories & file names ============
DIR0="0_chem_behavior"
SCRIPT0="reg_behavior_submit.sh"

DIR1="0_chem_fmri"
SCRIPT1="reg_fmri_submit.sh"

DIR2="1_finetune_by_behavior"
SCRIPT2="reg_finetune_submit_dynamic_array.sh"

DIR3="2_extract_reps_after_finetuning"
SCRIPT3="extract_reps_finetune_submit_dynamic_array.sh"

DIR4="3_finetunedchem_behavior"
SCRIPT4="reg_behavior_submit.sh"

DIR5="4_finetunedchem_fmri"
SCRIPT5="reg_fmri_submit.sh"



# Optional: share one RUN_ID across all stages
: "${RUN_ID:=rg_$(date +%Y%m%dT%H%M%S)_$RANDOM}"
export RUN_ID

POLL_SECS=60
# ================================================================

log() { echo "[$(date '+%a %b %d %T %Z %Y')] $*" >&2; }

# Strict numeric parse: only lines like "Submitted batch job 14105612"
# If you ever switch to `sbatch --parsable`, change the awk accordingly.
run_and_capture_ids_in_dir() {
  local dir="$1" script="$2"
  log "Running $script in $dir"
  (
    cd "$dir"
    chmod +x "$script" 2>/dev/null || true
    # Capture only pure integers from the standard sbatch line
    ./"$script" \
      | tee /dev/stderr \
      | awk '/^Submitted batch job [0-9]+$/ {print $4}'
  )
}

# Wait for all job IDs to disappear from squeue.
# We check each ID individually (avoids very long -j argument lists).
wait_for_jobs() {
  local -a ids=("$@")
  # De-duplicate IDs
  local -A seen=(); local uniq=()
  for id in "${ids[@]}"; do
    [[ "$id" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${seen[$id]:-}" ]]; then
      uniq+=("$id"); seen[$id]=1
    fi
  done

  if ((${#uniq[@]}==0)); then
    log "No job IDs found; moving on."
    return 0
  fi

  log "Waiting for ${#uniq[@]} jobs to finish: ${uniq[*]}"
  while true; do
    local remaining=0
    for id in "${uniq[@]}"; do
      # If the job (or its array) is still present, squeue prints at least one line
      # Use || true so errors (e.g., finished/unknown job) don’t stop the script
      local count
      count=$(squeue -h -j "$id" 2>/dev/null | wc -l | tr -d ' ')
      # count>0 => job (or some array tasks) still running/pending
      if [[ "$count" != "0" ]]; then
        remaining=$((remaining + count))
      fi
    done

    if (( remaining == 0 )); then
      log "All ${#uniq[@]} jobs finished."
      break
    fi

    log "Still running/pending: $remaining ... checking again in ${POLL_SECS}s."
    sleep "$POLL_SECS"
  done
}

# ------------------ STAGE 0 ------------------
# mapfile -t STAGE0_IDS < <(run_and_capture_ids_in_dir "$DIR0" "$SCRIPT0")
# wait_for_jobs "${STAGE0_IDS[@]}"



# # ------------------ STAGE 2 ------------------
mapfile -t STAGE2_IDS < <(run_and_capture_ids_in_dir "$DIR2" "$SCRIPT2")
wait_for_jobs "${STAGE2_IDS[@]}"

# ------------------ STAGE 3 ------------------
mapfile -t STAGE3_IDS < <(run_and_capture_ids_in_dir "$DIR3" "$SCRIPT3")
wait_for_jobs "${STAGE3_IDS[@]}"

# ------------------ STAGE 4 ------------------
# mapfile -t STAGE4_IDS < <(run_and_capture_ids_in_dir "$DIR4" "$SCRIPT4")
# wait_for_jobs "${STAGE4_IDS[@]}"


# ------------------ STAGE 1 ------------------
# mapfile -t STAGE1_IDS < <(run_and_capture_ids_in_dir "$DIR1" "$SCRIPT1")
# wait_for_jobs "${STAGE1_IDS[@]}"

# # ------------------ STAGE 5 ------------------
# mapfile -t STAGE5_IDS < <(run_and_capture_ids_in_dir "$DIR5" "$SCRIPT5")
# wait_for_jobs "${STAGE5_IDS[@]}"

log "✅ All 6 stages completed. RUN_ID=${RUN_ID}"
