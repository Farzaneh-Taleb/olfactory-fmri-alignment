#!/bin/bash
# set -euo pipefail

# source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
source "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/grid.sh"

compute_total() {
  echo $(( ${#datasets[@]} * ${#subjects[@]} * ${#folds[@]} \
         * ${#n_components[@]} * ${#models[@]} * ${#behavior_embeddings[@]} \
         * ${#z_scores[@]} ))
}

mkdir -p logs
total_jobs=$(compute_total)
echo "Submitting $total_jobs jobs..."
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
echo "RUN_ID=${RUN_ID}"

sbatch --export=ALL,RUN_ID="$RUN_ID" --array=0-$((total_jobs-1)) "$(dirname "$0")/pdc_regresion_behavior_run_job.sh"
