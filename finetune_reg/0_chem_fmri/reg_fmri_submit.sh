#!/bin/bash
# set -euo pipefail

# source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh
source "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_grid.sh"

compute_total() {
  echo $(( ${#datasets[@]} * ${#subjects[@]} * ${#folds[@]} \
         * ${#n_components[@]} * ${#models[@]} * ${#rois[@]} \
         * ${#trs[@]} * ${#z_scores[@]} ))
}

mkdir -p logs
total_jobs=$(compute_total)

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export OUT_DIR="${OUT_DIR:-May15_reg}"

echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "Submitting $total_jobs jobs..."

sbatch --export=ALL,RUN_ID="$RUN_ID" --array=0-$((total_jobs-1)) "$(dirname "$0")/pdc_regresion_fmri_run_job.sh"
# sbatch --array=0-1 "$(dirname "$0")/pdc_regresion_fmri_run_job.sh"