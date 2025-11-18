#!/bin/bash

# reset modules
source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# load fmri grid
source "/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"

compute_total() {
  echo $(( ${#datasets[@]} * ${#subjects[@]} * ${#n_folds[@]} \
  echo $(( ${#datasets[@]} * ${#subjects[@]} * ${#n_folds[@]} \
         * ${#n_components[@]} * ${#models[@]} * ${#rois[@]} \
         * ${#trs[@]} * ${#z_scores[@]} ))
}

mkdir -p logs
total_jobs=$(compute_total)



export OUT_DIR="${OUT_DIR:-May15_reg}"

echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "Submitting $total_jobs jobs..."

sbatch --export=ALL,RUN_ID="$RUN_ID" --array=0-$((total_jobs-1)) "$(dirname "$0")/pdc_regresion_fmri_run_job.sh"
# sbatch --array=0-1 "$(dirname "$0")/pdc_regresion_fmri_run_job.sh"