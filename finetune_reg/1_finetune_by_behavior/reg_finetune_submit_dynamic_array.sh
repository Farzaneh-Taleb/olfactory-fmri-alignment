#!/bin/bash

# Load environment (optional)
# source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# Define parameter grid
subjects=(1 2 3)
n_folds=(10)
models=("ibm/MoLFormer-XL-both-10pct" "seyonec/ChemBERTa-zinc-base-v1" "HUBioDataLab/SELFormer" "jonghyunlee/ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)

# Calculate total jobs
total_jobs=$(( ${#subjects[@]} * ${#n_folds[@]} * ${#models[@]} * ${#behavior_embeddings[@]} * ${#unfreeze_layers[@]} ))

echo "Submitting $total_jobs jobs..."
echo "sbatch --array=0-$((total_jobs - 1)) pdc_reg_finetune_run_job.sh"

# Submit the array job
sbatch --array=0-$((total_jobs - 1)) pdc_reg_finetune_run_job.sh
# sbatch --array=0-0 pdc_reg_finetune_run_job.sh  # For a single test run