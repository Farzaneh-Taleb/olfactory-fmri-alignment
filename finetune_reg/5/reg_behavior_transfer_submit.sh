#!/bin/bash

# Load environment (optional, recommended for module sanity)
source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# Define parameter grids
folds=(10)
subjects=(1 2 3)
nums_train_epochs=(40)
n_components=(None 30)
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)
z_scores=(True)  # Added z_score grid

# Calculate total combinations
total_jobs=$(( ${#subjects[@]} * ${#folds[@]} * ${#nums_train_epochs[@]} * ${#n_components[@]} * ${#behavior_embeddings[@]} * ${#models[@]} * ${#unfreeze_layers[@]} * ${#z_scores[@]} ))

echo "Submitting $total_jobs jobs..."

# Submit the array job
echo "sbatch --array=0-$((total_jobs - 1)) regresion_behavior_transfer_run_job.sh"
sbatch --array=0-$((total_jobs - 1)) pdc_regresion_behavior_transfer_run_job.sh
# sbatch --array=0-0 pdc_regresion_behavior_transfer_run_job.sh
