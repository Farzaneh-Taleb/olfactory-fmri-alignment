#!/bin/bash

# # (Optional) Load environment for correct modules
# source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# Define variables
subjects=(1 2 3)
n_folds=(10)
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)

# Calculate total number of combinations
n_subjects=${#subjects[@]}
n_folds_len=${#n_folds[@]}
n_models=${#models[@]}
n_behavior_embeddings=${#behavior_embeddings[@]}
n_unfreeze_layers=${#unfreeze_layers[@]}

total_jobs=$(( n_subjects * n_folds_len * n_models * n_behavior_embeddings * n_unfreeze_layers ))

echo "Submitting $total_jobs jobs..."

# Submit the job array
sbatch --array=0-$(($total_jobs - 1)) pdc_extract_reps_finetune_transfer_run_job.sh
# sbatch --array=0-0 ber_extract_reps_finetune_transfer_run_job.sh
