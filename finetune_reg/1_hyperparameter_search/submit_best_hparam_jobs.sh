#!/bin/bash

# === Define experiment grid ===
models=(
  "ibm/MoLFormer-XL-both-10pct"
  "seyonec/ChemBERTa-zinc-base-v1"
  "HUBioDataLab/SELFormer"
  "jonghyunlee/ChemBERT_ChEMBL_pretrained"
)
subjects=(1 2 3)
unfreeze_last_n=(0 1 2 3 4 5 6)
behavior_embeddings=(
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"
  # add more lists here if needed
)

# === Compute total number of combinations ===
total_jobs=$(( 
  ${#models[@]} * 
  ${#subjects[@]} * 
  ${#unfreeze_last_n[@]} * 
  ${#behavior_embeddings[@]}
))

echo "Submitting $total_jobs jobs for best hyperparameter selection..."
echo "  Models:              ${models[*]}"
echo "  Subjects:            ${subjects[*]}"
echo "  Unfreeze layers:     ${unfreeze_last_n[*]}"
echo "  Behavior embeddings: ${behavior_embeddings[*]}"

# === Fire off the SLURM array ===
sbatch --array=0-$((total_jobs - 1)) pdc_select_best_hparam_jobs.sh
