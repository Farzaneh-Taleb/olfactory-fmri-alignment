#!/bin/bash -l
#SBATCH -A naiss2024-22-886
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -p shared

# Load environment
module purge
module load miniconda3/24.7.1-0-cpeGNU-23.12
source /proj/conda.init.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1

# Define parameter grids
folds=(10)
subjects=(1 2 3)
nums_train_epochs=(40)
n_components=(None 30)
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
# models=("smiles-gpt" "molgpt" "ChemGPT-19M" "ChemGPT-1.2B" "ChemGPT-4.7M" "encoder_BARTSmiles")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)
z_scores=(True)

# Compute job index
index=$SLURM_ARRAY_TASK_ID

# Lengths of parameter arrays
num_folds=${#folds[@]}
num_subjects=${#subjects[@]}
num_epochs=${#nums_train_epochs[@]}
num_components=${#n_components[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_z_scores=${#z_scores[@]}

# Total number of jobs
total_combinations=$((num_folds * num_subjects * num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_z_scores))
if [ "$index" -ge "$total_combinations" ]; then
  echo "Index $index out of range (max $((total_combinations - 1))). Exiting."
  exit 1
fi

# Decode combination
fold_idx=$(( index / (num_subjects * num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_z_scores) % num_folds ))
subj_idx=$(( index / (num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_z_scores) % num_subjects ))
epoch_idx=$(( index / (num_components * num_models * num_behaviors * num_unfreeze * num_z_scores) % num_epochs ))
ncomp_idx=$(( index / (num_models * num_behaviors * num_unfreeze * num_z_scores) % num_components ))
model_idx=$(( index / (num_behaviors * num_unfreeze * num_z_scores) % num_models ))
behavior_idx=$(( index / (num_unfreeze * num_z_scores) % num_behaviors ))
unfreeze_idx=$(( index / num_z_scores % num_unfreeze ))
z_score_idx=$(( index % num_z_scores ))

# Assign variables
fold=${folds[$fold_idx]}
subject=${subjects[$subj_idx]}
num_train_epochs=${nums_train_epochs[$epoch_idx]}
c=${n_components[$ncomp_idx]}
model=${models[$model_idx]}
behavior_embedding=${behavior_embeddings[$behavior_idx]}
unfreeze_last_n=${unfreeze_layers[$unfreeze_idx]}
z_score=${z_scores[$z_score_idx]}

# Logging
echo "Running: fold=$fold, subject=$subject, epochs=$num_train_epochs, components=$c, model=$model, behavior_embedding=$behavior_embedding, unfreeze_last_n=$unfreeze_last_n, z_score=$z_score"
echo "Using Python at: $(which python)"
python -V

# Run the Python script
python -u regression_behavior.py \
  --subject "$subject" \
  --num_train_epochs "$num_train_epochs" \
  --n_components "$c" \
  --model "$model" \
  --behavior_embedding "$behavior_embedding" \
  --n_fold "$fold" \
  --unfreeze_last_n "$unfreeze_last_n" \
  --z_score "$z_score" \
  --out_dir 'May15_finetuned_reg'
