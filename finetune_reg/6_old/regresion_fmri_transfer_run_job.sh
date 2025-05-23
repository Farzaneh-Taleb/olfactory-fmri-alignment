#!/bin/bash -l
#
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH --gpus 1

# Load environment
module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fmri_proj

# Define parameter grids
folds=(5)
subjects=(1 2 3)
nums_train_epochs=(40)
n_components=(None 30)
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)
rois=("PirF" "PirT" "AMY" "OFC")

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
num_rois=${#rois[@]}

# Total number of jobs
total_combinations=$((num_folds * num_subjects * num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_rois))

# Optional: Print total combinations for easy tracking
echo "Total job combinations: $total_combinations"

# Check if index is valid
if [ "$index" -ge "$total_combinations" ]; then
  echo "Index $index out of range (max $((total_combinations - 1))). Exiting."
  exit 1
fi

# Decode combination
fold_idx=$(( index / (num_subjects * num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_rois) % num_folds ))
subj_idx=$(( index / (num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_rois) % num_subjects ))
epoch_idx=$(( index / (num_components * num_models * num_behaviors * num_unfreeze * num_rois) % num_epochs ))
ncomp_idx=$(( index / (num_models * num_behaviors * num_unfreeze * num_rois) % num_components ))
model_idx=$(( index / (num_behaviors * num_unfreeze * num_rois) % num_models ))
behavior_idx=$(( index / (num_unfreeze * num_rois) % num_behaviors ))
unfreeze_idx=$(( index / num_rois % num_unfreeze ))
roi_idx=$(( index % num_rois ))

# Assign variables
fold=${folds[$fold_idx]}
subject=${subjects[$subj_idx]}
num_train_epochs=${nums_train_epochs[$epoch_idx]}
c=${n_components[$ncomp_idx]}
model=${models[$model_idx]}
behavior_embedding=${behavior_embeddings[$behavior_idx]}
unfreeze_last_n=${unfreeze_layers[$unfreeze_idx]}
roi=${rois[$roi_idx]}

# Logging
echo "Running configuration:"
echo "  Fold: $fold"
echo "  Subject: $subject"
echo "  Num Train Epochs: $num_train_epochs"
echo "  Components: $c"
echo "  Model: $model"
echo "  Behavior Embedding: $behavior_embedding"
echo "  Unfreeze Last N Layers: $unfreeze_last_n"
echo "  ROI: $roi"
echo "Using Python at: $(which python)"
python -V

# Run the Python script
python -u regression_fmri_transfer.py \
  --subject "$subject" \
  --num_train_epochs "$num_train_epochs" \
  --n_components "$c" \
  --model "$model" \
  --behavior_embedding "$behavior_embedding" \
  --n_fold "$fold" \
  --unfreeze_last_n "$unfreeze_last_n" \
  --roi "$roi" \
  --out_dir 'finetuned_reg_nc_transfer'
