#!/bin/bash -l
#SBATCH -A naiss2024-22-886
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -p shared
#SBATCH --mem=8G
# === Load environment ===
module purge
module load Mambaforge/23.3.1-1-hpc1-bdist
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fmri_proj

# === Define grid values ===
subjects=(1 2 3)
folds=(10)
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)

# === Compute grid sizes ===
num_subjects=${#subjects[@]}
num_folds_vals=${#folds[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreezes=${#unfreeze_layers[@]}

total_combinations=$(( num_subjects * num_folds_vals * num_models * num_behaviors * num_unfreezes ))

# === Safety check ===
index=$SLURM_ARRAY_TASK_ID

if [ -z "$index" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID not set."
  exit 1
fi

if [ "$index" -ge "$total_combinations" ]; then
  echo "Index $index out of range (max $((total_combinations - 1))). Exiting."
  exit 1
fi

# === Decode grid indices ===
subj_idx=$(( index / (num_folds_vals * num_models * num_behaviors * num_unfreezes) % num_subjects ))
fold_idx=$(( index / (num_models * num_behaviors * num_unfreezes) % num_folds_vals ))
model_idx=$(( index / (num_behaviors * num_unfreezes) % num_models ))
behavior_idx=$(( index / num_unfreezes % num_behaviors ))
unfreeze_idx=$(( index % num_unfreezes ))

# === Pick actual values ===
subject=${subjects[$subj_idx]}
fold=${folds[$fold_idx]}
model_name=${models[$model_idx]}
behavior_emb=${behavior_embeddings[$behavior_idx]}
unfreeze_layer=${unfreeze_layers[$unfreeze_idx]}

# === Log info ===
echo "Running combo:"
echo "Subject: $subject"
echo "Fold: $fold"
echo "Model: $model_name"
echo "Behavior embedding: $behavior_emb"
echo "Unfreeze layers: $unfreeze_layer"
echo "Python path: $(which python)"
python -V

PYTHON_EXEC=/cfs/klemming/projects/supr/olfactory_alignment/conda-dirs/envs/fmri_proj/bin/python
echo "Running with Python: $PYTHON_EXEC"
$PYTHON_EXEC -V
$PYTHON_EXEC /cfs/klemming/projects/supr/olfactory_alignment/MoLFormer_fMRI/finetune_reg/22/extract_reps_finetune_transfer.py \
    --model_name "$model_name" \
    --input_dir "May15_finetuned_reg" \
    --subject "$subject" \
    --n_fold "$fold" \
    --behavior_embedding "$behavior_emb" \
    --unfreeze_last_n "$unfreeze_layer"
