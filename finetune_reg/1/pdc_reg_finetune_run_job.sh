#!/bin/bash -l
#SBATCH -A naiss2024-22-886
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -p shared
#SBATCH --mem=8G
module purge
module load miniconda3/24.7.1-0-cpeGNU-23.12
source /cfs/klemming/projects/supr/olfactory_alignment/conda.init.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1


echo "Using Python at: $(which python)"
python -V

# === Experiment grid ===
subjects=(1 2 3)
n_folds=(10)
models=("ibm/MoLFormer-XL-both-10pct" "seyonec/ChemBERTa-zinc-base-v1" "HUBioDataLab/SELFormer" "jonghyunlee/ChemBERT_ChEMBL_pretrained")

behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)

# === Index setup ===
index=$SLURM_ARRAY_TASK_ID
num_subjects=${#subjects[@]}
num_folds_vals=${#n_folds[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreezes=${#unfreeze_layers[@]}

total_combinations=$((num_subjects * num_folds_vals * num_models * num_behaviors * num_unfreezes))

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

# === Extract values ===
subject=${subjects[$subj_idx]}
n_fold=${n_folds[$fold_idx]}
model=${models[$model_idx]}
behavior_embedding=${behavior_embeddings[$behavior_idx]}
unfreeze_last_n=${unfreeze_layers[$unfreeze_idx]}

# Log
echo "Running: subject=$subject, n_folds=$n_fold, model=$model, behavior_embedding=$behavior_embedding, unfreeze_last_n=$unfreeze_last_n"
echo "Using Python at: $(which python)"
python -V

PYTHON_EXEC=/cfs/klemming/projects/supr/olfactory_alignment/conda-dirs/envs/fmri_proj/bin/python
echo "Running with Python: $PYTHON_EXEC"
$PYTHON_EXEC -V
$PYTHON_EXEC /cfs/klemming/projects/supr/olfactory_alignment/MoLFormer_fMRI/finetune_reg/1/reg_finetune.py \
  --subject "$subject" \
  --n_fold "$n_fold" \
  --model_name_path "$model" \
  --behavior_embedding "$behavior_embedding" \
  --unfreeze_last_n "$unfreeze_last_n" \
  --out_dir 'May15_finetuned_reg' \
  --num_train_epochs 40
