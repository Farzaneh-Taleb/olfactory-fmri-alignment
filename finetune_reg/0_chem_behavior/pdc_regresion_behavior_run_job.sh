#!/bin/bash -l
#
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH --gpus 1

module --force purge
module load Miniforge3/24.7.1-2-hpc1-bdist
source /software/sse/manual/Miniforge3/24.7.1-2/hpc1-bdist/etc/profile.d/conda.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Using Python at: $(which python)"
python -V
mkdir -p logs


# --- Load experiment grid ---
source "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"


# --- Index math ---
index=${SLURM_ARRAY_TASK_ID}

num_datasets=${#datasets[@]}
num_subjects=${#subjects[@]}
num_folds=${#n_folds[@]}
num_components=${#n_components[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_zscores=${#z_scores[@]}

total_combinations=$(( num_datasets * num_subjects * num_folds * num_components * num_models * num_behaviors * num_zscores ))
if (( index >= total_combinations )); then
  echo "Index $index out of range (max $((total_combinations - 1)))."; exit 1
fi

ds_idx=$(( index / (num_subjects * num_folds * num_components * num_models * num_behaviors * num_zscores) % num_datasets ))
subj_idx=$(( index / (num_folds * num_components * num_models * num_behaviors * num_zscores) % num_subjects ))
fold_idx=$(( index / (num_components * num_models * num_behaviors * num_zscores) % num_folds ))
ncomp_idx=$(( index / (num_models * num_behaviors * num_zscores) % num_components ))
model_idx=$(( index / (num_behaviors * num_zscores) % num_models ))
behavior_idx=$(( index / num_zscores % num_behaviors ))
zscore_idx=$(( index % num_zscores ))

ds=${datasets[$ds_idx]}
participant_id=${subjects[$subj_idx]}
n_fold=${n_folds[$fold_idx]}
c=${n_components[$ncomp_idx]}
model=${models[$model_idx]}
behavior_embedding=${behavior_embeddings[$behavior_idx]}
z_score=${z_scores[$zscore_idx]}

# --- Deterministic per-experiment seed ---
# RUN_ID="${RUN_ID:-DEFAULT_RUN}"

echo "RUN_ID=$RUN_ID"
echo "ds=$ds participant_id=$participant_id n_fold=$n_fold n_components=$c model=$model behavior_embeddings='$behavior_embedding' z_score=$z_score"

python regression_behavior_refactored.py \
  --participant_id "$participant_id" \
  --n_components "$c" \
  --model "$model" \
  --behavior_embeddings "$behavior_embedding" \
  --n_fold "$n_fold" \
  --z_score "$z_score" \
  --out_dir "$OUT_DIR" \
  --ds "$ds" \
  --run_id "$RUN_ID"
