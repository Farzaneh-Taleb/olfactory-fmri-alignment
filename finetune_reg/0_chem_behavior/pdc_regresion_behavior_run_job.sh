#!/bin/bash -l
#SBATCH -A naiss2025-22-958
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -p shared

# set -euo pipefail
mkdir -p logs

# --- Environment ---
module purge
module load miniconda3/24.7.1-0-cpeGNU-23.12
source /cfs/klemming/projects/supr/olfactory_alignment/conda.init.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1

# --- Load the shared grid ---
source "/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/grid.sh"

# --- Index math ---
index=${SLURM_ARRAY_TASK_ID}

num_datasets=${#datasets[@]}
num_subjects=${#subjects[@]}
num_folds=${#folds[@]}
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
n_fold=${folds[$fold_idx]}
c=${n_components[$ncomp_idx]}
model=${models[$model_idx]}
behavior_embedding=${behavior_embeddings[$behavior_idx]}
z_score=${z_scores[$zscore_idx]}

# --- Deterministic per-experiment seed ---
RUN_ID="${RUN_ID:-DEFAULT_RUN}"

echo "RUN_ID=$RUN_ID"
echo "ds=$ds participant_id=$participant_id n_fold=$n_fold n_components=$c model=$model behavior_embeddings='$behavior_embedding' z_score=$z_score"

"$PYTHON_EXEC" regression_behavior_refactored.py \
  --participant_id "$participant_id" \
  --n_components "$c" \
  --model "$model" \
  --behavior_embeddings "$behavior_embedding" \
  --n_fold "$n_fold" \
  --z_score "$z_score" \
  --out_dir "$OUT_DIR" \
  --ds "$ds" \
  --run_id "$RUN_ID"
