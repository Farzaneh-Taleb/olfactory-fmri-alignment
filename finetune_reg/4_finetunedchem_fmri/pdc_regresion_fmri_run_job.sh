#!/bin/bash -l
#
#SBATCH -J finetunedchem_behavior
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH --gpus 1

set -euo pipefail

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

PYTHON_EXEC="${PYTHON_EXEC:-python}"

GRID_FILE="/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid2.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
source "$GRID_FILE"

# FMRI-specific axes (not in grid.sh)
rois=("PirF" "PirT" "AMY" "OFC")
trs=(0 1 2 3 4 5 -1)

# Use arrays from grid.sh:
# datasets, subjects, n_folds, models, behavior_embeddings, unfreeze_layers, n_components, z_scores
# Also OUT_DIR, RUN_ID exist in grid.sh

local_index=${SLURM_ARRAY_TASK_ID}
global_index=$((OFFSET + local_index))

# Sizes
num_ds=${#datasets[@]}
num_subjects=${#subjects[@]}
num_folds=${#n_folds[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_ncomp=${#n_components[@]}
num_rois=${#rois[@]}
num_trs=${#trs[@]}
num_z=${#z_scores[@]}

total_jobs=$(( num_ds * num_subjects * num_folds * num_models * num_behaviors * num_unfreeze * num_ncomp * num_rois * num_trs * num_z ))

if (( global_index < 0 || global_index >= total_jobs )); then
  echo "Index $global_index out of range (0..$((total_jobs-1)))."
  exit 1
fi

# Decode (order must match total_jobs multiplication order above)
ds_idx=$(( global_index / (num_subjects * num_folds * num_models * num_behaviors * num_unfreeze * num_ncomp * num_rois * num_trs * num_z) % num_ds ))
subj_idx=$(( global_index / (num_folds * num_models * num_behaviors * num_unfreeze * num_ncomp * num_rois * num_trs * num_z) % num_subjects ))
fold_idx=$(( global_index / (num_models * num_behaviors * num_unfreeze * num_ncomp * num_rois * num_trs * num_z) % num_folds ))
model_idx=$(( global_index / (num_behaviors * num_unfreeze * num_ncomp * num_rois * num_trs * num_z) % num_models ))
behavior_idx=$(( global_index / (num_unfreeze * num_ncomp * num_rois * num_trs * num_z) % num_behaviors ))
unfreeze_idx=$(( global_index / (num_ncomp * num_rois * num_trs * num_z) % num_unfreeze ))
ncomp_idx=$(( global_index / (num_rois * num_trs * num_z) % num_ncomp ))
roi_idx=$(( global_index / (num_trs * num_z) % num_rois ))
tr_idx=$(( global_index / num_z % num_trs ))
z_idx=$(( global_index % num_z ))

# Values
ds=${datasets[$ds_idx]}
participant_id=${subjects[$subj_idx]}
fold=${n_folds[$fold_idx]}
model=${models[$model_idx]}
beh="${behavior_embeddings[$behavior_idx]}"   # "" means: default behavior columns in Python
unfreeze_last_n="${unfreeze_layers[$unfreeze_idx]}"  # can be "None"
ncomp="${n_components[$ncomp_idx]}"           # "None" or int
roi=${rois[$roi_idx]}
tr=${trs[$tr_idx]}
z_score=${z_scores[$z_idx]}

echo "Global index: $global_index (OFFSET=$OFFSET, local=$local_index)"
echo "ds=$ds subj=$participant_id fold=$fold model=$model beh='$beh' unf=$unfreeze_last_n ncomp=$ncomp roi=$roi tr=$tr z=$z_score"
echo "OUT_DIR=$OUT_DIR RUN_ID=$RUN_ID"
echo "Python: $(which python)"; python -V

export RUN_ID  # picked up by the Python script if not provided as CLI

python regression_fmri.py \
  --participant_id "$participant_id" \
  --model "$model" \
  --ds "$ds" \
  --n_components "$ncomp" \
  --out_dir "$OUT_DIR" \
  --n_fold "$fold" \
  --z_score "$z_score" \
  --roi "$roi" \
  --tr "$tr" \
  --behavior_embeddings "$beh" \
  --unfreeze_last_n "$unfreeze_last_n" \
  --run_id "$RUN_ID"
