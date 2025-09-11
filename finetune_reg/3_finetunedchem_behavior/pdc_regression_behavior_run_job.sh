#!/bin/bash -l
#
#SBATCH -J regjob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH --gpus 1

set -euo pipefail

# --- Env / modules ---
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

# --- Grid (only used for folds & subjects) ---
GRID_FILE="/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid2.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$GRID_FILE"

# --- Required exports from submitter (per-array constants) ---
: "${OUT_DIR}"
: "${RUN_ID}"
: "${model}"
: "${behavior_embeddings}"
: "${unfreeze_last_n}"
: "${ds}"
: "${z_score}"
: "${n_components}"

# --- Sizes (array spans: FOLD × SUBJECT) ---
num_folds_vals=${#n_folds[@]}
num_subjects=${#subjects[@]}
(( num_folds_vals > 0 && num_subjects > 0 )) || { echo "n_folds or subjects array is empty"; exit 1; }

# --- Decode SLURM_ARRAY_TASK_ID over FOLD × SUBJECT ---
task_id=${SLURM_ARRAY_TASK_ID:-0}
let per_fold=num_subjects
fold_idx=$(( task_id / per_fold ))
subject_idx=$(( task_id % per_fold ))

# --- Bounds check ---
if (( fold_idx >= num_folds_vals || subject_idx >= num_subjects )); then
  echo "Index out of range for task_id=$task_id; nothing to do."
  exit 0
fi

# --- Values ---
N_FOLD=${n_folds[$fold_idx]}
participant_id=${subjects[$subject_idx]}

echo "Running regression per-(model, beh_embed, unfreeze_last_n, ds, z_score, n_components) array task"
echo "  MODEL:            $model"
echo "  DS:               $ds"
echo "  Subject:          $participant_id"
echo "  n_fold:           $N_FOLD"
echo "  Unfreeze last N:  $unfreeze_last_n"
echo "  Embeddings:       $behavior_embeddings"
echo "  z_score:          $z_score"
echo "  n_components:     $n_components"
echo "  OUT_DIR:          $OUT_DIR"
echo "  RUN_ID:           $RUN_ID"
echo "  SLURM_ARRAY_TASK_ID: $task_id"

# --- Optional wait loop for finetuned embeddings ---
# SLEEP_SECS=$((5 * 60))
# until $PYTHON_EXEC check_embeddings_ready.py \
#   --participant_id "$participant_id" \
#   --model "$model" \
#   --n_fold "$N_FOLD" \
#   --out_dir "$OUT_DIR" \
#   --ds "$ds" \
#   --run_id "$RUN_ID" \
#   --unfreeze_last_n "${unfreeze_last_n:-None}" \
#   --behavior_embeddings "${behavior_embedding:-}"
# do
#   echo "[WAIT] Embeddings not ready yet. Sleeping for $SLEEP_SECS seconds..."
#   sleep "$SLEEP_SECS"
# done
# echo "[WAIT] Embeddings ready. Proceeding to regression."

# --- Launch regression ---
$PYTHON_EXEC regression_behavior_refactored.py \
  --participant_id "$participant_id" \
  --model "$model" \
  --n_fold "$N_FOLD" \
  --z_score "$z_score" \
  --out_dir "$OUT_DIR" \
  --ds "$ds" \
  --run_id "$RUN_ID" \
  --unfreeze_last_n "${unfreeze_last_n:-None}" \
  --n_components "$n_components" \
  --behavior_embeddings "$behavior_embeddings"
