#!/bin/bash -l
#
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH --gpus 1

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

# --- Grid (only used for folds & subjects) ---
GRID_FILE="/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$GRID_FILE"

# --- Required exports from submitter (per-array constants) ---
: "${OUT_DIR}"
: "${RUN_ID}"
: "${model}"
: "${behavior_embedding}"
: "${unfreeze_last_n}"
: "${ds}"

# If model is a full HF path, take basename for script arg
model_path="$model"
model_name="$(basename "$model")"

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
DS="$ds"
N_FOLD=${n_folds[$fold_idx]}
participant_id=${subjects[$subject_idx]}
behavior_cols="$behavior_embedding"
unf_last_n="$unfreeze_last_n"

echo "Running per-(model, beh_embed, unfreeze_last_n, ds) array task"
echo "  MODEL(path):      $model_path"
echo "  Model(name):      $model_name"
echo "  DS:               $DS"
echo "  Subject:          $participant_id"
echo "  Unfreeze last N:  $unf_last_n"
echo "  Embeddings:       $behavior_cols"
echo "  n_fold:           $N_FOLD"
echo "  OUT_DIR:          $OUT_DIR"
echo "  RUN_ID:           $RUN_ID"
echo "  SLURM_ARRAY_TASK_ID: $task_id"

# --- Launch ---
python extract_reps_finetune_transfer.py \
  --ds "$DS" \
  --out_dir "$OUT_DIR" \
  --model "$model_name" \
  --participant_id "$participant_id" \
  --behavior_embeddings "${behavior_cols}" \
  --unfreeze_last_n "$unf_last_n" \
  --run_id "$RUN_ID" \
  --n_fold "$N_FOLD"
