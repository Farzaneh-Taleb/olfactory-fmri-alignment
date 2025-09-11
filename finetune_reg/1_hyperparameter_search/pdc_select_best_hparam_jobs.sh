#!/bin/bash -l
#SBATCH -A naiss2025-22-958
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -p shared
#SBATCH --mem=8G
# set -euo pipefail

module purge
module load miniconda3/24.7.1-0-cpeGNU-23.12
source /cfs/klemming/projects/supr/olfactory_alignment/conda.init.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Using Python at: $(which python)"
python -V
mkdir -p logs

# ---- Load shared grid ----
GRID_FILE="/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$GRID_FILE"

# ---- Required inputs ----
: "${OUT_DIR:?Set OUT_DIR in the grid file or export OUT_DIR before submitting}"
: "${RUN_ID:?RUN_ID must be exported by submit_all.sh}"

# ---- Folders ----
BASE_DIR="/cfs/klemming/projects/supr/olfactory_alignment"
METRICS_DIR="${BASE_DIR}/${OUT_DIR}_finetune_metrics_${RUN_ID}"
SAVE_DIR="${BASE_DIR}/best_hparam_selection_logs"
mkdir -p "$SAVE_DIR" logs

# ---- Decode array index (5D: ds, model, subject, unfreeze, embedding) ----
num_datasets=${#datasets[@]}
num_models=${#models[@]}
num_subjects=${#subjects[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_embed=${#behavior_embeddings[@]}

task_id=${SLURM_ARRAY_TASK_ID:-0}

stride_model=$(( num_subjects * num_unfreeze * num_embed ))
stride_ds=$(( num_models * stride_model ))
stride_subject=$(( num_unfreeze * num_embed ))
stride_unfreeze=$(( num_embed ))

ds_idx=$(( task_id / stride_ds ))
remain=$(( task_id % stride_ds ))

model_idx=$(( remain / stride_model ))
remain=$(( remain % stride_model ))

subject_idx=$(( remain / stride_subject ))
remain=$(( remain % stride_subject ))

unfreeze_idx=$(( remain / stride_unfreeze ))
embed_idx=$(( remain % stride_unfreeze ))

DS="${datasets[$ds_idx]}"
model_path=${models[$model_idx]}
model_name=$(basename "$model_path")
participant_id=${subjects[$subject_idx]}
unfreeze_last_n=${unfreeze_layers[$unfreeze_idx]}
behavior_embedding=${behavior_embeddings[$embed_idx]}
N_FOLD=${#folds[@]}

echo "Selecting best hyperparameters"
echo "  DS:               $DS"
echo "  OUT_DIR:          $OUT_DIR"
echo "  Model:            $model_name"
echo "  Subject:          $participant_id"
echo "  Unfreeze last N:  $unfreeze_last_n"
echo "  Embedding:        $behavior_embedding"
echo "  n_fold:           $N_FOLD"
echo "  RUN_ID:           $RUN_ID"

python select_best_hparams.py \
  --ds "$DS" \
  --out_dir "$OUT_DIR" \
  --model "$model_name" \
  --participant_id "$participant_id" \
  --behavior_embeddings "$behavior_embedding" \
  --unfreeze_last_n "$unfreeze_last_n" \
  --run_id "$RUN_ID" \
  --n_fold "$N_FOLD" \
  --save_dir "$SAVE_DIR" \
  --metrics_dir "$METRICS_DIR"
