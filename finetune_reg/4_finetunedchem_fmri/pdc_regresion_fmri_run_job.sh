#!/bin/bash -l
#SBATCH -A naiss2025-22-958
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -p shared

mkdir -p logs

# --- Environment ---
module purge
module load miniconda3/24.7.1-0-cpeGNU-23.12
source /cfs/klemming/projects/supr/olfactory_alignment/conda.init.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1

PYTHON_EXEC="$(which python)"

# --- Grid arrays (subjects, n_folds, n_components) ---
: "${GRID_FILE:?GRID_FILE must be exported by the submitter}"
[ -f "$GRID_FILE" ] || { echo "Grid file not found: $GRID_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$GRID_FILE"

# --- fMRI axes (define here unless you move them into GRID_FILE) ---
rois=("PirF" "PirT" "AMY" "OFC")
trs=(0 1 2 3 4 5 -1)

# --- Required exports from submit script (per-array constants) ---
: "${OUT_DIR:?OUT_DIR is required}"
: "${RUN_ID:?RUN_ID is required}"

: "${MODEL:?MODEL is required}"
: "${BEH_EMB}"
: "${UNFREEZE_LAST_N:?UNFREEZE_LAST_N is required (can be 'None')}"
: "${DS:?DS is required}"
: "${Z_SCORE:?Z_SCORE is required}"
: "${NCOMP_IDX:?NCOMP_IDX is required (index into n_components[])}"

: "${OFFSET:?OFFSET is required}"
: "${COUNT:?COUNT is required}"

# --- Sanity / ranges ---
num_subjects=${#subjects[@]}
num_folds=${#n_folds[@]}
num_rois=${#rois[@]}
num_trs=${#trs[@]}

(( num_subjects > 0 )) || { echo "subjects array is empty"; exit 1; }
(( num_folds  > 0 )) || { echo "n_folds array is empty"; exit 1; }
(( num_rois   > 0 )) || { echo "rois array is empty"; exit 1; }
(( num_trs    > 0 )) || { echo "trs array is empty"; exit 1; }

if (( NCOMP_IDX < 0 || NCOMP_IDX >= ${#n_components[@]} )); then
  echo "NCOMP_IDX out of range: $NCOMP_IDX (size=${#n_components[@]})"
  exit 1
fi
n_components_val="${n_components[$NCOMP_IDX]}"

local_id=${SLURM_ARRAY_TASK_ID:-0}
if (( local_id < 0 || local_id >= COUNT )); then
  echo "Local array index $local_id out of [0,$((COUNT-1))]"
  exit 1
fi

# --- Decode subject × fold × roi × tr from linear index ---
linear=$(( OFFSET + local_id ))
total_sfrt=$(( num_subjects * num_folds * num_rois * num_trs ))
if (( linear < 0 || linear >= total_sfrt )); then
  echo "Global linear index $linear out of [0,$((total_sfrt-1))]"
  exit 1
fi

# order: subjects × folds × rois × trs
stride_f=$(( num_rois * num_trs ))
stride_s=$(( num_folds * stride_f ))

subj_idx=$(( linear / stride_s ))
rem=$(( linear % stride_s ))

fold_idx=$(( rem / stride_f ))
rem2=$(( rem % stride_f ))

roi_idx=$(( rem2 / num_trs ))
tr_idx=$(( rem2 % num_trs ))

participant_id="${subjects[$subj_idx]}"
fold="${n_folds[$fold_idx]}"
roi="${rois[$roi_idx]}"
tr="${trs[$tr_idx]}"

echo "Running fMRI regression (combo-per-array; chunked subjects×folds×rois×trs)"
echo "  DS:               $DS"
echo "  Subject:          $participant_id (idx=$subj_idx / ${num_subjects})"
echo "  n_fold:           $fold (idx=$fold_idx / ${num_folds})"
echo "  ROI / TR:         $roi / $tr (roi_idx=$roi_idx / ${num_rois}, tr_idx=$tr_idx / ${num_trs})"
echo "  MODEL:            $MODEL"
echo "  BEH_EMB:          $BEH_EMB"
echo "  UNFREEZE_LAST_N:  $UNFREEZE_LAST_N"
echo "  Z_SCORE:          $Z_SCORE"
echo "  NCOMP_IDX/value:  $NCOMP_IDX / $n_components_val"
echo "  OUT_DIR:          $OUT_DIR"
echo "  RUN_ID:           $RUN_ID"
echo "  OFFSET/COUNT:     $OFFSET / $COUNT"
echo "  SLURM_ARRAY_TASK_ID: $local_id"
echo "Python: $PYTHON_EXEC"; python -V

export RUN_ID  # make available to Python if not passed

# --- Launch regression ---
"$PYTHON_EXEC" regression_fmri.py \
  --participant_id "$participant_id" \
  --model "$MODEL" \
  --ds "$DS" \
  --n_components "$n_components_val" \
  --out_dir "$OUT_DIR" \
  --n_fold "$fold" \
  --z_score "$Z_SCORE" \
  --roi "$roi" \
  --tr "$tr" \
  --behavior_embeddings "$BEH_EMB" \
  --unfreeze_last_n "$UNFREEZE_LAST_N" \
  --run_id "$RUN_ID"
