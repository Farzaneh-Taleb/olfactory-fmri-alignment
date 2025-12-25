#!/bin/bash -l
#
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH --gpus 1
# #SBATCH --cpus-per-task=2   # uncomment if required by site

set -euo pipefail

# -------------------- Determinism & perf env (matches Python) --------------------
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# -------------------- Modules & Conda --------------------
module --force purge
module load Miniforge3/24.7.1-2-hpc1-bdist
source /software/sse/manual/Miniforge3/24.7.1-2/hpc1-bdist/etc/profile.d/conda.sh
conda activate fmri_proj

echo "Using Python at: $(which python)"
python -V
mkdir -p logs

# -------------------- Experiment grid --------------------
source "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"

# Harmonize flag
finetune_by="${finetune_by:-${finetyne_by:-beh}}"

echo "RUN_ID=${RUN_ID} (shared across all tasks in this campaign)"
echo "finetune_by=${finetune_by}"

# -------------------- Optional: LoRA controls via env vars --------------------
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_BIAS="${LORA_BIAS:-none}"                 # none | all | lora_only
LORA_TARGET="${LORA_TARGET:-auto}"             # auto | csv list like: q_proj,k_proj,v_proj,o_proj

echo "LoRA: USE_LORA=${USE_LORA}, R=${LORA_R}, ALPHA=${LORA_ALPHA}, DROPOUT=${LORA_DROPOUT}, BIAS=${LORA_BIAS}, TARGET=${LORA_TARGET}"

# -------------------- Search mode (grid vs Optuna) --------------------
USE_GRID_SEARCH="${USE_GRID_SEARCH:-1}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-0}"
export HPO_TRIALS="${OPTUNA_TRIALS}"   # Python reads this too
echo "Search: USE_GRID_SEARCH=${USE_GRID_SEARCH}, OPTUNA_TRIALS=${OPTUNA_TRIALS}"

# -------------------- Indexing over the grid --------------------
offset=${OFFSET:-0}
tasks_per_job=${TASKS_PER_JOB:-1}
base_index=$(( offset + SLURM_ARRAY_TASK_ID * tasks_per_job ))

num_datasets=${#datasets[@]}
num_subjects=${#subjects[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreezes=${#unfreeze_layers[@]}
num_lrs=${#lrs[@]}
num_wds=${#weight_decays[@]}
num_bss=${#batch_sizes[@]}

# Optional fmri axes (safe defaults if unset)
num_rois=${#rois[@]}
num_trs=${#trs[@]}

# --- Compute total combinations with i_fold as an axis (sum over n_folds) ---
total_combinations=0
for nf in "${n_folds[@]}"; do
  block=$(( num_datasets * num_subjects * nf * num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss ))
  if [ "$finetune_by" = "fmri" ]; then
    block=$(( block * num_rois * num_trs ))
  fi
  total_combinations=$(( total_combinations + block ))
done

for k in $(seq 0 $((tasks_per_job-1))); do
  index=$(( base_index + k ))
  if [ "$index" -ge "$total_combinations" ]; then
    echo "Index $index out of range (max $((total_combinations - 1))). Skipping."
    continue
  fi

  # === Decode grid indices (mixed radix) ===
  tmp=$index

  ds_idx=$(( tmp % num_datasets ));  tmp=$(( tmp / num_datasets ))
  subj_idx=$(( tmp % num_subjects )); tmp=$(( tmp / num_subjects ))

  n_fold=
  for nf in "${n_folds[@]}"; do
    fold_block=$(( nf * num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss ))
    if [ "$finetune_by" = "fmri" ]; then
      fold_block=$(( fold_block * num_rois * num_trs ))
    fi
    if [ "$tmp" -lt "$fold_block" ]; then
      n_fold=$nf
      i_fold=$(( tmp % nf ))
      tmp=$(( tmp / nf ))
      break
    else
      tmp=$(( tmp - fold_block ))
    fi
  done
  [ -z "${n_fold}" ] && { echo "Decode error for index=$index"; exit 2; }

  model_idx=$(( tmp % num_models ));         tmp=$(( tmp / num_models ))
  behavior_idx=$(( tmp % num_behaviors ));   tmp=$(( tmp / num_behaviors ))
  unfreeze_idx=$(( tmp % num_unfreezes ));   tmp=$(( tmp / num_unfreezes ))
  lr_idx=$(( tmp % num_lrs ));               tmp=$(( tmp / num_lrs ))
  wd_idx=$(( tmp % num_wds ));               tmp=$(( tmp / num_wds ))
  bs_idx=$(( tmp % num_bss ));               tmp=$(( tmp / num_bss ))

  if [ "$finetune_by" = "fmri" ]; then
    roi_idx=$(( tmp % num_rois ));  tmp=$(( tmp / num_rois ))
    tr_idx=$(( tmp % num_trs ))
  fi

  # === Extract values ===
  DS=${datasets[$ds_idx]}
  participant_id=${subjects[$subj_idx]}
  model_path=${models[$model_idx]}
  behavior_cols=${behavior_embeddings[$behavior_idx]}
  unfreeze_last_n=${unfreeze_layers[$unfreeze_idx]}
  lr=${lrs[$lr_idx]}
  weight_decay=${weight_decays[$wd_idx]}
  batch_size=${batch_sizes[$bs_idx]}

  if [ "$finetune_by" = "fmri" ]; then
    roi=${rois[$roi_idx]}
    tr=${trs[$tr_idx]}
  fi

  echo "=============================="
  echo "RUN_ID=$RUN_ID"
  echo "Global index=$index (task $k of $tasks_per_job in array task $SLURM_ARRAY_TASK_ID)"
  echo "DS=$DS embed_type=$embed_type OUT_DIR=$OUT_DIR EPOCHS=$EPOCHS"
  echo "participant_id=$participant_id n_fold=$n_fold i_fold=$i_fold finetune_by=$finetune_by"
  echo "model=$model_path"
  echo "behavior_embeddings=$behavior_cols"
  if [ "$finetune_by" = "fmri" ]; then
    echo "ROI=$roi TR=$tr"
  fi
  echo "unfreeze_last_n=$unfreeze_last_n lr=$lr weight_decay=$weight_decay batch_size=$batch_size"
  echo "LoRA USE=${USE_LORA}"
  echo "Grid/Optuna: USE_GRID_SEARCH=${USE_GRID_SEARCH}, OPTUNA_TRIALS=${OPTUNA_TRIALS}"
  echo "=============================="

  # --- Build common python args ---
  EXTRA_ARGS=(
    --participant_id "$participant_id"
    --n_fold "$n_fold"
    --i_fold "$i_fold"
    --model "$model_path"
    --behavior_embeddings "$behavior_cols"
    --unfreeze_last_n "$unfreeze_last_n"
    --learning_rate "$lr"
    --weight_decay "$weight_decay"
    --per_device_train_batch_size "$batch_size"
    --out_dir "$OUT_DIR"
    --num_train_epochs "$EPOCHS"
    --ds "$DS"
    --embed_type "$embed_type"
    --finetune_by "$finetune_by"
    --run_id "$RUN_ID"
    --use_grid_search "$USE_GRID_SEARCH"
    --optuna_trials "$OPTUNA_TRIALS"
  )

  if [ "$USE_LORA" = "1" ]; then
    EXTRA_ARGS+=(
      --use_lora 1
      --lora_r "$LORA_R"
      --lora_alpha "$LORA_ALPHA"
      --lora_dropout "$LORA_DROPOUT"
      --lora_bias "$LORA_BIAS"
      --lora_target "$LORA_TARGET"
    )
  fi

  if [ "$finetune_by" = "fmri" ]; then
    EXTRA_ARGS+=( --roi "$roi" --tr "$tr" )
  fi

  # Payload (plain python is fine on your site; switch to srun if cluster requires)
  python reg_finetune_refactored_lora.py "${EXTRA_ARGS[@]}"
done
