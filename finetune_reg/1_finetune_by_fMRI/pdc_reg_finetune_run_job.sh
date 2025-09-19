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

echo "RUN_ID=${RUN_ID} (shared across all tasks in this campaign)"

# === Index setup with OFFSET + bundle ===
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

# --- Compute total combinations with i_fold as an axis (sum over n_folds) ---
total_combinations=0
for nf in "${n_folds[@]}"; do
  total_combinations=$(( total_combinations + num_datasets * num_subjects * nf * num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss ))
done

for k in $(seq 0 $((tasks_per_job-1))); do
  index=$(( base_index + k ))
  if [ "$index" -ge "$total_combinations" ]; then
    echo "Index $index out of range (max $((total_combinations - 1))). Skipping."
    continue
  fi

  # === Decode grid indices (mixed radix) ===
  tmp=$index

  # Dataset
  ds_idx=$(( tmp % num_datasets ))
  tmp=$(( tmp / num_datasets ))

  # Subject
  subj_idx=$(( tmp % num_subjects ))
  tmp=$(( tmp / num_subjects ))

  # n_fold / i_fold block selection (handles multiple values in n_folds[])
  n_fold=
  for nf in "${n_folds[@]}"; do
    block_size=$(( nf * num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss ))
    if [ "$tmp" -lt "$block_size" ]; then
      n_fold=$nf
      i_fold=$(( tmp % nf ))
      tmp=$(( tmp / nf ))
      break
    else
      tmp=$(( tmp - block_size ))
    fi
  done

  if [ -z "${n_fold}" ]; then
    echo "Internal decode error: could not determine n_fold/i_fold for index=$index"; exit 2
  fi

  # Model
  model_idx=$(( tmp % num_models ))
  tmp=$(( tmp / num_models ))

  # Behavior
  behavior_idx=$(( tmp % num_behaviors ))
  tmp=$(( tmp / num_behaviors ))

  # Unfreeze
  unfreeze_idx=$(( tmp % num_unfreezes ))
  tmp=$(( tmp / num_unfreezes ))

  # LR
  lr_idx=$(( tmp % num_lrs ))
  tmp=$(( tmp / num_lrs ))

  # WD
  wd_idx=$(( tmp % num_wds ))
  tmp=$(( tmp / num_wds ))

  # BS
  bs_idx=$(( tmp % num_bss ))

  # === Extract values ===
  DS=${datasets[$ds_idx]}
  participant_id=${subjects[$subj_idx]}
  model_path=${models[$model_idx]}
  behavior_cols=${behavior_embeddings[$behavior_idx]}
  unfreeze_last_n=${unfreeze_layers[$unfreeze_idx]}
  lr=${lrs[$lr_idx]}
  weight_decay=${weight_decays[$wd_idx]}
  batch_size=${batch_sizes[$bs_idx]}

  echo "=============================="
  echo "RUN_ID=$RUN_ID"
  echo "Global index=$index (task $k of $tasks_per_job in array task $SLURM_ARRAY_TASK_ID)"
  echo "DS=$DS INPUT_TYPE=$INPUT_TYPE OUT_DIR=$OUT_DIR EPOCHS=$EPOCHS"
  echo "participant_id=$participant_id n_fold=$n_fold i_fold=$i_fold"
  echo "model=$model_path"
  echo "behavior_embeddings=$behavior_cols"
  echo "unfreeze_last_n=$unfreeze_last_n lr=$lr weight_decay=$weight_decay batch_size=$batch_size"
  echo "=============================="

  # --- Launch Python job ---
  HPO_TRIALS=50 python reg_finetune_refactored.py \
    --participant_id "$participant_id" \
    --n_fold "$n_fold" \
    --i_fold "$i_fold" \
    --model "$model_path" \
    --behavior_embeddings "$behavior_cols" \
    --unfreeze_last_n "$unfreeze_last_n" \
    --learning_rate "$lr" \
    --weight_decay "$weight_decay" \
    --per_device_train_batch_size "$batch_size" \
    --out_dir "$OUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --ds "$DS" \
    --embed_type "$embed_type" \
    --run_id "$RUN_ID"
done
