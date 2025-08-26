#!/bin/bash -l
#SBATCH -A naiss2025-22-958
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -p shared
#SBATCH --mem=8G

# --- Env ---
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

# --- Defaults ---
OUT_DIR="${OUT_DIR:-Aug25_finetuned_reg}"
INPUT_TYPE="${INPUT_TYPE:-smiles}"
EPOCHS="${EPOCHS:-2}"

# --- Load experiment grid ---
source "/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"

# === Index setup with OFFSET + bundle ===
offset=${OFFSET:-0}
tasks_per_job=${TASKS_PER_JOB:-1}
base_index=$(( offset + SLURM_ARRAY_TASK_ID * tasks_per_job ))

num_datasets=${#datasets[@]}
num_subjects=${#subjects[@]}
num_folds_vals=${#n_folds[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreezes=${#unfreeze_layers[@]}
num_lrs=${#lrs[@]}
num_wds=${#weight_decays[@]}
num_bss=${#batch_sizes[@]}

total_combinations=$((num_datasets * num_subjects * num_folds_vals * num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss))
for k in $(seq 0 $((tasks_per_job-1))); do
    index=$(( base_index + k ))
    if [ "$index" -ge "$total_combinations" ]; then
        echo "Index $index out of range (max $((total_combinations - 1))). Skipping."
        continue
    fi

    RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${k}"
    export RUN_ID

    # === Decode grid indices ===
    ds_idx=$(( index / (num_subjects * num_folds_vals * num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss) % num_datasets ))
    subj_idx=$(( index / (num_folds_vals * num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss) % num_subjects ))
    fold_idx=$(( index / (num_models * num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss) % num_folds_vals ))
    model_idx=$(( index / (num_behaviors * num_unfreezes * num_lrs * num_wds * num_bss) % num_models ))
    behavior_idx=$(( index / (num_unfreezes * num_lrs * num_wds * num_bss) % num_behaviors ))
    unfreeze_idx=$(( index / (num_lrs * num_wds * num_bss) % num_unfreezes ))
    lr_idx=$(( index / (num_wds * num_bss) % num_lrs ))
    wd_idx=$(( index / num_bss % num_wds ))
    bs_idx=$(( index % num_bss ))

    # === Extract values ===
    DS=${datasets[$ds_idx]}
    participant_id=${subjects[$subj_idx]}
    n_fold=${n_folds[$fold_idx]}
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
    echo "participant_id=$participant_id n_fold=$n_fold"
    echo "model=$model_path"
    echo "behavior_embeddings=$behavior_cols"
    echo "unfreeze_last_n=$unfreeze_last_n lr=$lr weight_decay=$weight_decay batch_size=$batch_size"
    echo "=============================="

    # --- Launch Python job ---
    python reg_finetune_refactored.py \
      --participant_id "$participant_id" \
      --n_fold "$n_fold" \
      --model "$model_path" \
      --behavior_embeddings "$behavior_cols" \
      --unfreeze_last_n "$unfreeze_last_n" \
      --learning_rate "$lr" \
      --weight_decay "$weight_decay" \
      --per_device_train_batch_size "$batch_size" \
      --out_dir "$OUT_DIR" \
      --num_train_epochs "$EPOCHS" \
      --ds "$DS" \
      --input_type "$INPUT_TYPE"

done
