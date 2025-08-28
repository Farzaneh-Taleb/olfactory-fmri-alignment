#!/bin/bash -l
#SBATCH -A naiss2024-22-886
#SBATCH -J best_hparam_select
#SBATCH -o logs/best_hparam_%A_%a.out
#SBATCH -e logs/best_hparam_%A_%a.err
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -p shared
#SBATCH --mem=4G

module purge
module load miniconda3/24.7.1-0-cpeGNU-23.12
source /cfs/klemming/projects/supr/olfactory_alignment/conda.init.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1

# === Define grid ===
models=(
  "ibm/MoLFormer-XL-both-10pct"
  "seyonec/ChemBERTa-zinc-base-v1"
  "HUBioDataLab/SELFormer"
  "jonghyunlee/ChemBERT_ChEMBL_pretrained"
)
subjects=(1 2 3)
unfreeze_layers=(0 1 2 3 4 5 6)
behavior_embeddings=(
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"
  # add more comma-lists here if desired
)

# === Compute total jobs ===
num_models=${#models[@]}
num_subjects=${#subjects[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_embed=${#behavior_embeddings[@]}

total_jobs=$(( num_models * num_subjects * num_unfreeze * num_embed ))

# === Decode SLURM_ARRAY_TASK_ID ===
task_id=$SLURM_ARRAY_TASK_ID

# four-dimensional indexing
model_idx=$(( task_id / (num_subjects * num_unfreeze * num_embed) ))
remain=$(( task_id % (num_subjects * num_unfreeze * num_embed) ))

subject_idx=$(( remain / (num_unfreeze * num_embed) ))
remain=$(( remain % (num_unfreeze * num_embed) ))

unfreeze_idx=$(( remain / num_embed ))
embed_idx=$(( remain % num_embed ))

# === Extract values ===
model_path=${models[$model_idx]}
model_name=$(basename "$model_path")
subject=${subjects[$subject_idx]}
unfreeze=${unfreeze_layers[$unfreeze_idx]}
embedding=${behavior_embeddings[$embed_idx]}

echo "Selecting best hyperparameters for:"
echo "  Model:           $model_name"
echo "  Subject:         $subject"
echo "  Unfreeze last N: $unfreeze"
echo "  Embedding:       $embedding"

# === Paths ===
base_dir="/cfs/klemming/projects/supr/olfactory_alignment"
out_dir="May27_finetuned_reg"
mkdir -p "${base_dir}/best_hparam_selection_logs"

# === Run Python selector ===
python select_best_hparams.py \
  --model_name "$model_name" \
  --subject "$subject" \
  --behavior_embedding "$embedding" \
  --unfreeze_last_n "$unfreeze" \
  --save_dir "${base_dir}/best_hparam_selection_logs" \
  --metrics_dir "${base_dir}/read_orig_avg/${out_dir}_metrics"
