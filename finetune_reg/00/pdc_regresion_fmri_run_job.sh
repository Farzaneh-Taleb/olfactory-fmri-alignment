#!/bin/bash -l
#SBATCH -A naiss2024-22-886
#SBATCH -J moljob
#SBATCH -o logs/output_%A_%a.out
#SBATCH -e logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -p shared

# Load environment
module purge
module load miniconda3/24.7.1-0-cpeGNU-23.12
source /cfs/klemming/projects/supr/olfactory_alignment/conda.init.sh
conda activate fmri_proj
export PYTHONNOUSERSITE=1

# Define parameter grids
folds=(5)
subjects=(1 2 3)
nums_train_epochs=(40)
n_components=(None 30)
# models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
models=("behavior" "MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)
rois=("PirF" "PirT" "AMY" "OFC")
nc_masks=(False)
func_masks=(True)
trs=(0 1 2 3 4 5 -1)
read_styles=('avg_orig')
z_scores=(True)

# Compute true index
OFFSET=${OFFSET:-0}
index=$((OFFSET + SLURM_ARRAY_TASK_ID))

# Lengths of parameter arrays
num_folds=${#folds[@]}
num_subjects=${#subjects[@]}
num_epochs=${#nums_train_epochs[@]}
num_components=${#n_components[@]}
num_models=${#models[@]}
num_behaviors=${#behavior_embeddings[@]}
num_unfreeze=${#unfreeze_layers[@]}
num_rois=${#rois[@]}
num_nc_masks=${#nc_masks[@]}
num_func_masks=${#func_masks[@]}
num_trs=${#trs[@]}
num_read_styles=${#read_styles[@]}
num_z_scores=${#z_scores[@]}

# Total number of jobs
total_combinations=$((num_folds * num_subjects * num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores))

# Check index range
if [ "$index" -ge "$total_combinations" ]; then
    echo "Index $index out of range (max $((total_combinations - 1))). Exiting."
    exit 1
fi

# Decode combination
fold_idx=$(( index / (num_subjects * num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_folds ))
subj_idx=$(( index / (num_epochs * num_components * num_models * num_behaviors * num_unfreeze * num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_subjects ))
epoch_idx=$(( index / (num_components * num_models * num_behaviors * num_unfreeze * num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_epochs ))
ncomp_idx=$(( index / (num_models * num_behaviors * num_unfreeze * num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_components ))
model_idx=$(( index / (num_behaviors * num_unfreeze * num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_models ))
behavior_idx=$(( index / (num_unfreeze * num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_behaviors ))
unfreeze_idx=$(( index / (num_rois * num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_unfreeze ))
roi_idx=$(( index / (num_nc_masks * num_func_masks * num_trs * num_read_styles * num_z_scores) % num_rois ))
nc_mask_idx=$(( index / (num_func_masks * num_trs * num_read_styles * num_z_scores) % num_nc_masks ))
func_mask_idx=$(( index / (num_trs * num_read_styles * num_z_scores) % num_func_masks ))
tr_idx=$(( index / (num_read_styles * num_z_scores) % num_trs ))
read_style_idx=$(( index / num_z_scores % num_read_styles ))
z_score_idx=$(( index % num_z_scores ))

# Assign variables
fold=${folds[$fold_idx]}
subject=${subjects[$subj_idx]}
num_train_epochs=${nums_train_epochs[$epoch_idx]}
c=${n_components[$ncomp_idx]}
model=${models[$model_idx]}
behavior_embedding=${behavior_embeddings[$behavior_idx]}
unfreeze_last_n=${unfreeze_layers[$unfreeze_idx]}
roi=${rois[$roi_idx]}
nc_mask=${nc_masks[$nc_mask_idx]}
func_mask=${func_masks[$func_mask_idx]}
tr=${trs[$tr_idx]}
read_style=${read_styles[$read_style_idx]}
z_score=${z_scores[$z_score_idx]}

# Logging
echo "Running configuration:"
echo "  Fold: $fold"
echo "  Subject: $subject"
echo "  Num Train Epochs: $num_train_epochs"
echo "  Components: $c"
echo "  Model: $model"
echo "  Behavior Embedding: $behavior_embedding"
echo "  Unfreeze Last N Layers: $unfreeze_last_n"
echo "  ROI: $roi"
echo "  NC Mask: $nc_mask"
echo "  Func Mask: $func_mask"
echo "  TR: $tr"
echo "  Read Style: $read_style"
echo "  Z-Score: $z_score"
echo "Using Python at: $(which python)"
echo "Running: fold=$fold, subject=$subject, epochs=$num_train_epochs, components=$c, model=$model, behavior_embedding=$behavior_embedding, unfreeze_last_n=$unfreeze_last_n, z_score=$z_score"
echo "Using Python at: $(which python)"
python -V
PYTHON_EXEC=/cfs/klemming/projects/supr/olfactory_alignment/conda-dirs/envs/fmri_proj/bin/python
echo "Running with Python: $PYTHON_EXEC"
$PYTHON_EXEC -V
$PYTHON_EXEC /cfs/klemming/projects/supr/olfactory_alignment/MoLFormer_fMRI/finetune_reg/00/regression_fmri.py \
    --subject "$subject" \
    --num_train_epochs "$num_train_epochs" \
    --n_components "$c" \
    --model "$model" \
    --behavior_embeddings "$behavior_embedding" \
    --n_fold "$fold" \
    --unfreeze_last_n "$unfreeze_last_n" \
    --roi "$roi" \
    --nc_mask "$nc_mask" \
    --func_mask "$func_mask" \
    --tr "$tr" \
    --read_style "$read_style" \
    --z_score "$z_score" \
    --out_dir 'May15_reg'
