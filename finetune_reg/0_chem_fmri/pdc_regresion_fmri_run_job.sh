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

# --- Load FMRI grid ---
source "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment/olfactory-fmri-alignment-NEW/fmri_grid.sh"

# --- Index math ---
index=${SLURM_ARRAY_TASK_ID}

num_datasets=${#datasets[@]}
num_subjects=${#subjects[@]}
num_folds=${#folds[@]}
num_components=${#n_components[@]}
num_models=${#models[@]}
num_rois=${#rois[@]}
num_trs=${#trs[@]}
num_zs=${#z_scores[@]}

total_combinations=$(( num_datasets * num_subjects * num_folds * num_components * num_models * num_rois * num_trs * num_zs ))
if (( index >= total_combinations )); then
  echo "Index $index out of range (max $((total_combinations - 1)))."; exit 1
fi

ds_idx=$(( index / (num_subjects * num_folds * num_components * num_models * num_rois * num_trs * num_zs) % num_datasets ))
subj_idx=$(( index / (num_folds * num_components * num_models * num_rois * num_trs * num_zs) % num_subjects ))
fold_idx=$(( index / (num_components * num_models * num_rois * num_trs * num_zs) % num_folds ))
ncomp_idx=$(( index / (num_models * num_rois * num_trs * num_zs) % num_components ))
model_idx=$(( index / (num_rois * num_trs * num_zs) % num_models ))
roi_idx=$(( index / (num_trs * num_zs) % num_rois ))
tr_idx=$(( index / num_zs % num_trs ))
zs_idx=$(( index % num_zs ))

ds=${datasets[$ds_idx]}
subject=${subjects[$subj_idx]}
n_fold=${folds[$fold_idx]}
n_comp=${n_components[$ncomp_idx]}
model=${models[$model_idx]}
roi=${rois[$roi_idx]}
tr=${trs[$tr_idx]}
z_score=${z_scores[$zs_idx]}

# --- Run id + out dir ---
RUN_ID="${RUN_ID:-DEFAULT_RUN}"
OUT_DIR="${OUT_DIR:-May15_reg}"

echo "RUN_ID=$RUN_ID"
echo "Using Python: $PYTHON_EXEC"
$PYTHON_EXEC -V

echo "Config:"
echo "  ds=$ds subject=$subject n_fold=$n_fold n_components=$n_comp"
echo "  model=$model roi=$roi tr=$tr z_score=$z_score"
echo "  OUT_DIR=$OUT_DIR"

# --- Launch ---
python regression_fmri_refactored.py \
  --participant_id "$subject" \
  --n_components "$n_comp" \
  --model "$model" \
  --n_fold "$n_fold" \
  --roi "$roi" \
  --tr "$tr" \
  --z_score "$z_score" \
  --out_dir "$OUT_DIR" \
  --ds "$ds" \
  --run_id "$RUN_ID"