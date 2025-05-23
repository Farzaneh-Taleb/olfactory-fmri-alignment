#!/bin/bash

# Load environment (optional, recommended for module sanity)
source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# Define parameter grids
folds=(5)
subjects=(1 2 3)
nums_train_epochs=(40)
n_components=(None 30)
models=("MoLFormer-XL-both-10pct" "ChemBERTa-zinc-base-v1" "SELFormer" "ChemBERT_ChEMBL_pretrained")
behavior_embeddings=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17")
unfreeze_layers=(0)
rois=("PirF" "PirT" "AMY" "OFC")
nc_masks=(False)
func_masks=(True)
trs=(-1 0 1 2 3 4 5)
read_styles=('avg_orig')
z_scores=(True)



# Calculate total combinations
total_jobs=$(( ${#subjects[@]} * ${#folds[@]} * ${#nums_train_epochs[@]} * ${#n_components[@]} * ${#models[@]} * ${#behavior_embeddings[@]} * ${#unfreeze_layers[@]} * ${#rois[@]} * ${#nc_masks[@]} * ${#func_masks[@]} * ${#trs[@]} * ${#read_styles[@]} * ${#z_scores[@]} ))

# Settings
chunk_size=1000            # Jobs per array
sleep_between_batches=5    # Seconds between individual batch submissions
sleep_between_retries=600  # Seconds between retries

# Initialize all batches
declare -a batches_to_submit=()

for start in $(seq 0 $chunk_size $((total_jobs - 1))); do
    batches_to_submit+=($start)
done

echo "Initial total batches: ${#batches_to_submit[@]}"

# Keep retrying until no batches left
while [ ${#batches_to_submit[@]} -gt 0 ]; do
    echo "New submission pass: ${#batches_to_submit[@]} batches to submit"

    failed_batches=()

    for start in "${batches_to_submit[@]}"; do
        end=$((start + chunk_size - 1))
        if [ $end -ge $total_jobs ]; then
            end=$((total_jobs - 1))
        fi
        count=$((end - start + 1))

        echo "Submitting batch: offset=$start, count=$count..."
        sbatch --export=OFFSET=$start,COUNT=$count --array=0-$((count-1)) pdc_regresion_fmri_run_job.sh
        submit_status=$?

        if [ $submit_status -ne 0 ]; then
            echo "Batch starting at offset=$start failed, will retry later."
            failed_batches+=($start)
        else
            echo "Batch starting at offset=$start submitted successfully."
        fi

        sleep "$sleep_between_batches"
    done

    # Update batches to retry
    batches_to_submit=("${failed_batches[@]}")

    if [ ${#batches_to_submit[@]} -gt 0 ]; then
        echo "Some batches failed. Waiting $sleep_between_retries seconds before next retry..."
        sleep "$sleep_between_retries"
    else
        echo "All batches submitted successfully!"
    fi
done
