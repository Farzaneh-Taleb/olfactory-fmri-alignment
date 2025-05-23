#!/bin/bash
# run_all.sh

# echo "Running scripts inside 1/"
# cd /proj/rep-learning-robotics/users/x_farzt/MoLFormer_fMRI/finetune_reg/1
# bash reg_finetune_submit_dynamic_array.sh

# echo "Running scripts inside 2/"
# cd /proj/rep-learning-robotics/users/x_farzt/MoLFormer_fMRI/finetune_reg/2
# bash extract_reps_finetune_submit_dynamic_array.sh

# echo "Running scripts inside 3/"
# cd /proj/rep-learning-robotics/users/x_farzt/MoLFormer_fMRI/finetune_reg/3
# bash reg_behavior_submit.sh

echo "Running scripts inside 4/"
cd /proj/rep-learning-robotics/users/x_farzt/MoLFormer_fMRI/finetune_reg/4
bash reg_fmri_submit.sh

# echo "All scripts finished."