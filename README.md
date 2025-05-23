# Olfactory fMRI Alignment

This repository supports a pipeline for evaluating how molecular representations align with olfactory fMRI and behavioral responses, with tasks including pretraining regression, finetuning on behavior, extracting representations, and transfer learning.

---

## Directory Structure

### `finetune_reg/`

#### `0_chem_behavior/`
Regression from pretrained chemical representations to behavioral measures.
- `pdc_regresion_behavior_run_job.sh`
- `reg_behavior_submit.sh`
- `regression_behavior.py`

#### `00_chem_fmri/`
Regression from pretrained chemical representations to fMRI data.
- `pdc_regresion_fmri_run_job.sh`
- `reg_fmri_submit.sh`
- `regression_fmri.py`

#### `1_finetune_by_behavior/`
Finetunes molecular models using behavioral data as targets.
- `pdc_reg_finetune_run_job.sh`
- `reg_finetune_submit_dynamic_array.sh`
- `reg_finetune.py`

#### `2_extract_reps_after_finetuning/`
Extracts representations from models finetuned on behavior.
- `extract_reps_finetune.py`
- `extract_reps_finetune_submit_dynamic_array.sh`
- `pdc_extract_reps_finetune_run_job.sh`

#### `3_finetunedchem_behavior/`
Uses finetuned chemical representations to predict behavior.
- `pdc_regresion_behavior_run_job.sh`
- `reg_behavior_submit.sh`
- `regression_behavior.py`

#### `4_finetunedchem_fmri/`
Uses finetuned chemical representations to predict fMRI responses.
- `pdc_regresion_fmri_run_job.sh`
- `reg_fmri_submit.sh`
- `regression_fmri.py`

#### `5_transfer_behavior/`
Evaluates transfer learning: representations from one task used to predict behavior.
- `pdc_regresion_behavior_transfer_run_job.sh`
- `reg_behavior_transfer_submit.sh`
- `regression_behavior_transfer.py`

#### `6_transfer_fmri/`
Transfer learning: representations from other sources used to predict fMRI.
- `pdc_regresion_fmri_run_job.sh`
- `reg_fmri_submit.sh`
- `regression_fmri.py`

#### `22_extract_reps_after_finetuning_for_transfer/`
Extracts representations from finetuned models for use in transfer learning.
- `extract_reps_finetune_transfer.py`
- `extract_reps_finetune_transfer_submit_dynamic_array.sh`
- `pdc_extract_reps_finetune_transfer_run_job.sh`

---

### `utils/`
- `helpers.py`: Common utilities used across training and evaluation scripts.

---

### Root Files
- `extract_representations.ipynb`: Jupyter notebook for interactive exploration and extraction of representations.

---

## Workflow Summary

## Workflow Summary

1. **Download fMRI Data**  
   Download the fMRI dataset from the [NEMO Scripts Repository](https://github.com/viveksgr/NEMO_scripts).

2. **Extract Required Representations**  
   Use `extract_representations.ipynb` to extract molecular representations from pretrained chemical transformer models.

3. **Pretraining Regression** (`0_chem_behavior/`, `00_chem_fmri/`)  
   Evaluate pretrained chemical representations on behavioral and fMRI prediction tasks.

4. **Finetuning** (`1_finetune_by_behavior/`)  
   Finetune chemical models using behavioral data.

5. **Representation Extraction After Finetuning** (`2_extract_reps_after_finetuning/`)  
   Extract learned internal representations from behaviorally finetuned models.

6. **Evaluation Using Finetuned Representations** (`3_finetunedchem_behavior/`, `4_finetunedchem_fmri/`)  
   Assess how well the finetuned representations predict behavioral and fMRI responses.

7. **Transfer Representation Extraction** (`22_extract_reps_after_finetuning_for_transfer/`)  
   Extract finetuned representations specifically for transfer learning tasks.

8. **Transfer Learning** (`5_transfer_behavior/`, `6_transfer_fmri/`)  
   Use representations trained in one participant for prediction in another.

---

## Requirements

- Python 3.8+
- PyTorch, Transformers
- scikit-learn, NumPy, SciPy
- SLURM for job scheduling

Install dependencies:
```bash
pip install -r requirements.txt
