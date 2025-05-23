# base_dir='/proj/rep-learning-robotics/users/x_farzt'
base_dir = '/cfs/klemming/projects/supr/olfactory_alignment'
import sys
parent_dir = f'{base_dir}/MoLFormer_fMRI'
sys.path.append(parent_dir)
from typing import Dict

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from sklearn.metrics import mean_squared_error
import argparse
from utils.helpers import *
from sklearn.model_selection import KFold
from collections import defaultdict
import json
import scipy.stats
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback
import glob
import shutil
import torch
from sklearn.preprocessing import StandardScaler
# === Load your CSVs ===
parser = argparse.ArgumentParser(description='chem_exploration')
parser.add_argument('--subject', type=int)
parser.add_argument('--n_fold', type=int)
parser.add_argument('--model_name_path', type=str)
parser.add_argument('--behavior_embedding', type=str)
parser.add_argument('--unfreeze_last_n', type=int)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--num_train_epochs', type=int)

def tokenize(batch, tokenizer,input_type="smiles"):
    # print("Keys in batch:", batch.keys())
    # print("Sample input from batch[input_type]:", batch[input_type][:5])
    # print("Type of batch[input_type]:", type(batch[input_type]))
    tokens = tokenizer(
        batch[input_type],
        padding=True
    )
    label_cols = [col for col in batch if col.startswith("prop")]
    tokens["labels"] = list(zip(*(batch[col] for col in label_cols)))
    return tokens

def compute_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    labels = np.array(labels)
    mse = ((predictions - labels) ** 2).mean(axis=0)
    metrics = {f"mse_{i}": m for i, m in enumerate(mse)}
    metrics["mse_mean"] = mse.mean()
    return metrics

def run_permutation_significance(predicteds, y_tests, times=1000):
    num_targets = y_tests.shape[1]
    p_value_correlation = np.zeros(num_targets)
    p_value_mse = np.zeros(num_targets)

    mse_errors = np.array([mean_squared_error(predicteds[:, i], y_tests[:, i]) for i in range(num_targets)])
    correlations = np.array([scipy.stats.pearsonr(predicteds[:, i], y_tests[:, i])[0] for i in range(num_targets)])

    for t in range(times):
        if t % 100 == 0:
            print(f"Permutation {t}/{times}")
        y_tests_shuffle = y_tests.copy()
        np.random.shuffle(y_tests_shuffle)

        mse_error_shuffle = np.array([mean_squared_error(predicteds[:, i], y_tests_shuffle[:, i]) for i in range(num_targets)])
        correlation_shuffle = np.array([scipy.stats.pearsonr(predicteds[:, i], y_tests_shuffle[:, i])[0] for i in range(num_targets)])

        p_value_correlation += correlation_shuffle > correlations
        p_value_mse += mse_error_shuffle < mse_errors

    p_value_correlation /= times
    p_value_mse /= times

    return correlations, mse_errors, p_value_correlation, p_value_mse

class DebugCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print(f"Checkpoint saved at epoch {state.epoch}")
class ManualSaveOnEvenEpochsCallback(TrainerCallback):
    def __init__(self, base_dir, model_name, i_fold, n_fold, num_train_epochs, subject,behavior_embedding,unfreeze_last_n, test_dataset, mse_tracking):
        self.base_dir = base_dir
        self.model_name = model_name
        self.i_fold = i_fold
        self.n_fold = n_fold
        self.num_train_epochs = num_train_epochs
        self.subject = subject
        self.test_dataset = test_dataset
        self.mse_tracking = mse_tracking
        self.behavior_embedding=behavior_embedding
        self.unfreeze_last_n = unfreeze_last_n

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        epoch = int(state.epoch)
        trainer = self.trainer
        test_results = trainer.predict(self.test_dataset)
        preds = test_results.predictions
        labels = test_results.label_ids
        mse = ((preds - labels) ** 2).mean(axis=0).mean()
        self.mse_tracking.append({
            "fold": self.i_fold,
            "epoch": epoch,
            "mean_mse": mse
        })
        return control


class ManualSaveOnBestEpochCallback(TrainerCallback):
    def __init__(
        self,
        base_dir,
        model_name,
        i_fold,
        n_fold,
        num_train_epochs,
        subject,
        behavior_embedding,
        unfreeze_last_n,
        test_dataset,
        global_preds,
        global_labels
    ):
        self.base_dir = base_dir
        self.model_name = model_name
        self.i_fold = i_fold
        self.n_fold = n_fold
        self.num_train_epochs = num_train_epochs
        self.subject = subject
        self.test_dataset = test_dataset
        self.global_preds = global_preds
        self.global_labels = global_labels
        self.behavior_embedding = behavior_embedding
        self.unfreeze_last_n = unfreeze_last_n

    
    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        
    
        # --- Monkeypatch the prediction_step temporarily ---
        
    
        
    
        # --- Load best checkpoint ---
        if state.best_model_checkpoint is not None:
            trainer.model = AutoModelForSequenceClassification.from_pretrained(
                state.best_model_checkpoint,
                num_labels=trainer.model.config.num_labels,
                trust_remote_code=True
            )
            print(f"Loading best model from {state.best_model_checkpoint}")
            trainer.model.to(trainer.args.device)
        

        # --- Predict safely ---
        test_results = trainer.predict(self.test_dataset)
        preds = test_results.predictions
        labels = test_results.label_ids
    
        self.global_preds.append(preds)
        self.global_labels.append(labels)
        
        return control

        

def main():
    seed = 2024
    set_seeds(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ds=''

    

    args = parser.parse_args()
    subject =args.subject 
    n_fold = args.n_fold
    behavior_embedding = args.behavior_embedding
    unfreeze_last_n = args.unfreeze_last_n
    num_train_epochs= args.num_train_epochs
    out_dir = args.out_dir
    out_dir= out_dir+ ds

    
    model_name_path = args.model_name_path
    model_path= model_name_path.split("/")[0]
    model_name = model_name_path.split("/")[1]

    if model_name == "SELFormer":
        input_type = "selfies"
    else:
        input_type = "smiles"

    overlapping_indices,_ = find_overlap(base_dir, ds, subject)

    
    labels_array = np.load(f"{base_dir}/embeddings{ds}/embeddings_behavior_{subject}_1{ds}.npy")
    smiles_df = pd.read_csv(f"{base_dir}/embeddings{ds}/CIDs_smiles_selfies_{subject}{ds}.csv")
    #drop na rows
    
    #fill labels_array nans with 0
    # labels_array = np.nan_to_num(labels_array, nan=0.0)
    # smiles_df = smiles_df.dropna(subset=["smiles"])
    # labels_array = labels_array[~np.isnan(labels_array).any(axis=1)]
    # print(labels_array.shape,"labels_array shape",smiles_df.shape,"smiles_df shape")
    
    behavior_indices = list(map(int, behavior_embedding.split(",")))
    if len(behavior_indices)==1 and behavior_indices[0]==-1:
        print("all indices")
        pass
    else:
        print("behavior indices",behavior_indices)
        labels_array = labels_array[:, behavior_indices]
    
    labels_array = np.nan_to_num(labels_array, nan=0)
    scaler = StandardScaler()
    labels_array = scaler.fit_transform(labels_array)  
    
    num_targets = labels_array.shape[1]
    label_columns = [f"prop{i}" for i in range(num_targets)]
    labels_df = pd.DataFrame(labels_array, columns=label_columns)
    df = pd.concat([smiles_df, labels_df], axis=1)
    #keep only rows with overlapping_indices
    df = df.iloc[overlapping_indices].reset_index(drop=True)
    print(df.shape)


    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        train_df_full = train_df
        train_df_train, train_df_val = train_test_split(train_df_full, test_size=0.2, random_state=seed)

        train_dataset = Dataset.from_pandas(train_df_train.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(train_df_val.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))


        train_dataset = train_dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})
        test_dataset = test_dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})
        val_dataset = val_dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})

        columns_to_remove = [col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
        train_dataset = train_dataset.remove_columns(columns_to_remove)
        val_dataset = val_dataset.remove_columns(columns_to_remove)
        test_dataset = test_dataset.remove_columns(columns_to_remove)
        

        test_indices = np.array(test_df["CIDs"])
        folds.append((train_dataset, test_dataset,val_dataset, test_indices))

    global_preds = []
    global_labels = []
    mse_tracking = []
    if not os.path.exists(f"{base_dir}/read_orig_avg/{out_dir}_metrics"):
        os.makedirs(f"{base_dir}/read_orig_avg/{out_dir}_metrics", exist_ok=True)
    if not os.path.exists(f"{base_dir}/read_orig_avg/{out_dir}_models"):
        os.makedirs(f"{base_dir}/read_orig_avg/{out_dir}_models", exist_ok=True)
    for i_fold, (train_dataset, test_dataset,val_dataset, test_indices) in enumerate(folds):
        pd.DataFrame({"test_indices": test_indices}).to_csv(
            f"{base_dir}/read_orig_avg/{out_dir}_metrics/test_CIDs_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embedding}_{unfreeze_last_n}_{i_fold}{ds}.csv", index=False)
        # 0_4_40_1_0_0
        model = AutoModelForSequenceClassification.from_pretrained(model_name_path, num_labels=num_targets, trust_remote_code=True)
        model.config.problem_type = "regression"

        # === Freeze all layers except the last N ===
        if unfreeze_last_n > 0:
            total_layers = max([
                int(name.split("encoder.layer.")[1].split(".")[0])
                for name, _ in model.named_parameters()
                if "encoder.layer." in name
            ]) + 1

            for name, param in model.named_parameters():
                if "encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    if layer_num < total_layers - unfreeze_last_n:
                        param.requires_grad = False
                elif not any(k in name for k in ["classifier", "regression", "score"]):
                    param.requires_grad = False
        # else:
        #     for name, param in model.named_parameters():
        #         if "embeddings" in name:
        #             param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params / total_params:.2%})")
        output_dir = os.path.join(
        base_dir,
        F"{out_dir}_models",
        f"model_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embedding}_{unfreeze_last_n}_{i_fold}{ds}"
        )
 
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=num_train_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",   # evaluate every epoch
            # logging_strategy="epoch",  # Add this
            # save_steps=None,  # Ensure step-based saving is disabled
            # logging_steps=1,           # Add this
            learning_rate=3e-5,
            load_best_model_at_end=True,  # restore best checkpoint
            metric_for_best_model="mse_mean",
            greater_is_better=False,
            no_cuda=False,
            save_total_limit=5,
            save_safetensors=False,  # <-- Add this
        )
        cb=ManualSaveOnEvenEpochsCallback(
                    base_dir=base_dir,
                    model_name=model_name,
                    i_fold=i_fold,
                    n_fold=n_fold,
                    num_train_epochs=num_train_epochs,
                    subject=subject,
                    behavior_embedding=behavior_embedding,
                    unfreeze_last_n=unfreeze_last_n,
                    test_dataset=test_dataset,
                    mse_tracking=mse_tracking
                )
        cb2=ManualSaveOnBestEpochCallback(
                    base_dir=base_dir,
                    model_name=model_name,
                    i_fold=i_fold,
                    n_fold=n_fold,
                    num_train_epochs=num_train_epochs,
                    subject=subject,
                    behavior_embedding=behavior_embedding,
                    unfreeze_last_n=unfreeze_last_n,
                    test_dataset=test_dataset,
                    global_preds=global_preds,
                    global_labels=global_labels
                )
        cb3 = DebugCallback()
       
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                cb,cb2,cb3,
                EarlyStoppingCallback(early_stopping_patience=5)
            ]
        )
        cb.trainer = trainer
        cb2.trainer = trainer
        cb3.trainer =trainer
        

        trainer.train()

        

        best_model_dir = trainer.state.best_model_checkpoint
        for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
            if path != best_model_dir:
                print(f"Removing {path}...")
                shutil.rmtree(path)


    all_metric_rows = []

    all_preds = np.concatenate(global_preds, axis=0)
    all_labels = np.concatenate(global_labels, axis=0) 
    print(f" {all_preds.shape}, {all_labels.shape}")
    correlations, mse_errors, p_corr, p_mse = run_permutation_significance(all_preds, all_labels, times=1000)
    
    for i in range(len(mse_errors)):
        all_metric_rows.append({
            "target": i,
            "mse": mse_errors[i],
            "correlation": correlations[i],
            "p_value_mse": p_mse[i],
            "p_value_correlation": p_corr[i]
        })
    #check if the directory exists, if not create it
 
    pd.DataFrame(mse_tracking).to_csv(f"{base_dir}/read_orig_avg/{out_dir}_metrics/mean_mse_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embedding}_{unfreeze_last_n}{ds}.csv", index=False)
    pd.DataFrame(all_metric_rows).to_csv(f"{base_dir}/read_orig_avg/{out_dir}_metrics/all_metrics_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embedding}_{unfreeze_last_n}{ds}.csv", index=False)

if __name__ == "__main__":
    main()