import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import json
from utils.config import BASE_DIR, SEED
from utils.helpers import *  # includes run_permutation_significance
from utils.model_config import MODELS, LAYERS_END, INPUT_TYPES_CAN, INPUT_TYPES_ISO
from utils.arg_parser import create_finetune_parser, parse_common_args
from utils.data_loader import  load_fold_cids  # <-- use your existing loader
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datasets import Dataset as HFDataset
from scipy.stats import pearsonr
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModel
)
import optuna
from sklearn.model_selection import train_test_split
import torch
import glob
import shutil
from utils.regression import compute_targetwise_metrics, permutation_test_metrics

parser = create_finetune_parser('finetune_by_behavior')



def tokenize(batch, tokenizer, input_type="smiles"):
    # Don't pad here; let the collator pad
    enc = tokenizer(batch[input_type], truncation=True)
    # collect prop* columns in batch order and build (B,K) float32 list-of-lists
    label_cols = [c for c in batch.keys() if c.startswith("prop")]
    labels = np.stack([batch[c] for c in label_cols], axis=1).astype("float32").tolist()
    enc["labels"] = labels
    return enc


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Per-target MSE + mean MSE."""
    # eval_pred can be tuple or EvalPrediction
    if isinstance(eval_pred, tuple):
        predictions, labels = eval_pred
    else:
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # squeeze if model returns (B,1,D)
    if predictions.ndim == 3 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :]

    mse = ((predictions - labels) ** 2).mean(axis=0)
    metrics = {f"mse_{i}": float(m) for i, m in enumerate(mse)}
    cors = [pearsonr(predictions[:, i], labels[:, i])[0] if labels[:, i].std() > 1e-8 else 0.0
            for i in range(labels.shape[1])]
    metrics["mse_mean"] = float(mse.mean())
    metrics["corr_mean"] = float(np.nanmean(cors))
    

    return metrics


class MSETrackingCallback(TrainerCallback):
    """Callback to track MSE during training."""
    def __init__(self, val_dataset, mse_tracking, fold_id):
        self.val_dataset = val_dataset
        self.mse_tracking = mse_tracking
        self.fold_id = fold_id
        self.trainer = None

    # def on_epoch_end(self, args, state, control, **kwargs):
    #     epoch = int(state.epoch)
    #     val_results = self.trainer.evaluate(eval_dataset=self.val_dataset)
    #     mse_dict = {
    #         "fold": self.fold_id,
    #         "epoch": epoch,
    #         "mean_mse": float(val_results["eval_mse_mean"])
    #     }
    #     self.mse_tracking.append(mse_dict)
    #     return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # metrics already includes eval_* from the scheduled evaluation
        if metrics and "eval_mse_mean" in metrics:
            self.mse_tracking.append({
                "fold": self.fold_id,
                "epoch": int(state.epoch or 0),
                "mean_mse": float(metrics["eval_mse_mean"]),
            })
        return control


class FinalPredictionCallback(TrainerCallback):
    """Callback to collect final predictions from best model."""
    def __init__(self, test_dataset, global_preds, global_labels):
        self.test_dataset = test_dataset
        self.global_preds = global_preds
        self.global_labels = global_labels

    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        if state.best_model_checkpoint is not None:
            trainer.model = AutoModelForSequenceClassification.from_pretrained(
                state.best_model_checkpoint,
                num_labels=trainer.model.config.num_labels,
                trust_remote_code=True
            )
            print(f"Loading best model from {state.best_model_checkpoint}")
            trainer.model.to(trainer.args.device)
        # With load_best_model_at_end=True, best weights are already loaded
        test_results = self.trainer.predict(self.test_dataset)
        preds = test_results.predictions
        labels = test_results.label_ids
        self.global_preds.append(preds)
        self.global_labels.append(labels)
        return control


# def create_model_and_freeze_layers(model_name_path, num_targets, unfreeze_last_n):
#     """Create model and apply layer freezing strategy."""
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name_path, num_labels=num_targets, trust_remote_code=True
#     )
#     model.config.problem_type = "regression"
    
#     # Freeze layers except the last N
#     if unfreeze_last_n is not None:
#         try:
#             total_layers = max([
#                 int(name.split("encoder.layer.")[1].split(".")[0])
#                 for name, _ in model.named_parameters()
#                 if "encoder.layer." in name
#             ]) + 1
#         except Exception:
#             total_layers = None

#         for name, param in model.named_parameters():
#             if total_layers is not None and "encoder.layer." in name:
#                 layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
#                 param.requires_grad = layer_num >= total_layers - unfreeze_last_n
#             else:
#                 # keep head trainable
#                 if any(k in name for k in ["classifier", "regression", "score"]):
#                     param.requires_grad = True
#                 else:
#                     if total_layers is None:
#                         # fallback: freeze non-head params if we can't parse layers
#                         param.requires_grad = False

#     # Print parameter statistics
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params / total_params:.2%})")
#     return model



def create_model_and_freeze_layers(model_name_path, num_targets, unfreeze_last_n):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_path, num_labels=num_targets, trust_remote_code=True
    )
    model.config.problem_type = "regression"

    if unfreeze_last_n is not None:
        # try to locate the encoder block stack
        enc = getattr(model, model.base_model_prefix, None)
        layers = None
        for attr in ["encoder", "transformer", "layers"]:
            enc = getattr(enc, attr, enc)
        for attr in ["layer", "layers", "h"]:
            layers = getattr(enc, attr, None) if enc is not None else None
            if layers is not None:
                break

        if hasattr(layers, "__len__"):
            total = len(layers)
            keep_start = max(0, total - unfreeze_last_n)

            for i, block in enumerate(layers):
                req_grad = i >= keep_start
                for name, p in block.named_parameters(recurse=True):
                    # keep LayerNorm/bias trainable even if frozen
                    if ("norm" in name.lower()) or name.endswith(".bias"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = req_grad

            # head stays trainable
        else:
            # fallback: freeze all except head + norms/bias
            for name, p in model.named_parameters():
                if any(k in name for k in ["classifier", "score", "regression"]):
                    p.requires_grad = True
                elif ("norm" in name.lower()) or name.endswith(".bias"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    # stats
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trn}/{tot} ({trn/tot:.2%})")
    return model


# class WeightedHuberTrainer(Trainer):
#     def __init__(self, *args, target_weights=None, huber_delta=1.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         if target_weights is not None:
#             self.target_weights = torch.tensor(target_weights, dtype=torch.float32)
#         else:
#             self.target_weights = None
#         self.huber_delta = float(huber_delta)

#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")  # (B, K)
#         outputs = model(**inputs)
#         logits = outputs.logits  # (B, K) or (B,1,K)
#         if logits.ndim == 3 and logits.size(1) == 1:
#             logits = logits[:, 0, :]
#         diff = logits - labels
#         abs_diff = diff.abs()
#         delta = self.huber_delta
#         huber = torch.where(abs_diff <= delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
#         if self.target_weights is not None:
#             huber = huber * self.target_weights  # broadcast over K
#         loss = huber.mean()
#         return (loss, outputs) if return_outputs else loss
    


class WeightedHuberTrainer(Trainer):
    def __init__(self, *args, target_weights=None, huber_delta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # keep as plain CPU tensor for now; don't bind device here
        if target_weights is not None:
            tw = torch.tensor(target_weights, dtype=torch.float32)
            # ensure it's 1D [K]
            self.target_weights = tw.view(-1)
        else:
            self.target_weights = None
        self.huber_delta = float(huber_delta)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # (B, K); already moved to device by Trainer
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.ndim == 3 and logits.size(1) == 1:
            logits = logits[:, 0, :]

        # match dtypes just in case
        if labels.dtype != logits.dtype:
            labels = labels.to(logits.dtype)

        diff = logits - labels
        abs_diff = diff.abs()
        d = self.huber_delta
        huber = torch.where(abs_diff <= d, 0.5 * diff**2, d * (abs_diff - 0.5 * d))

        if self.target_weights is not None:
            # move to the same device as tensors participating in the loss
            tw = self.target_weights.to(diff.device)
            # broadcast (B,K) * (K,) safely
            huber = huber * tw.view(1, -1)

        loss = huber.mean()
        return (loss, outputs) if return_outputs else loss



def main():
    set_seeds(seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args = parser.parse_args()
    args = parse_common_args(args)
    if args.embed_type == 'can':
        input_type = INPUT_TYPES_CAN
    elif args.embed_type == 'iso':
        input_type = INPUT_TYPES_ISO

    # ----- Optional HPO control via env var -----
    HPO_TRIALS = int(os.environ.get("HPO_TRIALS", "0"))  # e.g., export HPO_TRIALS=20

    # Extract parameters
    # model_name = args.model
    participant_id = args.participant_id
    n_fold = args.n_fold
    
    unfreeze_last_n = args.unfreeze_last_n
    if unfreeze_last_n in (None, "", "None"):
        # nothing to unfreeze
        print("No layers will be unfrozen.")
        unfreeze_last_n = None
    else:
        unfreeze_last_n = int(unfreeze_last_n)
    print(f"Unfreezing last {unfreeze_last_n} layers")
    num_train_epochs = args.num_train_epochs
    out_dir = args.out_dir
    ds = args.ds
    behavior_embeddings = args.behavior_embeddings or get_descriptors(ds)
    lr = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.per_device_train_batch_size
    model_name_path = args.model
    model_path = model_name_path.split('/')[0]
    model_name = model_name_path.split('/')[1]
    input_type = input_type.get(model_name)
    run_id = os.environ.get("RUN_ID")
    embed_type = args.embed_type
    # layer arg can exist but is unused for text fine-tuning

    # Create output directories
    out_base = Path(BASE_DIR) / f"{out_dir}_finetune_metrics_{run_id}"
    models_base = Path(BASE_DIR) / f"{out_dir}_finetune_models_{run_id}"
    out_base.mkdir(parents=True, exist_ok=True)
    models_base.mkdir(parents=True, exist_ok=True)
    
    out_file = out_base / f"metrics_model-{model_name}_ds-{ds}_runid-{run_id}.csv"
    models_dir = build_models_dir(out_dir, run_id)
    embeds_dir = build_embeds_dir(out_dir, run_id)
    beh_val = (
        json.dumps(behavior_embeddings)   # '["intensity","pleasantness","sweet"]'
        .replace('"', "'")                # -> "['intensity','pleasantness','sweet']"
        if isinstance(behavior_embeddings, (list, tuple))
        else str(behavior_embeddings or "")
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    global_preds: List[np.ndarray] = []
    global_labels: List[np.ndarray] = []
    mse_tracking = []

    # Loop through folds
    for i_fold in range(4,n_fold):
        print(f"Training fold {i_fold}, Model {model_name}")

        # Get fold CIDs
        train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)

        # Split training CIDs -> train/val
        train_cids_train, train_cids_val = train_test_split(train_cids, test_size=0.2, random_state=SEED)

        # Build raw HF datasets (text + prop*)
        train_raw, num_targets = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, train_cids_train, input_type)
        val_raw,   _           = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, train_cids_val,   input_type)
        test_raw,  _           = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, test_cids,        input_type)

        # Tokenize and pack labels
        train_dataset = train_raw.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})
        val_dataset   = val_raw.map(  tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})
        test_dataset  = test_raw.map( tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})

        # Keep only model-relevant columns
        keep_cols = ['input_ids', 'attention_mask', 'labels']
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
        val_dataset   = val_dataset.remove_columns([c for c in val_dataset.column_names if c not in keep_cols])
        test_dataset  = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])

        K = num_targets  # <- not behavior_embeddings

        # quick check on first example
        first_lbl = train_dataset[0]["labels"]
        assert isinstance(first_lbl, list) and len(first_lbl) == K

        # ----- OPTIONAL: Hyperparameter search (Optuna) -----
        best_params = None
        if HPO_TRIALS > 0:
            # model_init so each trial starts fresh
            def model_init():
                m = create_model_and_freeze_layers(model_name_path, num_targets, unfreeze_last_n).to(device)
                return m

            # Light-weight args for search: no checkpoint saving
            search_args = TrainingArguments(
                output_dir=models_base / f"_search_discard_{model_name}_{n_fold}_{num_train_epochs}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_train_epochs,
                metric_for_best_model="eval_corr_mean",  
                evaluation_strategy="epoch",
                save_strategy="epoch",
                report_to=[],  # or ["none"]
                remove_unused_columns=True,
                save_safetensors=False,
                greater_is_better=True,
                load_best_model_at_end=True,
                 warmup_ratio=0.1,
                  lr_scheduler_type="cosine",
                   max_grad_norm=1.0,
                # save_fold_indices=False,
            )

            search_trainer = WeightedHuberTrainer(
                model_init=model_init,
                args=search_args,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                target_weights=None,  # keep fixed during search
                huber_delta=1.0,
            )

            def hp_space_optuna(trial):
                return {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
                 "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
        "huber_delta": trial.suggest_float("huber_delta", 0.5, 2.0),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),

                }

            # Minimize eval_mse_mean (Trainer prefixes eval_*)
            def compute_objective(metrics):
                return metrics["eval_mse_mean"]

            print(f"[HPO] Running {HPO_TRIALS} Optuna trials…")
            best_run = search_trainer.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                n_trials=HPO_TRIALS,
                hp_space=hp_space_optuna,
                compute_objective=compute_objective,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0),
            )
            best_params = best_run.hyperparameters
            print("[HPO] Best:", best_params)

            # Override per-fold hyperparams with the best ones
            lr = float(best_params.get("learning_rate", lr))
            weight_decay = float(best_params.get("weight_decay", weight_decay))
            batch_size = int(best_params.get("per_device_train_batch_size", batch_size))
        # ----- END HPO -----

        # Create and configure model
        model = create_model_and_freeze_layers(model_name_path, num_targets, unfreeze_last_n).to(device)

        # Set up training arguments and output directory
        output_dir = models_base / f"model_{model_name}_{n_fold}_{num_train_epochs}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            weight_decay=weight_decay,
            lr_scheduler_type="linear",
            load_best_model_at_end=True,
            metric_for_best_model="eval_mse_mean",
            greater_is_better=False,
            save_total_limit=5,
            save_safetensors=False,
            remove_unused_columns=True,  # safe with tokenize -> keep only desired cols
        )

        # Set up callbacks
        mse_callback = MSETrackingCallback(val_dataset, mse_tracking, i_fold)
        pred_callback = FinalPredictionCallback(test_dataset, global_preds, global_labels)

        train_Y = np.stack([train_raw[f"prop{i}"] for i in range(num_targets)], axis=1)
        mu = train_Y.mean(axis=0)
        sigma = train_Y.std(axis=0) + 1e-8

        use_weights = True
        weights = (1.0 / (sigma**2 + 1e-8)).tolist() if use_weights else None

        trainer = WeightedHuberTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[mse_callback, pred_callback, EarlyStoppingCallback(early_stopping_patience=5)],
            target_weights=weights,
            huber_delta=1.0,
        )

        # Set trainer reference for callbacks
        mse_callback.trainer = trainer
        pred_callback.trainer = trainer

        # Train model
        trainer.train()

        # Clean up checkpoints (keep only best)
        best_model_dir = trainer.state.best_model_checkpoint
        print(f"Best model dir: {best_model_dir}")
        print(f"All checkpoints in {output_dir}:",glob.glob(os.path.join(output_dir, "checkpoint-*")))
        print("output_dir =", repr(output_dir))
        print("Exists?   ", os.path.exists(output_dir))
        print("Contents: ", os.listdir(output_dir))
        print("Glob:     ", glob.glob(os.path.join(output_dir, "checkpoint-*")))
        pattern = os.path.join(glob.escape(str(output_dir)), "checkpoint-*")
        paths = glob.glob(pattern)
        print("Glob (escaped) ->", paths)
        for path in paths:
            print("path", path)
            if path != best_model_dir:
                print(f"Removing {path}...")
                shutil.rmtree(path)

        discard_root = models_base / f"_search_discard_{model_name}_{n_fold}_{num_train_epochs}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
        try:
        # Safety: only remove if it's inside models_base and name starts with the marker
            discard_root_res = discard_root.resolve()
            models_base_res = models_base.resolve()
            if models_base_res in discard_root_res.parents and discard_root.name.startswith("_search_discard_"):
                if discard_root.exists():
                    shutil.rmtree(discard_root)
                    print(f"[CLEAN] Removed search discard folder: {discard_root}")
            else:
                print(f"[CLEAN] Skip: {discard_root} failed safety checks.")
        except Exception as e:
            print(f"[CLEAN] Could not remove {discard_root}: {e}")

    # Compute and save final metrics
    all_preds = np.concatenate(global_preds, axis=0)
    all_labels = np.concatenate(global_labels, axis=0)
    print(f"Final predictions shape: {all_preds.shape}, labels shape: {all_labels.shape}")
    correlations, mse_errors = compute_targetwise_metrics(all_preds, all_labels)
    p_value_correlation, p_value_mse = permutation_test_metrics(
        all_preds,
        all_labels,
        correlations,
        mse_errors,
        n_permutations=1000,
        use_abs_corr=False,
        seed=SEED,
    )
    # Create final metrics DataFrame
    metrics_rows = []
    for i in range(len(mse_errors)):
        metrics_rows.append({
            "target": get_descriptors(ds)[i],
            "target_id": i,
            "mse": float(mse_errors[i]),
            "correlation": float(correlations[i]),
            "p_value_mse": float(p_value_mse[i]),
            "p_value_correlation": float(p_value_correlation[i])
        })

    metrics_df = pd.DataFrame(metrics_rows).assign(
        model=model_name,
        ds=ds,
        participant_id=participant_id,
        n_fold=n_fold,
        behavior_embeddings=beh_val,
        unfreeze_last_n=unfreeze_last_n,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        run_id=os.environ.get("RUN_ID", "UNKNOWN"),
    )
    write_header = not out_file.exists()
    metrics_df.to_csv(out_file, mode="a", index=False, header=write_header)

    # Save MSE tracking results
    mse_df = pd.DataFrame(mse_tracking).assign(
        model=model_name,
        ds=ds,
        participant_id=participant_id,
        unfreeze_last_n=unfreeze_last_n,
        n_fold=n_fold,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        run_id=os.environ.get("RUN_ID", "UNKNOWN"),
        behavior_embeddings=beh_val,
    )

    mse_file = out_base / f"mse_tracking_model-{model_name}_ds-{ds}_runid-{run_id}.csv"
    write_header = not mse_file.exists()
    mse_df.to_csv(mse_file, mode="a", index=False, header=write_header)

    print("Fine-tuning completed successfully!")


    # per_model_ds_csv = embeds_dir / f"reps_{model_name}_ds-{ds}_runid-{run_id}.csv"

    # for i_fold in range(n_fold):
    #     # print(
    #     #     f"[Extract] model={model_name} fold={i_fold}/{n_fold} "
    #     #     f"epochs={num_train_epochs} subj={participant_id} emb={behavior_embeddings} "
    #     #     f"unfreeze={unfreeze_last_n} ds={ds} lr={lr} bs={batch_size} wd={weight_decay}",
    #     #     flush=True,
    #     # )

    #     train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)
    #     cids = list(train_cids) + list(test_cids)
    #     model_dir = models_dir / (
    #         f"model_{model_name}_{n_fold}_{num_train_epochs}_{participant_id}_"
    #         f"{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
    #     )
    #     ckpt = get_latest_checkpoint(model_dir)
    #     model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

    #     extract_representations(
    #         cids=cids,
    #         participant_id=participant_id,
    #         input_type=input_type,
    #         out_csv=per_model_ds_csv,
    #         tokenizer=tokenizer,
    #         model=model,
    #         model_name=model_name,
    #         n_fold=n_fold,
    #         i_fold=i_fold,
    #         subject=participant_id,
    #         behavior_embeddings=behavior_embeddings,
    #         unfreeze_last_n=unfreeze_last_n,
    #         ds=ds,
    #         token_index=0,
    #         embed_type=embed_type
    #     )
    #     print("extracted fold",i_fold)
    # print(f"All done! Extracted representations saved to: {per_model_ds_csv}", flush=True)

    


if __name__ == "__main__":
    main()
