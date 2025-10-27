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
from lion_pytorch import Lion

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
import matplotlib.pyplot as plt

parser = create_finetune_parser('finetune_by_behavior')
def plot_loss(trainer,test_dataset, out_base, model_name, n_fold, i_fold, ds):
    history = trainer.state.log_history

    # Helpers to collect (epoch, value) pairs
    def series(key):
        xs, ys = [], []
        for rec in history:
            if key in rec and "epoch" in rec:
                xs.append(rec["epoch"])
                ys.append(rec[key])
        return xs, ys

    # ---- Correlation plot (validation + optional train) ----
    e_x, e_y = series("eval_corr_mean")
    t_x, t_y = series("train_corr_mean")  # present only if LOG_TRAIN_CORR=1
    tex , tey = series("test_corr_mean")  # present only if LOG_TRAIN_CORR=1

    plt.figure()
    if e_x:
        plt.plot(e_x, e_y, label="Val corr (mean)")
    if t_x:
        plt.plot(t_x, t_y, label="Train corr (mean)")
    if tex:
        plt.plot(tex, tey, label="Test corr (mean)")
    plt.xlabel("Epoch"); plt.ylabel("Pearson r (mean)")
    plt.title(f"Correlation vs Epoch — {model_name} (fold {i_fold})")
    plt.grid(True, linewidth=0.3, alpha=0.5); plt.legend()
    corr_path = Path(out_base) / f"corr_curve_{model_name}_{n_fold}_{i_fold}_{ds}.png"
    plt.savefig(corr_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[PLOT] Saved correlation curve to: {corr_path}")

    # ---- Loss components: plot each separately (Train vs Val) ----
    components = ["loss_corr"]
    for comp in components:
        tx, ty = series(comp)
        vx, vy = series(f"eval_{comp}")
        tex, tey = series(f"test_{comp}")

        if not tx and not vx:
            continue

        plt.figure()
        if tx:
            plt.plot(tx, ty, label=f"Train {comp}")
        if vx:
            plt.plot(vx, vy, label=f"Val {comp}")
        if tex:
            plt.plot(tex, tey, label=f"Test {comp}")
        plt.xlabel("Epoch"); plt.ylabel(comp.replace("_", " ").title())
        plt.title(f"{comp.replace('_',' ').title()} vs Epoch — {model_name} (fold {i_fold})")
        plt.grid(True, linewidth=0.3, alpha=0.5); plt.legend()
        path = Path(out_base) / f"{comp}_curve_{model_name}_{n_fold}_{i_fold}_{ds}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[PLOT] Saved {comp} curve to: {path}")

def tokenize(batch, tokenizer, input_type="smiles"):
    # Don't pad here; let the collator pad
    enc = tokenizer(batch[input_type], truncation=True)
    # collect prop* columns in batch order and build (B,K) float32 list-of-lists
    label_cols = sorted([c for c in batch.keys() if c.startswith("prop")],
                    key=lambda x: int(x.replace("prop","")))
    labels = np.stack([batch[c] for c in label_cols], axis=1).astype("float32").tolist()
    enc["labels"] = labels
    return enc

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Per-target MSE + mean MSE."""
    if isinstance(eval_pred, tuple):
        predictions, labels = eval_pred
    else:
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

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

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_mse_mean" in metrics:
            self.mse_tracking.append({
                "fold": self.fold_id,
                "epoch": int(state.epoch or 0),
                "mean_mse": float(metrics["eval_mse_mean"]),
            })
        return control

class FinalPredictionCallback(TrainerCallback):
    def __init__(self, test_dataset, global_preds, global_labels, mu, sigma):
        self.test_dataset = test_dataset
        self.global_preds = global_preds
        self.global_labels = global_labels
        self.mu = mu
        self.sigma = sigma

    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        if state.best_model_checkpoint is not None:
            trainer.model = AutoModelForSequenceClassification.from_pretrained(
                state.best_model_checkpoint,
                num_labels=trainer.model.config.num_labels,
                trust_remote_code=True
            ).to(trainer.args.device)
        res = trainer.predict(self.test_dataset)
        preds = res.predictions
        labels = res.label_ids
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = preds[:, 0, :]
        if self.mu is not None and self.sigma is not None:
            preds = preds * self.sigma + self.mu
            labels = labels * self.sigma + self.mu
        self.global_preds.append(preds)
        self.global_labels.append(labels)
        return control

def add_std_labels(ds, mu, sigma, num_targets):
    lbls = np.stack([ds[f"prop{i}"] for i in range(num_targets)], axis=1)
    z = (lbls - mu) / sigma
    for i in range(num_targets):
        ds = ds.remove_columns([f"prop{i}"]).add_column(f"prop{i}", z[:, i].tolist())
    return ds


# ================== CHANGED: CorrHuberTrainer with discriminative LRs + L2-SP ==================
class CorrHuberTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, huber_delta=1.0, l2sp_lambda: float = 1e-3, ref_params=None, **kwargs):
        """
        alpha: weight for correlation loss (taken from args.ALPHA if present)
        huber_delta: Huber delta for regression
        l2sp_lambda: strength of L2-SP regularizer (0 disables)
        ref_params: dict[name -> tensor] of reference (pretrained) params for L2-SP
        """
        super().__init__(*args, **kwargs)
        self._alpha_default = alpha
        self.huber_delta = float(huber_delta)
        self.l2sp_lambda = float(l2sp_lambda)
        # If ref_params not provided, try to pull from model (used by HPO model_init)
        if ref_params is None and hasattr(self.model, "_ref_params"):
            self.ref_params = self.model._ref_params
        else:
            self.ref_params = ref_params or {}

    # -------------------- (2) Discriminative LRs + decoupled WD --------------------
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        head_params, enc_params, nb_params = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ("classifier" in n) or ("regression" in n) or ("score" in n):
                head_params.append(p)
            elif ("norm" in n.lower()) or n.endswith(".bias"):
                nb_params.append(p)
            else:
                enc_params.append(p)

        base_lr = self.args.learning_rate  # backbone LR from TrainingArguments/CLI
        opt_groups = []
        if head_params:
            opt_groups.append({"params": head_params, "lr": max(1e-5, base_lr * 50.0), "weight_decay": 0.05})
        if enc_params:
            opt_groups.append({"params": enc_params,  "lr": base_lr,                   "weight_decay": 0.01})
        if nb_params:
            opt_groups.append({"params": nb_params,   "lr": base_lr * 3.0,             "weight_decay": 0.0})

        # Use AdamW (decoupled WD). Lion left intact elsewhere if you switch trainers.
        self.optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

    # -------------------- original loss + (3) L2-SP regularizer --------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        out = model(**inputs)
        preds = out.logits
        if preds.ndim == 3 and preds.size(1) == 1:
            preds = preds[:, 0, :]
        labels = labels.to(preds.dtype)

        alpha =self._alpha_default

        d = self.huber_delta
        diff = preds - labels
        huber = torch.where(diff.abs() <= d, 0.5 * diff**2, d * (diff.abs() - 0.5 * d)).mean()

        x = preds - preds.mean(dim=0, keepdim=True)
        y = labels - labels.mean(dim=0, keepdim=True)
        denom = (x.norm(dim=0) * y.norm(dim=0) + 1e-8)
        r = (x * y).sum(dim=0) / denom
        # r= pearsonr(preds, labels)
        corr_loss = (1.0 - r).mean()

        total = (1.0 - alpha) * huber + alpha * corr_loss

        # ----- L2-SP: keep *non-head* trainable params near their reference values -----
        if self.l2sp_lambda > 0.0 and self.ref_params:
            reg = None
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if ("classifier" in n) or ("regression" in n) or ("score" in n):
                    continue  # exclude heads
                ref = self.ref_params.get(n, None)
                if ref is None:
                    continue
                dtheta = p - ref.to(p.device)
                term = (dtheta * dtheta).sum()
                reg = term if reg is None else reg + term
            if reg is not None:
                denom = sum(v.numel() for v in self.ref_params.values()) + 1e-8
                total = total + self.l2sp_lambda * reg / denom
         # ---- log the components ----
        # ---- log the components (train vs eval) ----
        prefix = "" if model.training else "val_"
        if model.training:
            self.log({
            #     f"{prefix}loss_huber": huber.item(),
                f"{prefix}loss_corr":  corr_loss.item(),
            #     f"{prefix}loss_reg":   reg.item() if (self.l2sp_lambda > 0.0 and 'reg' in locals() and reg is not None) else 0.0,
            #     f"{prefix}loss_total": total.item(),
            })
        return (total, out) if return_outputs else total
# =================================================================================================
class TrainEvalCorrCallback(TrainerCallback):
    def __init__(self,test_dataset,val_dataset, enable: bool = False):
        self.enable = enable
        self.trainer = None
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.enable:
            return control
        # Evaluate on training set to get train_corr_mean (and train_loss, etc.)
        metrics = self.trainer.evaluate(self.trainer.train_dataset, metric_key_prefix="train")
        # Also push into log_history for plotting
        self.trainer.log(metrics)
        metrics = self.trainer.evaluate(self.test_dataset, metric_key_prefix="test")
        self.trainer.log(metrics)
        metrics = self.trainer.evaluate(self.val_dataset, metric_key_prefix="eval")
        #log only eval_corr_mean
        self.trainer.log({'eval_corr_mean':metrics['eval_corr_mean']})
        print("metrics:", metrics)
        return control

# 

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
    HPO_TRIALS = int(os.environ.get("HPO_TRIALS", "0"))

    # Extract parameters
    participant_id = args.participant_id
    n_fold = args.n_fold
    unfreeze_last_n = args.unfreeze_last_n  # 'adaptive' | 'all' | int or numeric-as-string
    if isinstance(unfreeze_last_n, str):
        s = unfreeze_last_n.strip().lower()
        if s in {"", "none", "all"}:
            unfreeze_last_n = "all"
        elif s in {"adaptive"}:
            unfreeze_last_n = "adaptive"
        else:
            unfreeze_last_n = int(s)
    
    print(f"Unfreeze mode = {unfreeze_last_n}")  # 'adaptive' | 'all' | int
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
    run_id = args.run_id
    embed_type = args.embed_type
    alpha = 1
    lr_scheduler_type = "cosine"  # sensible default
    i_fold = args.i_fold
    roi = None
    tr = None  # unused  # 'beh' or 'fmri'
    finetune_by=args.finetune_by
    if args.finetune_by=='fmri':
        roi = args.roi
        tr=args.tr

    # Create output directories
    out_base = Path(BASE_DIR) / f"{out_dir}_finetune_metrics_{run_id}"
    models_base = Path(BASE_DIR) / f"{out_dir}_finetune_models_{run_id}"
    out_base.mkdir(parents=True, exist_ok=True)
    models_base.mkdir(parents=True, exist_ok=True)
    
    
    beh_val = (
        json.dumps(behavior_embeddings).replace('"', "'")
        if isinstance(behavior_embeddings, (list, tuple))
        else str(behavior_embeddings or "")
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    global_preds: List[np.ndarray] = []
    global_labels: List[np.ndarray] = []
    mse_tracking = []

    print(f"Training fold {i_fold}, Model {model_name}")
    train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)
    train_cids_train, train_cids_val = train_test_split(train_cids, test_size=(1/(n_fold-1)), random_state=SEED)
    train_raw, num_targets = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, train_cids_train, input_type,finetune_by=finetune_by,roi=roi,tr=tr)
    val_raw,   _           = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, train_cids_val,   input_type,finetune_by=finetune_by,roi=roi,tr=tr)
    test_raw,  _           = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, test_cids,        input_type,finetune_by=finetune_by,roi=roi,tr=tr)
    train_Y = np.stack([train_raw[f"prop{i}"] for i in range(num_targets)], axis=1)
    # mu = train_Y.mean(axis=0)
    # sigma = train_Y.std(axis=0) + 1e-8
    mu = None
    sigma = None
    # train_raw = add_std_labels(train_raw, mu, sigma, num_targets)
    # val_raw   = add_std_labels(val_raw,   mu, sigma, num_targets)
    # test_raw  = add_std_labels(test_raw,  mu, sigma, num_targets)

    train_dataset = train_raw.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})
    val_dataset   = val_raw.map(  tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})
    test_dataset  = test_raw.map( tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer, "input_type": input_type})

    keep_cols = ['input_ids', 'attention_mask', 'labels']
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
    val_dataset   = val_dataset.remove_columns([c for c in val_dataset.column_names if c not in keep_cols])
    test_dataset  = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])

    train_dataset = train_dataset.with_format("torch")
    val_dataset   = val_dataset.with_format("torch")
    test_dataset  = test_dataset.with_format("torch")
    K = num_targets
    EPOCH_MIN = 3
    EPOCH_MAX = 40

    first_lbl = train_dataset[0]["labels"]
    print(f"First train label: {len(first_lbl)}, {K}, {isinstance(first_lbl, list)} targets")
    assert  len(first_lbl) == K
    callbacks_final = []
    callbacks_search = []

    # Progressive unfreezing callback in search (unchanged), if used
    prog_cb_search = None
    if unfreeze_last_n == "adaptive":
        schedule = [(0, 0), (1, 2), (3, 6), (5, "all")]
        prog_cb_search = ProgressiveUnfreezeCallback(schedule)
        callbacks_search.append(prog_cb_search)

    # ================== HPO section (CHANGED minimally to provide ref_params to trainer) ==================
    best_params = None
    if HPO_TRIALS > 0:
        def model_init():
            m = AutoModelForSequenceClassification.from_pretrained(
                model_name_path, num_labels=num_targets, trust_remote_code=True
            )
            m.config.problem_type = "regression"
            if unfreeze_last_n != "adaptive":
                last_n = 1_000_000 if unfreeze_last_n == "all" else int(unfreeze_last_n)
                apply_unfreeze_last_n(m, last_n)
            # snapshot reference weights for L2-SP
            ref = {n: p.detach().clone()
                   for n, p in m.named_parameters()
                   if not (("classifier" in n) or ("regression" in n) or ("score" in n))}
            m._ref_params = ref
            return m

        search_args = TrainingArguments(
            output_dir=models_base / f"_search_discard_{model_name}_{n_fold}_{num_train_epochs}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}_{tr}_{roi}",
            metric_for_best_model="eval_corr_mean",
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_strategy="no",
            report_to=[],
            remove_unused_columns=True,
            save_safetensors=True,
            greater_is_better=True,
            max_grad_norm=1.0
        )

        search_trainer = CorrHuberTrainer(
            model_init=model_init,
            args=search_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks_search,
            huber_delta=1.0,
            # L2-SP on during search; trainer will pull ref from model._ref_params
            l2sp_lambda=0,
            ref_params=None,
        )
        if prog_cb_search is not None:
            prog_cb_search.trainer = search_trainer

        def hp_space_optuna(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
                "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
                "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
                "num_train_epochs": trial.suggest_int("num_train_epochs", EPOCH_MIN,EPOCH_MAX),
                # "alpha": trial.suggest_float("alpha", 0.0, 0.5),
                # You can also expose l2sp_lambda if desired:
                # "l2sp_lambda": trial.suggest_float("l2sp_lambda", 1e-4, 5e-3, log=True),
            }

        def compute_objective(metrics):
            return metrics["eval_corr_mean"]
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

        lr = float(best_params.get("learning_rate", lr))
        weight_decay = float(best_params.get("weight_decay", weight_decay))
        per_device_train_batch_size = int(best_params.get("per_device_train_batch_size", batch_size))
        warmup_ratio = float(best_params.get("warmup_ratio", 0.1))
        lr_scheduler_type = str(best_params.get("lr_scheduler_type", lr_scheduler_type))
        gradient_accumulation_steps = int(best_params.get("gradient_accumulation_steps", 1))
        # alpha = float(best_params.get("alpha", 0.0))
    
        num_train_epochs = int(best_params.get("num_train_epochs", num_train_epochs))
        
    # ========================================================================================================

    # Create and configure model respecting the mode (unchanged behavior)
    if unfreeze_last_n == "adaptive":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_path, num_labels=num_targets, trust_remote_code=True
        ).to(device)
        model.config.problem_type = "regression"
        schedule = [(0, 0), (1, 2), (3, 6), (5, "all")]
        prog_cb_final = ProgressiveUnfreezeCallback(schedule)
        callbacks_final.append(prog_cb_final)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_path, num_labels=num_targets, trust_remote_code=True
        ).to(device)
        model.config.problem_type = "regression"
        last_n = 1_000_000 if unfreeze_last_n == "all" else int(unfreeze_last_n)
        apply_unfreeze_last_n(model, last_n)

    # ===== snapshot reference weights for L2-SP on the final trainer =====
    ref_params = {n: p.detach().clone()
                  for n, p in model.named_parameters()
                  if not (("classifier" in n) or ("regression" in n) or ("score" in n))}

    output_dir = models_base / f"model_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",        # <— add this
        learning_rate=lr,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_corr_mean",
        greater_is_better=True,
        save_total_limit=1,
        save_safetensors=False,
        remove_unused_columns=True,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        gradient_accumulation_steps=gradient_accumulation_steps,
       
        
    )

    mse_callback = MSETrackingCallback(val_dataset, mse_tracking, i_fold)
    pred_callback = FinalPredictionCallback(test_dataset, global_preds, global_labels, mu, sigma)
    # log_train_corr = os.environ.get("LOG_TRAIN_CORR", "0") == "1"
    log_train_corr = "1"
    train_corr_cb = TrainEvalCorrCallback(test_dataset,val_dataset,enable=log_train_corr)
    callbacks_final.append(train_corr_cb)
    # es_callback = EarlyStoppingCallback(early_stopping_patience=10)
    callbacks_final.append(mse_callback)
    callbacks_final.append(pred_callback)
    # callbacks_final.append(es_callback)

    # NOTE: choose CorrHuberTrainer (AdamW) or LionTrainer (Lion) — both now use discriminative groups
    trainer = CorrHuberTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks_final,
        alpha=alpha,  
        huber_delta=1.0,
        l2sp_lambda=0,   # strength of L2-SP; sweep 5e-4 ~ 5e-3 if needed
        ref_params=ref_params,
    )
    
    for cb in callbacks_final:
        cb.trainer = trainer

    trainer.train()
    plot_loss(trainer,test_dataset, out_base, model_name, n_fold, i_fold, ds)

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

    try:
        pattern = models_base / f"_search_discard_{model_name}_{n_fold}_*_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
        for cand in glob.glob(str(pattern)):
            try:
                shutil.rmtree(cand)
                print(f"[CLEAN] Removed search discard folder: {cand}")
            except Exception as e:
                print(f"[CLEAN] Could not remove {cand}: {e}")
    except Exception as e:
        print(f"[CLEAN] Error during cleanup: {e}")

    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
