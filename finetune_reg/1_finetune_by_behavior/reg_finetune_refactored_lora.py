import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import json
from utils.config import BASE_DIR, SEED
from utils.helpers import *  # includes: set_seeds, get_descriptors, build_hf_text_dataset_for_cids,
                             # apply_unfreeze_last_n, ProgressiveUnfreezeCallback, run_permutation_significance
from utils.model_config import MODELS, LAYERS_END, INPUT_TYPES_CAN, INPUT_TYPES_ISO
from utils.arg_parser import create_finetune_parser, parse_common_args
from utils.data_loader import load_fold_cids  # <-- your existing loader
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json, shlex, time
from pathlib import Path
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


def _namespace_to_plain(ns):
    """argparse.Namespace -> plain dict (Path -> str, etc.)."""
    out = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out
# ---- NEW: LoRA / PEFT ----
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


import optuna
from sklearn.model_selection import train_test_split
import torch
import glob
import shutil
from utils.regression import compute_targetwise_metrics, permutation_test_metrics
import matplotlib.pyplot as plt

# --------------------------- ARGPARSE ---------------------------
parser = create_finetune_parser('finetune_by_behavior')

# # --- LoRA flags ---
# parser.add_argument("--use_lora", type=int, default=0, help="Enable LoRA adapters")
# parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
# parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
# parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
# parser.add_argument(
#     "--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"],
#     help="Bias handling in LoRA"
# )
# parser.add_argument(
#     "--lora_target", type=str, default="auto",
#     help="Target module names: 'auto' | comma-separated list (e.g. 'q_proj,k_proj,v_proj,o_proj' or 'query,key,value,dense')"
# )

# --------------------------- PLOTTING ---------------------------
def plot_loss(trainer, test_dataset, out_base, model_name, n_fold, i_fold, ds):
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

    # ---- Loss components: plot each separately (Train vs Val vs Test) ----
    components = ["loss_corr"]
    for comp in components:
        tx, ty = series(comp)
        vx, vy = series(f"eval_{comp}")
        tex, tey = series(f"test_{comp}")

        if not tx and not vx and not tex:
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

# --------------------------- TOKENIZATION & METRICS ---------------------------
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
    """Per-target MSE + mean MSE, and corr_mean."""
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

# --------------------------- CALLBACKS ---------------------------
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
    def __init__(self, test_dataset, global_preds, global_labels, mu, sigma, use_lora=False, base_model_name_or_path=None):
        self.test_dataset = test_dataset
        self.global_preds = global_preds
        self.global_labels = global_labels
        self.mu = mu
        self.sigma = sigma
        self.use_lora = use_lora
        self.base_model_name_or_path = base_model_name_or_path
        self.trainer = None

    def on_train_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        if state.best_model_checkpoint is not None:
            if self.use_lora:
                # Load base model then attach best adapter from checkpoint
                base = AutoModelForSequenceClassification.from_pretrained(
                    self.base_model_name_or_path,
                    num_labels=trainer.model.config.num_labels,
                    trust_remote_code=True
                ).to(trainer.args.device)
                trainer.model = PeftModel.from_pretrained(base, state.best_model_checkpoint).to(trainer.args.device)
            else:
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

class TrainEvalCorrCallback(TrainerCallback):
    def __init__(self, test_dataset, val_dataset, enable: bool = False):
        self.enable = enable
        self.trainer = None
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.enable:
            return control
        # Evaluate on train set
        metrics = self.trainer.evaluate(self.trainer.train_dataset, metric_key_prefix="train")
        self.trainer.log(metrics)
        # Evaluate on test set
        metrics = self.trainer.evaluate(self.test_dataset, metric_key_prefix="test")
        self.trainer.log(metrics)
        # Evaluate on val set (and log only eval_corr_mean explicitly)
        metrics = self.trainer.evaluate(self.val_dataset, metric_key_prefix="eval")
        self.trainer.log({'eval_corr_mean': metrics.get('eval_corr_mean', None)})
        print("metrics:", metrics)
        return control

# --------------------------- LoRA Utilities ---------------------------
def guess_lora_targets(model) -> list:
    """
    Infer sensible target module names from the model's attention layers.
    Covers BERT/Roberta-like and LLaMA-style names.
    """
    names = {n for n, _ in model.named_modules()}
    candidates = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],                     # LLaMA-like
        ["query", "key", "value", "dense"],                           # BERT/Roberta attention
        ["self.query", "self.key", "self.value", "output.dense"],     # HF classic naming
    ]
    for group in candidates:
        if any(any(g in n for n in names) for g in group):
            return group
    attn_proj = sorted({n.split(".")[-1] for n in names if "attn" in n and "proj" in n})
    return attn_proj or ["query", "key", "value", "dense"]

def ensure_classifier_trainable(model):
    for n, p in model.named_parameters():
        if ("classifier" in n) or ("regression" in n) or ("score" in n):
            p.requires_grad = True

# ================== CorrHuberTrainer with discriminative LRs + L2-SP ==================
class CorrHuberTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, huber_delta=1.0, l2sp_lambda: float = 1e-3, ref_params=None, **kwargs):
        """
        alpha: weight for correlation loss
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

    # -------------------- Discriminative LRs + decoupled WD --------------------
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Detect LoRA params by PEFT naming
        def is_lora_param(name):
            return ("lora_" in name) or ("lora_A" in name) or ("lora_B" in name)

        using_lora = any(is_lora_param(n) and p.requires_grad for n, p in self.model.named_parameters())

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

        base_lr = self.args.learning_rate
        opt_groups = []
        if using_lora:
            # Stable setup: adapters + head at base LR
            if head_params:
                opt_groups.append({"params": head_params, "lr": base_lr, "weight_decay": 0.01})
            if enc_params:
                opt_groups.append({"params": enc_params, "lr": base_lr, "weight_decay": 0.01})
            if nb_params:
                opt_groups.append({"params": nb_params, "lr": base_lr, "weight_decay": 0.0})
        else:
            if head_params:
                opt_groups.append({"params": head_params, "lr": max(1e-5, base_lr * 50.0), "weight_decay": 0.05})
            if enc_params:
                opt_groups.append({"params": enc_params, "lr": base_lr, "weight_decay": 0.01})
            if nb_params:
                opt_groups.append({"params": nb_params, "lr": base_lr * 3.0, "weight_decay": 0.0})

        self.optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

    # -------------------- original loss + L2-SP regularizer --------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        out = model(**inputs)
        preds = out.logits
        if preds.ndim == 3 and preds.size(1) == 1:
            preds = preds[:, 0, :]
        labels = labels.to(preds.dtype)

        alpha = self._alpha_default

        d = self.huber_delta
        diff = preds - labels
        huber = torch.where(diff.abs() <= d, 0.5 * diff**2, d * (diff.abs() - 0.5 * d)).mean()

        x = preds - preds.mean(dim=0, keepdim=True)
        y = labels - labels.mean(dim=0, keepdim=True)
        denom = (x.norm(dim=0) * y.norm(dim=0) + 1e-8)
        r = (x * y).sum(dim=0) / denom
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

        # ---- log components (train only here) ----
        if model.training:
            self.log({"loss_corr": corr_loss.item()})
        return (total, out) if return_outputs else total

# --------------------------- MAIN ---------------------------
def main():
    set_seeds(seed=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args = parser.parse_args()
    args = parse_common_args(args)
    HPO_TRIALS = args.optuna_trials if getattr(args, "use_optuna", False) else 0

    # Determine input type mapping
    if args.embed_type == 'can':
        input_type_map = INPUT_TYPES_CAN
    elif args.embed_type == 'iso':
        input_type_map = INPUT_TYPES_ISO
    else:
        raise ValueError(f"Unknown embed_type: {args.embed_type}")

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

    print(f"Unfreeze mode = {unfreeze_last_n}")
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
    input_type = input_type_map.get(model_name)
    run_id = args.run_id
    embed_type = args.embed_type
    alpha = 1.0
    lr_scheduler_type = "cosine"  # sensible default
    i_fold = args.i_fold
    roi = None
    tr = None  # unused  # 'beh' or 'fmri'
    finetune_by = args.finetune_by
    if args.finetune_by == 'fmri':
        roi = args.roi
        tr = args.tr

    # LoRA toggle
    use_lora = bool(getattr(args, "use_lora", 0))

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
    train_cids_train, train_cids_val = train_test_split(train_cids, test_size=0.33, random_state=SEED)
    train_raw, num_targets = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, train_cids_train, input_type, finetune_by=finetune_by, roi=roi, tr=tr)
    val_raw,   _           = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, train_cids_val,   input_type, finetune_by=finetune_by, roi=roi, tr=tr)
    test_raw,  _           = build_hf_text_dataset_for_cids(ds, participant_id, behavior_embeddings, test_cids,        input_type, finetune_by=finetune_by, roi=roi, tr=tr)

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
    EPOCH_MIN = 2
    EPOCH_MAX = 20

    first_lbl = train_dataset[0]["labels"]
    print(f"First train label: {len(first_lbl)}, {K}, {isinstance(first_lbl, list)} targets")
    assert len(first_lbl) == K

    callbacks_final = []
    callbacks_search = []

    # Progressive unfreezing callback in search (only used if unfreeze_last_n == "adaptive" and LoRA is off)
    prog_cb_search = None
    if (unfreeze_last_n == "adaptive") and (not use_lora):
        schedule = [(0, 0), (1, 2), (3, 6), (5, "all")]
        prog_cb_search = ProgressiveUnfreezeCallback(schedule)
        callbacks_search.append(prog_cb_search)

    # ================== HPO section (provides ref_params to trainer if needed) ==================
    best_params = None
    # defaults in case HPO disabled
    per_device_train_batch_size = batch_size
    warmup_ratio = 0.1
    gradient_accumulation_steps = 1

    if HPO_TRIALS > 0:
        def model_init():
            m = AutoModelForSequenceClassification.from_pretrained(
                model_name_path, num_labels=num_targets, trust_remote_code=True
            )
            m.config.problem_type = "regression"

            if not use_lora:
                if unfreeze_last_n != "adaptive":
                    last_n = 1_000_000 if unfreeze_last_n == "all" else int(unfreeze_last_n)
                    apply_unfreeze_last_n(m, last_n)
                # snapshot reference weights for L2-SP
                ref = {n: p.detach().clone()
                       for n, p in m.named_parameters()
                       if not (("classifier" in n) or ("regression" in n) or ("score" in n))}
                m._ref_params = ref
            else:
                # Wrap with LoRA for search too if requested
                if args.lora_target.strip().lower() == "auto":
                    target_modules = guess_lora_targets(m)
                else:
                    target_modules = [t.strip() for t in args.lora_target.split(",") if t.strip()]
                lora_cfg = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias=args.lora_bias,
                    target_modules=target_modules,
                )
                m = get_peft_model(m, lora_cfg)
                ensure_classifier_trainable(m)
                # no L2-SP snapshot when using adapters
                m._ref_params = {}
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
            l2sp_lambda=0,   # off in search
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
                "num_train_epochs": trial.suggest_int("num_train_epochs", EPOCH_MIN, EPOCH_MAX),
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
        num_train_epochs = int(best_params.get("num_train_epochs", num_train_epochs))
    else:
        # No HPO: use provided args
        per_device_train_batch_size = batch_size
        warmup_ratio = 0.1
        gradient_accumulation_steps = 1

    # -------------------- Build base model --------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_path, num_labels=num_targets, trust_remote_code=True
    ).to(device)
    model.config.problem_type = "regression"

    # ----- Optional LoRA -----
    if use_lora:
        if args.lora_target.strip().lower() == "auto":
            target_modules = guess_lora_targets(model)
        else:
            target_modules = [t.strip() for t in args.lora_target.split(",") if t.strip()]

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        ensure_classifier_trainable(model)
        print(f"[LoRA] Enabled with targets={target_modules}, r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    else:
        # Your existing unfreezing policies only apply when LoRA is OFF
        if unfreeze_last_n == "adaptive":
            schedule = [(0, 0), (1, 2), (3, 6), (5, "all")]
            prog_cb_final = ProgressiveUnfreezeCallback(schedule)
            callbacks_final.append(prog_cb_final)
        else:
            last_n = 1_000_000 if unfreeze_last_n == "all" else int(unfreeze_last_n)
            apply_unfreeze_last_n(model, last_n)

    # ===== snapshot reference weights for L2-SP on the final trainer =====
    if not use_lora:
        ref_params = {n: p.detach().clone()
                      for n, p in model.named_parameters()
                      if not (("classifier" in n) or ("regression" in n) or ("score" in n))}
    else:
        ref_params = {}  # L2-SP off when LoRA is active (adapters are new)

    output_dir = models_base / f"model_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
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
    pred_callback = FinalPredictionCallback(test_dataset, global_preds, global_labels, mu=None, sigma=None,
                                            use_lora=use_lora, base_model_name_or_path=model_name_path)
    # Always log train/test/val metrics at epoch end
    train_corr_cb = TrainEvalCorrCallback(test_dataset, val_dataset, enable=True)
    callbacks_final.append(train_corr_cb)
    callbacks_final.append(mse_callback)
    callbacks_final.append(pred_callback)
    # Optionally: callbacks_final.append(EarlyStoppingCallback(early_stopping_patience=10))

    # NOTE: choose CorrHuberTrainer (AdamW) — uses discriminative groups; LoRA handled inside
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
        l2sp_lambda=0 if use_lora else 0,   # off by default; enable if you want
        ref_params=ref_params,
    )

    for cb in callbacks_final:
        cb.trainer = trainer

    trainer.train()
    plot_loss(trainer, test_dataset, out_base, model_name, n_fold, i_fold, ds)

    best_model_dir = trainer.state.best_model_checkpoint
    print(f"Best model dir: {best_model_dir}")
    print(f"All checkpoints in {output_dir}:", glob.glob(os.path.join(output_dir, "checkpoint-*")))
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

    # Cleanup HPO discard dirs
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


    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    args_path = out_base / f"cli_args_{stamp}.json"
    with open(args_path, "w") as f:
        json.dump(
            {
                "argv": sys.argv,                     # exact invocation
                "args": _namespace_to_plain(args),    # normalized dict of parsed args
            },
            f,
            indent=2,
        )
    print(f"[ARGS] Saved parsed args -> {args_path}")

if __name__ == "__main__":
    main()
