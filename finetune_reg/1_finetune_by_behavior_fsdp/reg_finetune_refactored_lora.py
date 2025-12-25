# ======= Small-data deterministic fine-tuning (with optional grid search) =======

import os
# ---- Determinism-friendly env before importing torch/tokenizers ----
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # deterministic matmul
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ---------- persist best params and full grid summary ----------
from datetime import datetime as _dt
import sys
import os as _os
current_dir = _os.path.dirname(_os.path.abspath(__file__))
parent_dir = _os.path.dirname(_os.path.dirname(current_dir))
sys.path.append(parent_dir)
from torch.nn import functional as F    
import random
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
from lion_pytorch import Lion  # optional; not used by default

import pandas as pd
import numpy as np
from datasets import Dataset as HFDataset
from scipy.stats import pearsonr
from datetime import datetime as _dt
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
torch.set_float32_matmul_precision("high")

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

# --------------------------- Determinism helper ---------------------------
def set_full_determinism(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PyTorch >= 1.8
    # try:
    #     torch.use_deterministic_algorithms(True, warn_only=True)
    # except Exception as e:
    #     print(f"[WARN] use_deterministic_algorithms not fully enabled: {e}")

# --------------------------- ARGPARSE ---------------------------
parser = create_finetune_parser('finetune_by_behavior')

# deterministic small grid instead of Optuna
parser.add_argument("--use_grid_search", type=int, default=0,
                    help="Deterministic small Cartesian grid instead of Optuna")

# --------------------------- PLOTTING ---------------------------
def plot_loss(trainer, test_dataset, out_base, model_name, n_fold,num_train_epochs, i_fold, ds,participant_id,beh_val,unfreeze_last_n,stamp):
    history = trainer.state.log_history

    def series(key):
        xs, ys = [], []
        for rec in history:
            if key in rec and "epoch" in rec:
                xs.append(rec["epoch"])
                ys.append(rec[key])
        return xs, ys

    e_x, e_y = series("eval_corr_mean")
    t_x, t_y = series("train_corr_mean")
    tex , tey = series("test_corr_mean")

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
    corr_path = Path(out_base) / f"corr_curve_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}_{stamp}.png"
    plt.savefig(corr_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[PLOT] Saved correlation curve to: {corr_path}")

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
        path = Path(out_base) / f"{comp}_curve_{model_name}_{n_fold}_{i_fold}_{ds}_{stamp}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[PLOT] Saved {comp} curve to: {path}")

# --------------------------- TOKENIZATION & METRICS ---------------------------
def tokenize(batch, tokenizer, input_type="smiles"):
    enc = tokenizer(batch[input_type], truncation=True)
    label_cols = sorted([c for c in batch.keys() if c.startswith("prop")],
                        key=lambda x: int(x.replace("prop","")))
    labels = np.stack([batch[c] for c in label_cols], axis=1).astype("float32").tolist()
    enc["labels"] = labels
    return enc

def compute_metrics(eval_pred) -> Dict[str, float]:
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
        if (self.mu is not None) and (self.sigma is not None):
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
        metrics = self.trainer.evaluate(self.trainer.train_dataset, metric_key_prefix="train")
        self.trainer.log(metrics)
        metrics = self.trainer.evaluate(self.test_dataset, metric_key_prefix="test")
        self.trainer.log(metrics)
        metrics = self.trainer.evaluate(self.val_dataset, metric_key_prefix="eval")
        self.trainer.log({'eval_corr_mean': metrics.get('eval_corr_mean', None)})
        print("metrics:", metrics)
        return control

# --------------------------- LoRA Utilities ---------------------------
def guess_lora_targets(model) -> list:
    names = {n for n, _ in model.named_modules()}
    candidates = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],                     # LLaMA-like
        ["query", "key", "value"],                           # BERT/Roberta attention
        ["self.query", "self.key", "self.value"],     # HF classic naming
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
def print_lora_summary(model, target_modules):
    print("[LoRA] Inspecting target modules:")
    for t in target_modules:
        print("   target:", t)
    matched = []
    for name, module in model.named_modules():
        if any(name.endswith("." + t) or name == t for t in target_modules):
            matched.append(name)
    if not matched:
        print("[LoRA][WARN] No modules matched these targets! LoRA will be NO-OP.")
    else:
        print("[LoRA] Will attach to the following modules (examples):")
        for n in matched[:20]:
            print("   ", n)
        if len(matched) > 20:
            print(f"   ... and {len(matched)-20} more")
# ================== CorrHuberTrainer with discriminative LRs + L2-SP ==================
class CorrHuberTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, huber_delta=1.0, l2sp_lambda: float = 1e-3, ref_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha_default = alpha
        self.huber_delta = float(huber_delta)
        self.l2sp_lambda = float(l2sp_lambda)
        if ref_params is None and hasattr(self.model, "_ref_params"):
            self.ref_params = self.model._ref_params
        else:
            self.ref_params = ref_params or {}

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

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

        if self.l2sp_lambda > 0.0 and self.ref_params:
            reg = None
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if ("classifier" in n) or ("regression" in n) or ("score" in n):
                    continue
                ref = self.ref_params.get(n, None)
                if ref is None:
                    continue
                dtheta = p - ref.to(p.device)
                term = (dtheta * dtheta).sum()
                reg = term if reg is None else reg + term
            if reg is not None:
                denom = sum(v.numel() for v in self.ref_params.values()) + 1e-8
                total = total + self.l2sp_lambda * reg / denom

        if model.training:
            self.log({"loss_corr": corr_loss.item()})
        return (total, out) if return_outputs else total

class MSETrainer(Trainer):
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
        # 1. Extract labels
        labels = inputs.pop("labels")

        # 2. Forward pass
        out = model(**inputs)
        preds = out.logits

        # Handle shape [batch, 1, dim] -> [batch, dim]
        if preds.ndim == 3 and preds.size(1) == 1:
            preds = preds[:, 0, :]

        # Make sure dtypes match
        labels = labels.to(preds.dtype)

        # 3. PURE MSE LOSS
        mse_loss = F.mse_loss(preds, labels)

        # 4. Logging (no correlation anymore)
        prefix = "" if model.training else "val_"
        if model.training:
            self.log({
                f"{prefix}loss_mse": mse_loss.item(),
            })

        # 5. Return in HF format
        return (mse_loss, out) if return_outputs else mse_loss
# --------------------------- Grid search candidates (EXTENDED WITH LoRA) ---------------------------
from itertools import product
def grid_candidates(using_lora=False):
    """
    Returns dicts containing Trainer args AND, if using_lora=True, LoRA hyperparams.
    Keeping the grid small to preserve determinism and runtime.
    """
    if using_lora:
        lrs = [2e-5, 3e-5, 4e-5, 1e-6,2e-6]
        wd  = [0.0, 0.01]
        bsz = [16,32]
        warm = [0.0, 0.1]
        sched = ["cosine","linear"]
        gas = [1,2]
        epochs = [20]
        # NEW: LoRA grid
        lora_r = [4, 8, 16,32,64,128,256]
        lora_alpha = [8, 16, 32]
        lora_dropout = [ 0]       # keep 0.0 if you want strict determinism
        lora_bias = ["none"]             # stable default
        lora_target = ["auto"]           # or explicit comma list like "q_proj,k_proj,v_proj,o_proj"
        for lr, w, bs, wa, sc, ga, ep, r, a, dr, b, tgt in product(
            lrs, wd, bsz, warm, sched, gas, epochs, lora_r, lora_alpha, lora_dropout, lora_bias, lora_target
        ):
            print(f"Grid candidate: lr={lr}, wd={w}, bs={bs}, warm={wa}, sched={sc}, ga={ga}, ep={ep}, LoRA r={r}, alpha={a}, dropout={dr}, bias={b}, target={tgt}")  
            yield dict(
                learning_rate=lr, weight_decay=w,
                per_device_train_batch_size=bs, warmup_ratio=wa,
                lr_scheduler_type=sc, gradient_accumulation_steps=ga,
                num_train_epochs=ep,
                lora_r=r, lora_alpha=a, lora_dropout=dr, lora_bias=b, lora_target=tgt
            )
    else:
        lrs = [3e-4,1e-5,2e-5,3e-5,3e-6]
        wd  = [0.0]
        bsz = [16,32]
        warm = [0.0, 0.1]
        sched = ["cosine","linear"]
        gas = [1]
        epochs = [40]
        for lr, w, bs, wa, sc, ga, ep in product(lrs, wd, bsz, warm, sched, gas, epochs):
            print(f"Grid candidate: lr={lr}, wd={w}, bs={bs}, warm={wa}, sched={sc}, ga={ga}, ep={ep}")
            yield dict(learning_rate=lr, weight_decay=w,
                       per_device_train_batch_size=bs, warmup_ratio=wa,
                       lr_scheduler_type=sc, gradient_accumulation_steps=ga,
                       num_train_epochs=ep)

# --------------------------- MAIN ---------------------------
def main():
    # Your helper seed (kept) + strict determinism
    set_seeds(seed=SEED)
    set_full_determinism(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args = parser.parse_args()
    args = parse_common_args(args)
    HPO_TRIALS = args.optuna_trials if getattr(args, "use_optuna", False) else 0
    use_grid = bool(getattr(args, "use_grid_search", 0))

    # Determine input type mapping
    if args.embed_type == 'can':
        input_type_map = INPUT_TYPES_CAN
    elif args.embed_type == 'iso':
        input_type_map = INPUT_TYPES_ISO
    else:
        raise ValueError(f"Unknown embed_type: {args.embed_type}")

    # Optional env override (kept)
    HPO_TRIALS = int(os.environ.get("HPO_TRIALS", str(HPO_TRIALS)))

    # Extract parameters
    participant_id = args.participant_id
    n_fold = args.n_fold
    unfreeze_last_n = args.unfreeze_last_n
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
    alpha = 0.5      # <-- steadier on small data
    lr_scheduler_type = "cosine"
    i_fold = args.i_fold
    roi = None
    tr = None
    finetune_by = args.finetune_by
    # global_trainer = CorrHuberTrainer
    global_trainer=MSETrainer
    global_eval_metric='mse'
    if global_eval_metric=='corr':
        greater_is_better = True
    elif global_eval_metric=='mse':
        greater_is_better = False
    if args.finetune_by == 'fmri':
        roi = args.roi
        tr = args.tr

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
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    global_preds: List[np.ndarray] = []
    global_labels: List[np.ndarray] = []
    mse_tracking = []

    output_dir = models_base / f"model_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"

    print(f"Training fold {i_fold}, Model {model_name}")
    train_cids, test_cids = load_fold_cids(n_fold, i_fold, ds)
    train_cids_train, train_cids_val = train_test_split(train_cids, test_size=0.33, random_state=SEED)

    # ----- Build raw datasets -----
    train_raw, num_targets = build_hf_text_dataset_for_cids(
        ds, participant_id, behavior_embeddings, train_cids_train, input_type,
        finetune_by=finetune_by, roi=roi, tr=tr
    )
    val_raw,   _ = build_hf_text_dataset_for_cids(
        ds, participant_id, behavior_embeddings, train_cids_val, input_type,
        finetune_by=finetune_by, roi=roi, tr=tr
    )
    test_raw,  _ = build_hf_text_dataset_for_cids(
        ds, participant_id, behavior_embeddings, test_cids, input_type,
        finetune_by=finetune_by, roi=roi, tr=tr
    )

    # ----- Per-fold label standardization (train stats only) -----
    label_cols = sorted([c for c in train_raw.column_names if c.startswith("prop")],
                        key=lambda x: int(x.replace("prop","")))
    train_mat = np.stack([train_raw[c] for c in label_cols], axis=1).astype("float32")
    mu = train_mat.mean(axis=0)
    sigma = train_mat.std(axis=0) + 1e-8

    def scale_labels(batch):
        for i, col in enumerate(label_cols):
            batch[col] = ((np.array(batch[col], dtype="float32") - mu[i]) / sigma[i]).tolist()
        return batch

    train_raw = train_raw.map(scale_labels, batched=True)
    val_raw   = val_raw.map(scale_labels,   batched=True)
    test_raw  = test_raw.map(scale_labels,  batched=True)

    # ----- Tokenize -----
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

    prog_cb_search = None
    if (unfreeze_last_n == "adaptive") and (not use_lora):
        schedule = [(0, 0), (1, 2), (3, 6), (5, "all")]
        prog_cb_search = ProgressiveUnfreezeCallback(schedule)
        callbacks_search.append(prog_cb_search)

    # ================== HPO / Grid ==================
    best_params = None
    per_device_train_batch_size = batch_size
    warmup_ratio = 0.1
    gradient_accumulation_steps = 1

    if use_grid:
        print("[HPO] Running deterministic grid search...")
        best_score = -1e9
        best_params = None

        # -------- model_init factory that reads the current candidate (incl. LoRA) --------
        def model_init_for_grid(cand):
            def _factory():
                m = AutoModelForSequenceClassification.from_pretrained(
                    model_name_path, num_labels=num_targets, trust_remote_code=True
                )
                m.config.problem_type = "regression"
                if not use_lora:
                    if unfreeze_last_n != "adaptive":
                        last_n = 1_000_000 if unfreeze_last_n == "all" else int(unfreeze_last_n)
                        apply_unfreeze_last_n(m, last_n)
                    ref = {n: p.detach().clone()
                           for n, p in m.named_parameters()
                           if not (("classifier" in n) or ("regression" in n) or ("score" in n))}
                    m._ref_params = ref
                else:
                    target_modules = guess_lora_targets(m) if cand["lora_target"] == "auto" \
                        else [t.strip() for t in cand["lora_target"].split(",") if t.strip()]
                    print_lora_summary(m, target_modules)
                    lcfg = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        r=int(cand["lora_r"]),
                        lora_alpha=int(cand["lora_alpha"]),
                        lora_dropout=float(cand["lora_dropout"]),
                        bias=cand["lora_bias"],
                        target_modules=target_modules,
                    )
                    m = get_peft_model(m, lcfg)
                    ensure_classifier_trainable(m)
                    m._ref_params = {}
                return m
            return _factory

        base_kwargs = dict(
            metric_for_best_model=f"{global_eval_metric}_mean",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="no",
            report_to=[],
            remove_unused_columns=True,
            save_safetensors=False,
            greater_is_better=greater_is_better,
            max_grad_norm=1.0,
            dataloader_num_workers=4,
            seed=SEED,
        )

        for cand in grid_candidates(using_lora=use_lora):
            grid_results = [] 
            set_full_determinism(SEED)
            # unique (but disposable) output dir per candidate
            cand_tag = "_".join(str(cand[k]) for k in sorted(cand.keys()))
            args_cand = TrainingArguments(
                output_dir=models_base / f"_grid_discard_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}_{hash(cand_tag)%100000}",
                **base_kwargs,
                learning_rate=cand["learning_rate"],
                weight_decay=cand["weight_decay"],
                per_device_train_batch_size=cand["per_device_train_batch_size"],
                warmup_ratio=cand["warmup_ratio"],
                lr_scheduler_type=cand["lr_scheduler_type"],
                gradient_accumulation_steps=cand["gradient_accumulation_steps"],
                num_train_epochs=cand["num_train_epochs"],
                per_device_eval_batch_size=32,
                bf16=True,
                fp16=False,        # keep False if bf16 is available on A100; switch if needed
                tf32=True,         # HF forwards this to torch.backends flags
                load_best_model_at_end=True,
            )
            callbacks_grid = [EarlyStoppingCallback(early_stopping_patience=5)]
            if prog_cb_search is not None:
                callbacks_grid.append(prog_cb_search)

            trainer_g = global_trainer(
                model_init=model_init_for_grid(cand),
                args=args_cand,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                huber_delta=1.0,
                l2sp_lambda=0,   # off during search
                ref_params=None,
            )
            for cb in callbacks_grid:
                trainer_g.add_callback(cb)

            trainer_g.train()
            metrics = trainer_g.evaluate()
            score = metrics.get(f"eval_{global_eval_metric}_mean", float("-inf"))
            grid_results.append({
                "params": {k: (v if not isinstance(v, Path) else str(v)) for k, v in cand.items()},
                f"eval_{global_eval_metric}_mean": float(score),
            })
            if score > best_score:
                best_score = score
                best_params = cand
            discard_root = models_base / f"_grid_discard_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}_{hash(cand_tag)%100000}"
            try:
                discard_root_res = discard_root.resolve()
                models_base_res = models_base.resolve()
                if models_base_res in discard_root_res.parents and discard_root.name.startswith("_grid_discard_"):
                    if discard_root.exists():
                        shutil.rmtree(discard_root)
                        print(f"[CLEAN] Removed search discard folder: {discard_root}")
                    else:
                        print(f"[CLEAN] Skip: {discard_root} failed safety checks.")
            except Exception as e:
                print(f"[CLEAN][WARN] Could not resolve paths for cleanup: {e}")

        print(f"[HPO] Grid best params: {best_params}  (eval_{global_eval_metric}_mean={best_score:.4f})")


         

        stamp = _dt.utcnow().strftime("%Y%m%d-%H%M%S")
        best_blob = {
            "timestamp_utc": stamp,
            "seed": SEED,
            "mode": "grid_search",
            "dataset": ds,
            "participant_id": participant_id,
            "fold": {"n_fold": n_fold, "i_fold": i_fold},
            "model_name_path": model_name_path,
            "use_lora": bool(use_lora),
            "unfreeze_last_n": unfreeze_last_n,
            "best": {
                "params": {k: (v if not isinstance(v, Path) else str(v)) for k, v in best_params.items()},
                f"eval_{global_eval_metric}_mean": float(best_score),
            },
        }

        # Sort all candidates by score (descending) and include in a separate file
        grid_results_sorted = sorted(grid_results, key=lambda x: x[f"eval_{global_eval_metric}_mean"], reverse=True)

        best_path = out_base / f"grid_best_params_{model_name}_{n_fold}_{participant_id}_{i_fold}_{ds}.json"
        all_path  = out_base / f"grid_all_candidates_{model_name}_{n_fold}_{participant_id}_{i_fold}_{ds}.jsonl"

        with open(best_path, "w") as f:
            json.dump(best_blob, f, indent=2)
        with open(all_path, "w") as f:
            for row in grid_results_sorted:
                f.write(json.dumps(row) + "\n")

        print(f"[HPO] Saved best grid params -> {best_path}")
        print(f"[HPO] Saved all grid candidates (JSONL) -> {all_path}")

        # adopt best
        lr = float(best_params["learning_rate"])
        weight_decay = float(best_params["weight_decay"])
        per_device_train_batch_size = int(best_params["per_device_train_batch_size"])
        warmup_ratio = float(best_params["warmup_ratio"])
        lr_scheduler_type = str(best_params["lr_scheduler_type"])
        gradient_accumulation_steps = int(best_params["gradient_accumulation_steps"])
        num_train_epochs = int(best_params["num_train_epochs"])

        # also store best LoRA params in args for final model build
        if use_lora:
            args.lora_r = int(best_params["lora_r"])
            args.lora_alpha = int(best_params["lora_alpha"])
            args.lora_dropout = float(best_params["lora_dropout"])
            args.lora_bias = best_params["lora_bias"]
            args.lora_target = best_params["lora_target"]

    elif HPO_TRIALS > 0:
        # ====== Seeded Optuna HPO (default branch unchanged – LoRA tuning is handled in Option B) ======
        def model_init():
            m = AutoModelForSequenceClassification.from_pretrained(
                model_name_path, num_labels=num_targets, trust_remote_code=True
            )
            m.config.problem_type = "regression"

            if not use_lora:
                if unfreeze_last_n != "adaptive":
                    last_n = 1_000_000 if unfreeze_last_n == "all" else int(unfreeze_last_n)
                    apply_unfreeze_last_n(m, last_n)
                ref = {n: p.detach().clone()
                       for n, p in m.named_parameters()
                       if not (("classifier" in n) or ("regression" in n) or ("score" in n))}
                m._ref_params = ref
            else:
                target_modules = guess_lora_targets(m) if args.lora_target.strip().lower() == "auto" \
                    else [t.strip() for t in args.lora_target.split(",") if t.strip()]
                # print_lora_summary("lora_summary",m, target_modules)
                lora_cfg = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                    bias=args.lora_bias, target_modules=target_modules,
                )
                m = get_peft_model(m, lora_cfg)
                ensure_classifier_trainable(m)
                m._ref_params = {}
            return m

        search_args = TrainingArguments(
            output_dir=models_base / f"_search_discard_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}",
            metric_for_best_model=f"{global_eval_metric}_mean",
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_strategy="no",
            report_to=[],
            remove_unused_columns=True,
            save_safetensors=False,
            greater_is_better=greater_is_better,
            max_grad_norm=1.0,
            dataloader_num_workers=4,
            seed=SEED,
            bf16=True,
        fp16=False,        # keep False if bf16 is available on A100; switch if needed
            tf32=True,         # HF forwards this to torch.backends flags
            
        )

        search_trainer = global_trainer(
            model_init=model_init,
            args=search_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks_search,
            huber_delta=1.0,
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
                "num_train_epochs": trial.suggest_int("num_train_epochs", EPOCH_MIN, EPOCH_MAX),
            }

        def compute_objective(metrics):
            return metrics[f"eval_{global_eval_metric}_mean"]

        sampler = optuna.samplers.TPESampler(seed=SEED)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0)

        print(f"[HPO] Running {HPO_TRIALS} Optuna trials…")
        best_run = search_trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            n_trials=HPO_TRIALS,
            hp_space=hp_space_optuna,
            compute_objective=compute_objective,
            sampler=sampler,
            pruner=pruner,
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
        per_device_train_batch_size = batch_size
        warmup_ratio = 0.1
        gradient_accumulation_steps = 1

    # -------------------- Build base model --------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_path, num_labels=num_targets, trust_remote_code=True
    ).to(device)
    model.config.problem_type = "regression"

    if use_lora:
        # Use tuned LoRA params if grid selected them, otherwise args.*
        lora_r = int(getattr(args, "lora_r", 8))
        lora_alpha = int(getattr(args, "lora_alpha", 16))
        lora_dropout = float(getattr(args, "lora_dropout", 0.0))
        lora_bias = getattr(args, "lora_bias", "none")
        lora_target = getattr(args, "lora_target", "auto")

        target_modules = guess_lora_targets(model) if str(lora_target).strip().lower() == "auto" \
            else [t.strip() for t in str(lora_target).split(",") if t.strip()]
        print_lora_summary(model, target_modules)
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        ensure_classifier_trainable(model)
        print(f"[LoRA] Enabled with targets={target_modules}, r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    else:
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
        ref_params = {}

    
    training_args = TrainingArguments(
        output_dir=output_dir,
        # per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=f"{global_eval_metric}_mean",
        greater_is_better=greater_is_better,
        save_total_limit=5,
        save_safetensors=False,
        remove_unused_columns=True,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=4,  # deterministic data loading
        seed=SEED,                 # propagate seed to HF
    )

    mse_callback = MSETrackingCallback(val_dataset, mse_tracking, i_fold)
    pred_callback = FinalPredictionCallback(
        test_dataset, global_preds, global_labels, mu=mu, sigma=sigma,  # <-- pass scaling
        use_lora=use_lora, base_model_name_or_path=model_name_path
    )
    # train_corr_cb = TrainEvalCorrCallback(test_dataset, val_dataset, enable=True)
    # callbacks_final.append(train_corr_cb)
    callbacks_final.append(mse_callback)
    callbacks_final.append(pred_callback)
    callbacks_final.append(EarlyStoppingCallback(early_stopping_patience=5))

    # L2-SP ON (small) when not using LoRA
    l2sp_lambda = 1e-3 if not use_lora else 0.0

    trainer = global_trainer(
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
        l2sp_lambda=l2sp_lambda,
        ref_params=ref_params,
        
    )

    for cb in callbacks_final:
        cb.trainer = trainer

    trainer.train()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    plot_loss(trainer, test_dataset, out_base, model_name, n_fold,num_train_epochs, i_fold, ds,participant_id,beh_val,unfreeze_last_n,stamp)

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
    
    # Cleanup HPO/grid discard dirs
    discard_root = models_base / f"_search_discard_{model_name}_{n_fold}_{participant_id}_{beh_val}_{unfreeze_last_n}_{i_fold}_{ds}"
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
