import os, sys, re, math, ast
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", None)  # don't truncate long strings
pd.set_option("display.width", 0)            # don't wrap to fit console width
# --- Project imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from utils.model_config import MODELS, LAYERS_END, ROIS, P_VALUES  # noqa: F401  (imported but not used)

def audit_metrics(df: pd.DataFrame, name="df", tuned=False) -> dict:
    issues: dict[str, pd.DataFrame] = OrderedDict()


    # --- required/expected columns ---
    base_expected = {
        "target_id","correlation","mse","model","ds",
        "participant_id","layer","n_fold","n_components",
        "target","run_id","type"
    }
    tuned_extra = {"z_score","behavior_embeddings","unfreeze_last_n"}
    expected = base_expected | (tuned_extra if tuned else set())

    missing_cols = sorted(expected - set(df.columns))
    if missing_cols:
        print(f"[{name}] Missing columns: {missing_cols}")

    # Make safe copies of key columns (so coercions don’t mutate original df)
    d = df.copy()

    # --- numeric sanity ---
    def bad_numeric(col, *, min_val=None, max_val=None, allow_nan=False):
        if col not in d.columns:
            return pd.DataFrame(index=[])
        vals = pd.to_numeric(d[col], errors="coerce")
        bad = ~np.isfinite(vals)
        if allow_nan:
            bad &= ~vals.isna()  # treat NaN as ok if allowed
        if min_val is not None:
            bad |= vals < min_val
        if max_val is not None:
            bad |= vals > max_val
        return d[bad]

    # correlation in [-1, 1]
    issues["bad_correlation"] = bad_numeric("correlation", min_val=-1.0, max_val=1.0)

    # mse >= 0
    issues["bad_mse"] = bad_numeric("mse", min_val=0.0)

    # p-values in [0,1] if present
    for pcol in ["p_value_correlation","p_value_mse"]:
        if pcol in d.columns:
            issues[f"bad_{pcol}"] = bad_numeric(pcol, min_val=0.0, max_val=1.0)

    # --- IDs and indices ---
    for icol in ["participant_id","layer","n_fold","target_id"]:
        if icol in d.columns:
            issues[f"bad_{icol}"] = bad_numeric(icol)  # non-finite after coercion

    # --- model and layer consistency ---
    if "model" in d.columns:
        bad_model_nan = d[d["model"].isna() | d["model"].map(_is_blank_str)]
        issues["bad_model_nan_or_blank"] = bad_model_nan

        if MODEL_MAX:
            # layer should be integer and within [0, max_layer] inclusive (adjust if your layers are 1-indexed)
            layers = pd.to_numeric(d.get("layer", np.nan), errors="coerce")
            max_allowed = d["model"].map(MODEL_MAX).astype("float")
            out_of_range = d[(~layers.isna()) & (~max_allowed.isna()) & ((layers < 0) | (layers > max_allowed))]
            issues["bad_layer_out_of_range"] = out_of_range

    # --- dataset, target, type text sanity ---
    for scol in ["ds","target","type","run_id"]:
        if scol in d.columns:
            issues[f"bad_{scol}_empty"] = d[d[scol].isna() | d[scol].map(_is_blank_str)]

    # --- tuned-specific checks ---
    if tuned:
        # unfreeze_last_n should be integer-like or NaN
        if "unfreeze_last_n" in d.columns:
            coerced_unf = d["unfreeze_last_n"].apply(_coerce_int)
            bad_unf = d[coerced_unf.isna() & ~d["unfreeze_last_n"].isna() & ~(d["unfreeze_last_n"].astype(str).str.strip().str.lower().isin(["", "none", "nan", "null"]))]
            issues["bad_unfreeze_last_n"] = bad_unf

        # z_score should be bool-ish
        if "z_score" in d.columns:
            z_norm = d["z_score"].astype(str).str.strip().str.lower()
            ok = z_norm.isin(["true","false","1","0","t","f","none","nan",""])
            issues["bad_z_score"] = d[~ok]

        # behavior_embeddings should parse to tuple OR be empty/NA
        if "behavior_embeddings" in d.columns:
            parsed = d["behavior_embeddings"].apply(_beh_tuple)
            # parsing failures (None) while the source is not NA/blank
            src = d["behavior_embeddings"].astype(str).str.strip().str.lower()
            bad_beh_parse = d[(parsed.isna()) & ~(src.isin(["", "none", "nan", "null"]))]
            issues["bad_behavior_embeddings_parse"] = bad_beh_parse

    # --- duplicates on a key that should be unique per row ---
    # Adjust keys to your file’s uniqueness (here: everything but metrics)
    key_cols = [c for c in ["model","ds","participant_id","n_fold","layer","target","type","run_id",
                            "unfreeze_last_n","behavior_embeddings","n_components"] if c in d.columns]
    if key_cols:
        dup_mask = d.duplicated(subset=key_cols, keep=False)
        issues["duplicates_on_key"] = d[dup_mask].sort_values(key_cols)

    # --- collect only non-empty issue tables ---
    issues = {k: v for k, v in issues.items() if not v.empty}

    # Print a compact summary
    print(f"\n=== Audit summary: {name} ===")
    if not issues:
        print("No obvious bad rows found ✅")
    else:
        for k, baddf in issues.items():
            print(f"- {k}: {len(baddf)} rows")

    return issues

def _slug(x) -> str:
    """Safe filename token."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "None"
    if isinstance(x, (list, tuple)):
        x = "-".join(map(str, x))
    s = str(x)
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s if s else "NA"
# ----------------- Helpers -----------------
def _norm_beh(x):
    """Normalize behavior list-like fields to a comparable tuple of strings."""
    if isinstance(x, (list, tuple)):
        return tuple(map(str, x))
    if pd.isna(x):
        return tuple()
    s = str(x).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return tuple(map(str, parsed))
    except Exception:
        pass
    s = s.strip("[]")
    parts = [p.strip(" '\"\t") for p in s.split(",") if p.strip()]
    return tuple(parts)

def _is_na_value(val) -> bool:
    """Return True if filter value means NA/empty."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False

def df_filter(
    df: pd.DataFrame,
    *,
    filters: Optional[Dict[str, Union[str, int, float, list]]] = None,
    # groupby: Optional[List[str]] = None,
    # agg: str = "mean",
) -> pd.DataFrame:
    """
    Filter df on 'filters' and aggregate correlation by 'groupby'.
    Supports NA filtering by passing None/'nan'/'' for any column
    (including unfreeze_last_n). Special handling for behavior list columns.
    """
    f = df.copy()

    # Coerce common numerics
    for col in ("layer", "participant_id", "n_fold", "unfreeze_last_n", "target_id"):
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce")

    # z_score to bool if present
    if "z_score" in f.columns:
        f["z_score"] = (
            f["z_score"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        )

    # Normalize behavior list columns
    if "behavior_embeddings" in f.columns:
        f["_beh"] = f["behavior_embeddings"].map(_norm_beh)
    
    # Apply filters with NA-aware semantics
    for key, val in (filters or {}).items():
        if key in ("behavior_embeddings"):
            col ="_beh"
            if col not in f.columns:
                raise KeyError(f"Filter column '{key}' not found.")
            if _is_na_value(val):
                f = f[f[col].apply(lambda t: len(t) == 0)]
            else:
                f = f[f[col] == _norm_beh(val)]
            continue

        if key not in f.columns:
            raise KeyError(f"Filter column '{key}' not found.")

        vals = val if isinstance(val, (list, tuple, set)) else [val]
        want_na = any(_is_na_value(v) for v in vals)
        non_na_vals = [v for v in vals if not _is_na_value(v)]

        # Coerce to numeric when the column is numeric
        if pd.api.types.is_numeric_dtype(f[key]):
            coerced = []
            for v in non_na_vals:
                try:
                    coerced.append(int(v))
                except Exception:
                    try:
                        coerced.append(float(v))
                    except Exception:
                        coerced.append(v)
            non_na_vals = coerced

        mask = pd.Series(False, index=f.index)
        if non_na_vals:
            mask |= f[key].isin(non_na_vals)
        if want_na:
            mask |= f[key].isna()
        f = f[mask]
    return f
    # if f.empty:
    #     raise ValueError("No rows matched the filters.")

    # if not groupby:
    #     groupby = ["model", "layer"]

    # aggfunc = "mean" if agg == "mean" else "median"
    # out = (
    #     f.groupby(groupby, dropna=False)["correlation"]
    #     .agg([(f"correlation_{aggfunc}", aggfunc), ("n", "count")])
    #     .reset_index()
    #     .sort_values(groupby)
    # )
    # return out

# -------------- File naming utilities --------------
def parse_metrics_filename(path: Union[str, Path]) -> Dict[str, str]:
    """
    Extract model, dataset, and runid from filename:
    behavior_metrics_model-<MODEL>_ds-<DATASET>_runid-<RUNID>.csv
    """
    fname = Path(path).name
    m = re.match(r".*metrics_model-(.+?)_ds-(.+?)_runid-(.+?)\.csv", fname)
    if not m:
        raise ValueError(f"Filename does not match pattern: {fname}")
    return {"model": m.group(1), "ds": m.group(2), "run_id": m.group(3)}

def parse_metrics_filename_tuned(path: Union[str, Path]) -> Dict[str, str]:
    """
    Extract model, dataset, and runid from filename:
    behavior_metrics_model-<MODEL>_ds-<DATASET>_runid-<RUNID>.csv
    """
    fname = Path(path).name
    m = re.match(r".*metrics_model-(.+?)_ds-(.+?)_runid-(.+?)_unfreeze-(.+?)_behembd-(.+?)\.csv", fname)
    if not m:
        raise ValueError(f"Filename does not match pattern: {fname}")
    return {"model": m.group(1), "ds": m.group(2), "run_id": m.group(3), "unfreeze_last_n": m.group(4), "behavior_embeddings": m.group(5)}


def load_metrics_file(path: Union[str, Path]) -> pd.DataFrame:
    meta = parse_metrics_filename(path)
    df = pd.read_csv(path)
    df["source_file"] = Path(path).name
    df["row_idx"] = df.index   # preserve original index from CSV
    for k, v in meta.items():
        df[k] = v
    return df

def load_metrics_tuned_file(path: Union[str, Path]) -> pd.DataFrame:
    meta = parse_metrics_filename_tuned(path)
    df = pd.read_csv(path)
    df["source_file"] = Path(path).name
    df["row_idx"] = df.index
    for k, v in meta.items():
        df[k] = v
    return df



def load_all_metrics(directory: Union[str, Path]) -> pd.DataFrame:
    directory = Path(directory)
    files = list(directory.glob("*metrics_model-*_ds-*_runid-*.csv"))
    if not files:
        raise ValueError(f"No metrics CSVs found in {directory}")
    dfs = [load_metrics_file(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_tuned_metrics(directory: Union[str, Path]) -> pd.DataFrame:
    directory = Path(directory)
    files = list(directory.glob("*metrics_model-*_ds-*_runid-*_unfreeze-*_behembd-*.csv"))
    if not files:
        raise ValueError(f"No metrics CSVs found in {directory}")
    dfs = [load_metrics_tuned_file(f) for f in files]
    return pd.concat(dfs, ignore_index=True)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import re

def _slug(x) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(x)).strip("-").lower()

def reduce_to_max_layer_per_target(
    df: pd.DataFrame,
    *,
    group_cols=("participant_id", "model", "type", "target"),
    corr_col="correlation",
    layer_col="layer",
    pval_col: str | None = None,
    pval_thresh: float | None = None,
) -> pd.DataFrame:
    """
    For each (pid, model, type, target), keep the row with the maximum correlation.
    Renames outputs to corr_max and layer_at_max (and keeps pval from that row if present).
    """
    # missing = set(group_cols) | {corr_col, layer_col} - set(df.columns)
    # if missing:
    #     raise KeyError(f"Missing columns for max-layer reduction: {missing}")
    if pval_col is not None  and pval_thresh is not None:
        df = df[df[pval_col] <= pval_thresh]
    # idx of max corr per group
    idx = df.groupby(list(group_cols), dropna=False)[corr_col].idxmax()
    out = df.loc[idx].copy()

    # rename to explicit columns
    # out = out.rename(columns={
    #     corr_col: "corr_max",
    #     layer_col: "layer_at_max",
    # })

    # (Optional) keep p-value from the argmax row, rename for clarity
    # if pval_col and pval_col in out.columns:
    #     out = out.rename(columns={pval_col: "p_value_corr_at_max"})

    return out

def plot_correlation_bars_by_participant_grid(
    df_all: pd.DataFrame,
    *,
    corr_col: str = "correlation",
    target_col: str = "target",
    pid_col: str = "participant_id",
    model_col: str = "model",            # <-- NEW: columns in the grid
    type_col: str = "type",              # bars within each subplot
    pval_col: str | None = None,
    pval_thresh: float | None = None,
    sort_targets: str = "mean",
    figsize_per_sub=(5, 3.5),
    sharey: bool = True,
    save_dir: str | None = None,
    filename_stub: str | None = None,
    show: bool = False
):
    df = df_all.copy()

    # Optional p-value filtering
    if pval_col is not None and pval_col in df.columns and pval_thresh is not None:
        df = df[df[pval_col] <= pval_thresh]

    # Sanity check
    needed_cols = {corr_col, target_col, pid_col, model_col}
    if type_col:
        needed_cols.add(type_col)
    missing = needed_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_all: {missing}")

    # Orders
    type_order = sorted(df[type_col].dropna().unique().tolist()) if type_col in df.columns else []
    n_types = max(1, len(type_order))

    models = sorted(df[model_col].dropna().unique().tolist())
    if len(models) == 0:
        print("[plot] No models found; aborting.")
        return

    pids = sorted(df[pid_col].dropna().unique().tolist())
    if len(pids) == 0:
        print("[plot] No participants found; aborting.")
        return

    targets = df[target_col].dropna().unique().tolist()
    if sort_targets == "alpha":
        targets = sorted(targets, key=lambda x: str(x))
    elif sort_targets == "mean":
        order = (df.groupby(target_col, dropna=False)[corr_col]
                   .mean()
                   .sort_values(ascending=False)
                   .index.tolist())
        seen = set(targets)
        targets = [t for t in order if t in seen]
    n_targets = len(targets)
    if n_targets == 0:
        print("[plot] No targets found; aborting.")
        return

    # Figure + axes: rows = participants, cols = models
    nrows, ncols = len(pids), len(models)
    fig_w = figsize_per_sub[0] * ncols
    fig_h = figsize_per_sub[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=sharey)

    # Normalize axes array shapes
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    x = np.arange(n_targets)
    total_width = 0.8
    bar_width = total_width / n_types if n_types > 0 else total_width

    for r, pid in enumerate(pids):
        for c, model in enumerate(models):
            ax = axes[r, c]
            d = df[(df[pid_col] == pid) & (df[model_col] == model)]

            if d.empty:
                ax.axis("off")
                ax.set_title(f"pid={pid}, model={model}\n(no data)")
                continue

            # Draw grouped bars by type_col (or single bars if no type_col present)
            if n_types > 0:
                for i, tname in enumerate(type_order):
                    heights = []
                    for tgt in targets:
                        val = (d[(d[target_col] == tgt) & (d[type_col] == tname)][corr_col]
                               .mean())
                        heights.append(val if pd.notna(val) else 0.0)
                    offsets = x - total_width/2 + i*bar_width + bar_width/2
                    ax.bar(offsets, heights, width=bar_width, label=str(tname))
            else:
                heights = []
                for tgt in targets:
                    val = d[d[target_col] == tgt][corr_col].mean()
                    heights.append(val if pd.notna(val) else 0.0)
                ax.bar(x, heights, width=total_width)

            # Cosmetics
            ax.set_title(f"pid={pid}, model={model}")
            if r == nrows - 1:
                ax.set_xlabel(str(target_col))
                ax.set_xticks(x, [str(t) for t in targets], rotation=45, ha="right")
            else:
                # still set ticks (rotated) but hide labels to reduce clutter
                ax.set_xticks(x, ["" for _ in targets])

            if c == 0:
                ax.set_ylabel(str(corr_col))

    # Build single legend (if we had types)
    first_valid = None
    for r in range(nrows):
        for c in range(ncols):
            if axes[r, c].has_data():
                first_valid = axes[r, c]
                break
        if first_valid:
            break

    if first_valid and n_types > 0:
        handles, labels = first_valid.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title=type_col, loc="upper center",
                       ncol=max(1, len(labels)))
            fig.subplots_adjust(top=0.88)

    # Title + layout
    ttl = "Correlation by Target — rows: participants, cols: models"
    if pval_col and pval_thresh is not None:
        ttl += f" (filtered {pval_col} ≤ {pval_thresh})"
    fig.suptitle(ttl, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save/show
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stub = filename_stub or "bars_participant_x_model"
        ptag = f"__pvalle{pval_thresh}" if (pval_col and pval_thresh is not None) else ""
        out = Path(save_dir) / f"{_slug(stub)}{ptag}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.show()


def plot_maxcorr_bars_by_participant_x_model(
    df_max: pd.DataFrame,
    *,
    corr_col: str = "corr_max",
    target_col: str = "target",
    pid_col: str = "participant_id",
    model_col: str = "model",
    type_col: str = "type",
    layer_at_max_col: str = "layer_at_max",
    pval_col: str | None = "p_value_corr_at_max",
    pval_thresh: float | None = None,         # threshold on p-value of the argmax layer
    sort_targets: str = "mean",               # "mean" or "alpha"
    annotate_layer: bool = True,              # show layer index above bars
    figsize_per_sub=(5, 3.5),
    sharey: bool = True,
    save_dir: str | None = None,
    filename_stub: str | None = None,
    show: bool = False,
):
    df = df_max.copy()

    # Optional p-value filtering (on the pval from the argmax layer)
    if pval_col and pval_col in df.columns and pval_thresh is not None:
        df = df[df[pval_col] <= pval_thresh]

    needed = {corr_col, target_col, pid_col, model_col, type_col, layer_at_max_col}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_max: {missing}")

    type_order = sorted(df[type_col].dropna().unique().tolist())
    n_types = max(1, len(type_order))

    models = sorted(df[model_col].dropna().unique().tolist())
    pids = sorted(df[pid_col].dropna().unique().tolist())
    if not models or not pids:
        print("[plot] No models or participants found; aborting.")
        return

    targets = df[target_col].dropna().unique().tolist()
    if sort_targets == "alpha":
        targets = sorted(targets, key=lambda x: str(x))
    elif sort_targets == "mean":
        order = (df.groupby(target_col, dropna=False)[corr_col]
                   .mean()
                   .sort_values(ascending=False)
                   .index.tolist())
        seen = set(targets)
        targets = [t for t in order if t in seen]
    if not targets:
        print("[plot] No targets found; aborting.")
        return

    # Figure grid
    nrows, ncols = len(pids), len(models)
    fig_w = figsize_per_sub[0] * ncols
    fig_h = figsize_per_sub[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=sharey)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    x = np.arange(len(targets))
    total_width = 0.8
    bar_width = total_width / n_types

    for r, pid in enumerate(pids):
        for c, model in enumerate(models):
            ax = axes[r, c]
            d = df[(df[pid_col] == pid) & (df[model_col] == model)]

            if d.empty:
                ax.axis("off")
                ax.set_title(f"pid={pid}, model={model}\n(no data)")
                continue

            # grouped bars by type
            for i, tname in enumerate(type_order):
                heights, layers = [], []
                for tgt in targets:
                    row = d[(d[target_col] == tgt) & (d[type_col] == tname)]
                    val = row[corr_col].mean() if not row.empty else 0.0
                    heights.append(val if pd.notna(val) else 0.0)

                    # pick the (single) argmax layer label if available
                    if annotate_layer and not row.empty:
                        layers.append(row.iloc[0][layer_at_max_col])
                    else:
                        layers.append(None)

                offsets = x - total_width/2 + i*bar_width + bar_width/2
                bars = ax.bar(offsets, heights, width=bar_width, label=str(tname))

                if annotate_layer:
                    for rect, lab in zip(bars, layers):
                        if lab is None:
                            continue
                        h = rect.get_height()
                        ax.text(
                            rect.get_x() + rect.get_width()/2, h,
                            f"L{int(lab)}",
                            ha="center", va="bottom", fontsize=8, rotation=0
                        )

            # cosmetics
            ax.set_title(f"pid={pid}, model={model}")
            if r == nrows - 1:
                ax.set_xlabel(str(target_col))
                ax.set_xticks(x, [str(t) for t in targets], rotation=45, ha="right")
            else:
                ax.set_xticks(x, [""] * len(targets))
            if c == 0:
                ax.set_ylabel(str(corr_col))

    # legend + title
    first_valid = None
    for r in range(nrows):
        for c in range(ncols):
            if axes[r, c].has_data():
                first_valid = axes[r, c]
                break
        if first_valid:
            break

    if first_valid and type_order:
        handles, labels = first_valid.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title=type_col, loc="upper center", ncol=max(1, len(labels)))
            fig.subplots_adjust(top=0.88)

    ttl = "Max correlation across layers — rows: participants, cols: models"
    if pval_col and pval_thresh is not None:
        ttl += f" (filtered {pval_col} ≤ {pval_thresh})"
    fig.suptitle(ttl, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stub = filename_stub or "maxcorr_participant_x_model"
        ptag = f"__pvalle{pval_thresh}" if (pval_col and pval_thresh is not None) else ""
        out = Path(save_dir) / f"{_slug(stub)}{ptag}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.show()



def filter_last_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows corresponding to the last layer of each model.
    Uses LAYERS_END from utils.model_config.
    """
    if "model" not in df.columns or "layer" not in df.columns:
        raise KeyError("Both 'model' and 'layer' columns must be present in df")

    last_layer_map = {model: LAYERS_END[i] for i, model in enumerate(MODELS)}

    # Keep only rows where layer == last layer for that model
    mask = df.apply(lambda row: row["layer"] == last_layer_map.get(row["model"], None), axis=1)
    return df[mask].copy()


# ----------------- Usage -----------------
base_dir = "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment"
out_dir = f"{base_dir}/figs_corr_bars"
df_behavior = load_all_metrics(f"{base_dir}/Sep9_metrics_20250909T230310Z")
df_behavior['type'] = 'behavior'
print("df_behavior:", df_behavior.columns.values.tolist(), df_behavior.shape)
df_behaviortuned = load_tuned_metrics(f"{base_dir}/Sep10_behaviortuned_metrics_02")
df_behaviortuned['type'] = 'behavior_finetuned'
print("df_behaviortuned:", df_behaviortuned.columns.values.tolist(), df_behaviortuned.shape)

# for unfreeze_last_n in [None, 1, 2]:
#     for behavior_embeddings in df_behaviortuned["behavior_embeddings"].unique():
#         for n_components in df_behavior["n_components"].unique():
#             df_behaviortuned_filtered = df_filter(
#                 df_behaviortuned,
#                 filters={
#                     "ds": "sagar2023",
#                     "unfreeze_last_n": unfreeze_last_n,
#                     "behavior_embeddings": behavior_embeddings,
#                     "n_components": n_components,
#                 })
#             df_behavior_filtered = df_filter(
#                 df_behavior,
#                 filters={"ds": "sagar2023"}
#             )

#             # combine, keep last layer
#             df_all = pd.concat([df_behaviortuned_filtered, df_behavior_filtered], ignore_index=True)
#             print("df_all:", df_all.columns.values.tolist(), df_all.shape)
#             df_all_last = filter_last_layer(df_all.copy())
#             print("df_all_last:", df_all.columns.values.tolist(), df_all_last.shape)
#             df_all_max = reduce_to_max_layer_per_target(
#             df_all,
#             group_cols=("participant_id", "model", "type", "target"),
#             corr_col="correlation",
#             layer_col="layer",
#             pval_col="p_value_correlation"  # if present
#             , pval_thresh=0.05  # if present
# )


#             # build a filename stub from the selections
#             beh_stub = _slug(behavior_embeddings)
#             stub = f"ds-sagar2023__unf-{_slug(unfreeze_last_n)}__beh-{beh_stub}__ncomp-{_slug(n_components)}"
#             print(df_all_last['target'].unique())
#             # save one PNG per participant automatically
#             plot_correlation_bars_by_participant_grid(
#                 df_all_last,
#                 corr_col="correlation",
#                 target_col="target",
#                 pid_col="participant_id",
#                 type_col="type",
#                 pval_col="p_value_correlation",
#                 pval_thresh=0.05,
#                 sort_targets="mean",
#                 save_dir=out_dir,
#                 filename_stub=stub,
#                 show=False
#             )
#             plot_maxcorr_bars_by_participant_x_model(
#             df_all_max,
#             corr_col="correlation",
#             target_col="target",
#             pid_col="participant_id",
#             model_col="model",
#             type_col="type",
#             layer_at_max_col="layer",
#             pval_col="p_value_correlation",      # threshold on pval from the max layer
#             pval_thresh=0.05,
#             sort_targets="mean",
#             annotate_layer=True,
#             save_dir=out_dir,
#             filename_stub=stub + "__maxlayerrrr",
#             show=False
#                 )



# ===== Bad-row auditor =====
from collections import OrderedDict

# Build model -> max_layer map if available
MODEL_MAX = {}
try:
    MODEL_MAX = {m: LAYERS_END[i] for i, m in enumerate(MODELS)}
except Exception:
    pass  # ok if not available

def _is_blank_str(s):
    return isinstance(s, str) and s.strip() == ""

def _coerce_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan

def _beh_tuple(x):
    try:
        return _norm_beh(x)
    except Exception:
        return None  # mark as failed to parse


# issues_behavior = audit_metrics(df_behavior, name="df_behavior", tuned=False)
# issues_beh_tuned = audit_metrics(df_behaviortuned, name="df_behaviortuned", tuned=True)

# for tag, bad in issues_behavior.items():
#     print(f"\n[behavior] {tag}")
#     print(bad[["source_file","row_idx"]].head())

# for tag, bad in issues_beh_tuned.items():
#     print(f"\n[behaviortuned] {tag}s")
#     print(bad[["source_file","row_idx"]].head())

for unfreeze_last_n in [None, 1, 2]:
    for behavior_embeddings in df_behaviortuned["behavior_embeddings"].unique():
        for n_components in df_behavior["n_components"].unique():
            # print(f"behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
            print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
            df_behaviortuned_filtered = df_filter(
                df_behaviortuned,
                filters={
                    "ds": "sagar2023",
                    "unfreeze_last_n": unfreeze_last_n,
                    "behavior_embeddings": behavior_embeddings,
                    "n_components": n_components,
                })
            df_behavior_filtered = df_filter(
                df_behavior,
                filters={"ds": "sagar2023"}
            )

            # combine, keep last layer
            df_all = pd.concat([df_behaviortuned_filtered, df_behavior_filtered], ignore_index=True)
            print("df_all:", df_all.columns.values.tolist(), df_all.shape)
            df_all_last = filter_last_layer(df_all.copy())
            print("df_all_last:", df_all.columns.values.tolist(), df_all_last.shape)
            df_all_max = reduce_to_max_layer_per_target(
            df_all,
            group_cols=("participant_id", "model", "type", "target"),
            corr_col="correlation",
            layer_col="layer",
            pval_col="p_value_correlation"  # if present
            , pval_thresh=0.05  # if present
)


            # build a filename stub from the selections
            beh_stub = _slug(behavior_embeddings)
            stub = f"ds-sagar2023__unf-{_slug(unfreeze_last_n)}__beh-{beh_stub}__ncomp-{_slug(n_components)}"
            print(df_all_last['target'].unique())
            # save one PNG per participant automatically
            plot_correlation_bars_by_participant_grid(
                df_all_last,
                corr_col="correlation",
                target_col="target",
                pid_col="participant_id",
                type_col="type",
                pval_col="p_value_correlation",
                pval_thresh=0.05,
                sort_targets="mean",
                save_dir=out_dir,
                filename_stub=stub,
                show=False
            )
            plot_maxcorr_bars_by_participant_x_model(
            df_all_max,
            corr_col="correlation",
            target_col="target",
            pid_col="participant_id",
            model_col="model",
            type_col="type",
            layer_at_max_col="layer",
            pval_col="p_value_correlation",      # threshold on pval from the max layer
            pval_thresh=0.05,
            sort_targets="mean",
            annotate_layer=True,
            save_dir=out_dir,
            filename_stub=stub + "__maxlayerrrr",
            show=False
                )



