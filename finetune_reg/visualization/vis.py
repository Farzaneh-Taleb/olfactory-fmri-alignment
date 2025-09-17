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

def audit_metrics(df: pd.DataFrame, name="df", tuned=False, drop_dupes=False) -> dict:
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
            bad_unf = d[coerced_unf.isna() & ~d["unfreeze_last_n"].isna() & ~(d["unfreeze_last_n"].astype(str).str.strip().str.lower().isin(["adaptive","all", "none", "nan", "null",""]))]
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
                            "unfreeze_last_n","behavior_embeddings","n_components","target_id","cid","z_score"] if c in d.columns]
    

    

    if key_cols:
        dup_mask = d.duplicated(subset=key_cols, keep=False)
        dup_df = d[dup_mask].sort_values(key_cols)
        if not dup_df.empty:
            issues["duplicates_on_key"] = dup_df
            if drop_dupes:
                # drop and keep the first occurrence
                before = len(d)
                d = d.drop_duplicates(subset=key_cols, keep="first")
                print(f"[{name}] Dropped {before - len(d)} duplicate rows")

    # --- collect only non-empty issue tables ---
    issues = {k: v for k, v in issues.items() if not v.empty}

    print(f"\n=== Audit summary: {name} ===")
    if not issues:
        print("No obvious bad rows found ✅")
    else:
        for k, baddf in issues.items():
            print(f"- {k}: {len(baddf)} rows")


    # Print a compact summary
    print(f"\n=== Audit summary: {name} ===")
    if not issues:
        print("No obvious bad rows found ✅")
    else:
        for k, baddf in issues.items():
            print(f"- {k}: {len(baddf)} rows")

    return issues,d

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

def df_filter(df: pd.DataFrame, *, filters=None):
    f = df.copy()

    # Coerce common numerics (add n_components here!)
    for col in ("layer", "participant_id", "n_fold", "target_id", "n_components"):
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce")

    if "z_score" in f.columns:
        f["z_score"] = f["z_score"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    if "behavior_embeddings" in f.columns:
        f["_beh"] = f["behavior_embeddings"].map(_norm_beh)

    for key, val in (filters or {}).items():
        if key in ("behavior_embeddings",):
            col = "_beh"
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

        # Force numeric coercion for numeric-like cols (including n_components) even if dtype is object
        numeric_like = {"layer", "participant_id", "n_fold", "target_id", "n_components"}
        if key in numeric_like:
            f[key] = pd.to_numeric(f[key], errors="coerce")
            coerced = []
            for v in non_na_vals:
                try:
                    coerced.append(float(v))
                except Exception:
                    pass
            non_na_vals = coerced

            # float-safe matching (np.isclose) if we have non-NA numeric values
            mask = pd.Series(False, index=f.index)
            if non_na_vals:
                colvals = f[key].astype(float)
                for v in non_na_vals:
                    mask |= np.isclose(colvals, float(v), rtol=1e-6, atol=1e-8, equal_nan=False)
            else:
                mask = pd.Series(False, index=f.index)
        else:
            # regular exact-match path
            mask = pd.Series(False, index=f.index)
            if non_na_vals:
                mask |= f[key].isin(non_na_vals)

        if want_na:
            mask |= f[key].isna()

        f = f[mask]

    return f

# -------------- File naming utilities --------------
def parse_metrics_filename(path: Union[str, Path]) -> Dict[str, str]:
    """
    Extract model, dataset, and runid from filename:
    behavior_metrics_model-<MODEL>_ds-<DATASET>_runid-<RUNID>.csv
    """
    fname = Path(path).name
    m = re.match(r".*metrics.*_model-(.+?)_ds-(.+?)\.csv", fname)
    if not m:
        raise ValueError(f"Filename does not match pattern: {fname}")
    return {"model": m.group(1), "ds": m.group(2)}



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


def parse_metrics_filename_tuned2(path: Union[str, Path]) -> Dict[str, str]:
    """
    Extract model, dataset, and runid from filename:
    behavior_metrics_model-<MODEL>_ds-<DATASET>_runid-<RUNID>.csv
    """
    fname = Path(path).name
    m = re.match(r".*metrics.*_model-(.+?)_ds-(.+?)_unfreeze-(.+?)_behembd-(.+?)\.csv", fname)
    if not m:
        raise ValueError(f"Filename does not match pattern: {fname}")
    return {"model": m.group(1), "ds": m.group(2), "unfreeze_last_n": m.group(3), "behavior_embeddings": m.group(4)}


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

def load_metrics_tuned_file2(path: Union[str, Path]) -> pd.DataFrame:
    meta = parse_metrics_filename_tuned2(path)
    df = pd.read_csv(path)
    df["source_file"] = Path(path).name
    df["row_idx"] = df.index
    for k, v in meta.items():
        df[k] = v
    return df



def load_all_metrics(directory: Union[str, Path]) -> pd.DataFrame:
    directory = Path(directory)
    files = list(directory.glob("*metrics*_model-*_ds-*.csv"))
    if not files:
        raise ValueError(f"No metrics CSVs found in {directory}")
    dfs = [load_metrics_file(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def load_tuned_metrics(directory: Union[str, Path]) -> pd.DataFrame:
    directory = Path(directory)
    files = list(directory.glob("*metrics_model-*_ds-*_runid-*_unfreeze-*_behembd-*.csv"))
    if not files:
        raise ValueError(f"No metrics CSVs found in {directory}")
    
def load_tuned_metrics2(directory: Union[str, Path]) -> pd.DataFrame:
    directory = Path(directory)
    files = list(directory.glob("*metrics_model-*_ds-*_unfreeze-*_behembd-*.csv"))
    if not files:
        raise ValueError(f"No metrics CSVs found in {directory}")
    
    dfs = [load_metrics_tuned_file2(f) for f in files]
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

def average_over_descriptors(
    df: pd.DataFrame,
    *,
    value_col: str = "correlation",
    target_col: str = "target",
    group_cols: tuple[str, ...] = ("participant_id", "model", "type"),
    pval_col: str | None = None,
    pval_thresh: float | None = None,
    min_targets: int = 1,
) -> pd.DataFrame:
    """
    Average the metric in `value_col` over all descriptors (i.e., over `target_col`)
    for each group in `group_cols`. Returns per-group mean/median/std/sem and count.

    Typical use:
        # last-layer slice
        df_last = filter_last_layer(df_all)
        avg_last = average_over_descriptors(df_last)

        # or: max-layer-per-target slice
        df_max  = reduce_to_max_layer_per_target(df_all, ...)
        avg_max = average_over_descriptors(df_max)

    Args:
        df: input DataFrame.
        value_col: metric to average (e.g., "correlation").
        target_col: column that marks descriptors (e.g., "target").
        group_cols: group keys to aggregate by.
        pval_col: optional p-value column to filter on.
        pval_thresh: keep rows with pval <= threshold (if provided).
        min_targets: require at least this many targets per group to keep the row.

    Returns:
        DataFrame with columns:
            * group_cols...
            * n_targets
            * mean, median, std, sem  (computed over targets)
    """
    if value_col not in df.columns or target_col not in df.columns:
        missing = {value_col, target_col} - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")

    d = df.copy()

    # Optional p-value filtering
    if pval_col is not None and pval_col in d.columns and pval_thresh is not None:
        d = d[d[pval_col] <= pval_thresh].copy()

    # Make sure we only aggregate once per target within the group if duplicates exist
    # (e.g., multiple layers). If the slice already resolved that (last or max), this is a no-op.
    # We collapse duplicates by taking the mean within (group_cols + target).
    collapse_keys = list(group_cols) + [target_col]
    d = (d.groupby(collapse_keys, dropna=False, as_index=False)[value_col]
           .mean())

    # Now aggregate over the descriptors (targets)
    def _sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        n = len(x)
        return float(x.std(ddof=1) / math.sqrt(n)) if n >= 2 else float("nan")

    out = (d.groupby(list(group_cols), dropna=False)[value_col]
             .agg(n_targets="count",
                  mean="mean",
                  median="median",
                  std=lambda x: pd.to_numeric(x, errors="coerce").std(ddof=1),
                  sem=_sem)
             .reset_index())

    # Enforce a minimum number of descriptors
    out = out[out["n_targets"] >= int(min_targets)].reset_index(drop=True)
    return out

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel

def plot_avg_over_descriptors_by_participant_grid(
    df_all: pd.DataFrame,
    *,
    value_col: str = "correlation",
    target_col: str = "target",
    pid_col: str = "participant_id",
    model_col: str = "model",
    type_col: str = "type",
    pval_col: str | None = None,
    pval_thresh: float | None = None,
    agg: str = "mean",           # "mean" or "median"
    error: str | None = "sem",   # "sem", "std", or None
    min_targets: int = 1,
    figsize_per_sub=(4.8, 3.2),
    sharey: bool = True,
    save_dir: str | None = None,
    filename_stub: str | None = None,
    show: bool = False,
    # --- NEW: testing options ---
    compare_types: tuple[str, str] | None = None,  # e.g., ("frozen","behavior_tuned")
    test_kind: str = "wilcoxon",                   # "wilcoxon" or "ttest"
    fisher_z: bool = True,                         # apply Fisher z to correlations for tests & summaries
    star_levels: tuple[float, float, float] = (0.05, 0.01, 0.001),
    alternative="greater"
):
    """
    Plot average metric over all descriptors (targets) for each (participant, model),
    with bars split by `type`. Each bar is annotated with its average value.
    Additionally:
      • Per-subject paired test across descriptors between two `type`s (before vs after),
        with significance stars drawn between the two bars.
      • Group-level test across subjects per model on subject-wise mean improvements (after - before)
        with stars appended to the column (model) title.

    Assumes exactly two comparison types are provided via `compare_types`.
    """

    d = df_all.copy()

    # Optional p-value filtering
    if pval_col is not None and pval_col in d.columns and pval_thresh is not None:
        d = d[d[pval_col] <= pval_thresh].copy()

    # Basic checks
    needed = {value_col, target_col, pid_col, model_col, type_col}
    missing = needed - set(d.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Ensure numeric correlations and within [-1, 1]
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    # Decide which two types to compare
    all_types = d[type_col].dropna().unique().tolist()
    if compare_types is None:
        if len(all_types) != 2:
            raise ValueError(
                f"Expected exactly two {type_col} levels for comparison, found {len(all_types)}: {all_types}. "
                f"Pass them via compare_types=('before','after') etc."
            )
        compare_types = (all_types[0], all_types[1])
    else:
        for t in compare_types:
            if t not in all_types:
                raise ValueError(f"compare_types includes '{t}' which is not in {type_col} levels: {all_types}")

    t_before, t_after = compare_types

    # Fisher z transform helper
    def fisher_z_fn(r: pd.Series | np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    # Collapse duplicates to one row per (pid, model, type, target) by mean
    base_keys = [pid_col, model_col, type_col, target_col]
    d = (
        d.groupby(base_keys, dropna=False, as_index=False)[value_col]
         .mean()
    )

    # If Fisher z is requested, compute it for testing and for averaging
    if fisher_z:
        d["_val_for_stats"] = fisher_z_fn(d[value_col].values)
        val_for_plot = "_val_for_stats"
        y_label_metric = f"{agg}(Fisher-z({value_col}))"
    else:
        d["_val_for_stats"] = d[value_col].values
        val_for_plot = value_col
        y_label_metric = f"{agg}({value_col})"

    # Aggregate across targets
    def _sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        n = len(x)
        return float(x.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")

    agg_fun = "mean" if agg == "mean" else "median"
    grouped = d.groupby([pid_col, model_col, type_col], dropna=False)
    summary = grouped[val_for_plot].agg(
        n_targets="count",
        val=agg_fun,
        std=lambda x: pd.to_numeric(x, errors="coerce").std(ddof=1),
        sem=_sem
    ).reset_index()

    # Filter groups with too few targets
    summary = summary[summary["n_targets"] >= int(min_targets)].reset_index(drop=True)

    # Setup grid
    models = sorted(summary[model_col].dropna().unique().tolist())
    pids   = sorted(summary[pid_col].dropna().unique().tolist())
    types  = [t_before, t_after]  # enforce order for plotting

    if not models or not pids:
        print("[plot] Nothing to plot.")
        return

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

    x = np.arange(len(types))
    bar_width = 0.7
    err_key = error if error in {"std", "sem"} else None

    # --- Group-level test per model (across subjects) ---
    # Build subject-wise mean improvement (after - before) per model
    group_pvals_by_model: dict[str, float | None] = {}
    for model in models:
        deltas = []
        for pid in pids:
            # pivot by target to ensure pairing
            sub = d[(d[pid_col] == pid) & (d[model_col] == model)]
            piv = sub.pivot(index=target_col, columns=type_col, values="_val_for_stats")
            if t_before in piv.columns and t_after in piv.columns:
                paired = piv[[t_before, t_after]].dropna()
                if len(paired) >= max(2, min_targets):
                    delta = (paired[t_after] - paired[t_before]).mean()
                    deltas.append(delta)
        if len(deltas) >= 2:
            if test_kind.lower() == "ttest":
                _, p = ttest_rel(deltas, np.zeros_like(deltas), alternative=alternative)
            else:
                # Wilcoxon vs zero
                try:
                    _, p = wilcoxon(deltas, zero_method="wilcox", alternative=alternative, mode="auto")
                except ValueError:
                    p = np.nan
        else:
            p = np.nan
        group_pvals_by_model[model] = p

    def p_to_stars(p: float | None) -> str:
        if p is None or np.isnan(p):
            return ""
        a, b, c = star_levels
        if p <= c: return "***"
        if p <= b: return "**"
        if p <= a: return "*"
        return ""

    # Draw bars + per-subject test stars
    for c, model in enumerate(models):
        for r, pid in enumerate(pids):
            ax = axes[r, c]
            sl = summary[(summary[pid_col] == pid) & (summary[model_col] == model)]
            if sl.empty:
                ax.axis("off")
                ax.set_title(f"pid={pid}, model={model}\n(no data)")
                continue

            y, yerr = [], []
            for t in types:
                row = sl[sl[type_col] == t]
                if row.empty:
                    y.append(0.0); yerr.append(0.0)
                else:
                    y.append(float(row["val"].iloc[0]))
                    yerr.append(float(row[err_key].iloc[0]) if err_key else 0.0)

            bars = ax.bar(x, y, width=bar_width,
                          yerr=(yerr if err_key else None),
                          capsize=3 if err_key else 0)

            # Annotate each bar with its value
            for rect, val in zip(bars, y):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9
                )

            # --- Per-subject paired test across descriptors ---
            sub_full = d[(d[pid_col] == pid) & (d[model_col] == model)]
            piv = sub_full.pivot(index=target_col, columns=type_col, values="_val_for_stats")
            p_pair = np.nan
            if t_before in piv.columns and t_after in piv.columns:
                paired = piv[[t_before, t_after]].dropna()
                if len(paired) >= max(2, min_targets):
                    if test_kind.lower() == "ttest":
                        _, p_pair = ttest_rel(paired[t_after].values, paired[t_before].values, alternative=alternative)
                    else:
                        try:
                            _, p_pair = wilcoxon(
                                paired[t_after].values,
                                paired[t_before].values,
                                zero_method="wilcox",
                                alternative=alternative,
                                mode="auto"
                            )
                        except ValueError:
                            p_pair = np.nan

            # Draw significance bar if significant
            star = p_to_stars(p_pair)
            if star:
                # position above the taller bar
                y_max = max(y)
                y_min = min(y)
                h = 0.04 * (abs(y_max) + abs(y_min) + 1e-6)  # small height
                y_bar = y_max + 1.5 * h
                ax.plot([x[0], x[0], x[1], x[1]], [y_bar, y_bar + h, y_bar + h, y_bar], linewidth=1.0)
                ax.text(np.mean([x[0], x[1]]), y_bar + h, star, ha="center", va="bottom", fontsize=11)

            # Titles & labels
            ax.set_title(f"pid={pid}")
            ax.set_xticks(x, [str(t) for t in types], rotation=0)
            if c == 0:
                ax.set_ylabel(f"{y_label_metric} over {target_col}s")

        # Add group-level star to the column header (model)
        gstar = p_to_stars(group_pvals_by_model.get(model, np.nan))
        # Put a bold model title with group-level stars centered above the column
        axes[0, c].set_title(f"{model} {gstar}".rstrip(), fontweight="bold")

    fig.suptitle(
        f"Average over descriptors — bars={type_col} • agg={agg}"
        + (f" • p≤{pval_thresh}" if pval_col and pval_thresh is not None else ""),
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stub = filename_stub or "avg_over_descriptors_participant_x_model_with_tests"
        out = Path(save_dir) / f"{_slug(stub)}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.show()




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
p_val = 1
# ----------------- Usage -----------------
base_dir = "/proj/rep-learning-robotics/users/x_farzt/olfactory_alignment"
out_dir = f"{base_dir}/figs_corr_bars"
start = 'Sep16'
end = '1'
# df_behavior = load_all_metrics(f"{base_dir}/Sep16_behaviormetrics_1")
# df_behavior['type'] = 'frozen'
# print("df_behavior:", df_behavior.columns.values.tolist(), df_behavior.shape)

# df_behaviortuned = load_tuned_metrics(f"{base_dir}/{start}_behaviortuned_metrics_{end}")
# df_behaviortuned['type'] = 'fine-tuned'
# print("df_behaviortuned:", df_behaviortuned.columns.values.tolist(), df_behaviortuned.shape)



# for unfreeze_last_n in df_behaviortuned["unfreeze_last_n"].unique():
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






# issues_behavior,df_behavior_clean = audit_metrics(df_behavior, name="df_behavior", tuned=False)
# issues_beh_tuned,df_behaviortuned_clean = audit_metrics(df_behaviortuned, name="df_behaviortuned", tuned=True)

# for tag, bad in issues_behavior.items():
#     print(f"\n[behavior] {tag}")
#     print(bad[["source_file","row_idx"]].head())

# for tag, bad in issues_beh_tuned.items():
#     print(f"\n[behaviortuned] {tag}s")
#     print(bad[["source_file","row_idx"]].head())

# for unfreeze_last_n in df_behaviortuned_clean["unfreeze_last_n"].unique():
#     for behavior_embeddings in df_behaviortuned_clean["behavior_embeddings"].unique():
#         for n_components in df_behaviortuned_clean["n_components"].unique():
#             # print(f"behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
#             print(f"\n=== Processing: unfreeze_last_n={unfreeze_last_n}, behavior_embeddings={behavior_embeddings}, n_components={n_components} ===")
#             df_behaviortuned_filtered = df_filter(
#                 df_behaviortuned_clean,
#                 filters={
#                     "ds": "sagar2023",
#                     # "unfreeze_last_n": unfreeze_last_n,
#                     "behavior_embeddings": behavior_embeddings,
#                     "n_components": n_components,
#                 })
#             df_behavior_filtered = df_filter(
#                 df_behavior_clean,
#                 filters={"ds": "sagar2023"}
#             )
#             print("df_behaviortuned_filtered:", df_behaviortuned_filtered.columns.values.tolist(), df_behaviortuned_filtered.shape)

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
#             , pval_thresh=p_val  # if present
# )


#             # build a filename stub from the selections
#             beh_stub = _slug(behavior_embeddings)
#             stub = f"ds-sagar2023__unf-{_slug(unfreeze_last_n)}__beh-{beh_stub}__ncomp-{_slug(n_components)}_{start}_{end}"
#             print(df_all_last['target'].unique())






            # save one PNG per participant automatically
            # plot_correlation_bars_by_participant_grid(
            #     df_all_last,
            #     corr_col="correlation",
            #     target_col="target",
            #     pid_col="participant_id",
            #     type_col="type",
            #     pval_col="p_value_correlation",
            #     pval_thresh=0.05,
            #     sort_targets="mean",
            #     save_dir=out_dir,
            #     filename_stub=stub,
            #     show=False
            # )
            # plot_maxcorr_bars_by_participant_x_model(
            # df_all_max,
            # corr_col="correlation",
            # target_col="target",
            # pid_col="participant_id",
            # model_col="model",
            # type_col="type",
            # layer_at_max_col="layer",
            # pval_col="p_value_correlation",      # threshold on pval from the max layer
            # pval_thresh=0.05,
            # sort_targets="mean",
            # annotate_layer=True,
            # save_dir=out_dir,
            # filename_stub=stub + "_maxlayer",
            # show=False
            #     )
            

            # plot_avg_over_descriptors_by_participant_grid(
            # df_all_last,
            # value_col="correlation",
            # target_col="target",
            # pid_col="participant_id",
            # model_col="model",
            # type_col="type",
            # pval_col="p_value_correlation",
            # pval_thresh=0.05,      # optional
            # agg="mean",            # or "median"
            # error="sem",           # "std" or None
            # min_targets=3,
            # save_dir=out_dir,
            # filename_stub=stub + f"__avg_over_desc_last_{p_val}",
            # show=False
            # )       
#             fisher_z = True      
#             plot_avg_over_descriptors_by_participant_grid(
#     df_all_max,
#     value_col="correlation",
#     pid_col="participant_id",
#     model_col="model",
#     type_col="type",
#     target_col="target",
#     compare_types=("frozen","fine-tuned"),  # BEFORE, AFTER
#     test_kind="wilcoxon",                       # or "ttest"
#     fisher_z=fisher_z,
#     show=False,
#     filename_stub=stub + f"__avg_over_desc_max_p_{p_val}_{fisher_z}",
#     save_dir=out_dir,
#     error="sem",
#      pval_thresh=p_val,
     
# )
            

#             fisher_z = True      
#             plot_avg_over_descriptors_by_participant_grid(
#     df_all_max,
#     value_col="correlation",
#     pid_col="participant_id",
#     model_col="model",
#     type_col="type",
#     target_col="target",
#     compare_types=("frozen","fine-tuned"),  # BEFORE, AFTER
#     test_kind="wilcoxon",                       # or "ttest"
#     fisher_z=fisher_z,
#     show=False,
#     filename_stub=stub + f"__avg_over_desc_max_p_{p_val}_{fisher_z}",
#     save_dir=out_dir,
#     error="sem",
#      pval_thresh=p_val,
     
# )


            # plot_avg_over_descriptors_by_participant_grid(
            # df_all_max,
            # value_col="correlation",
            # target_col="target",
            # pid_col="participant_id",
            # model_col="model",
            # type_col="type",
            # agg="mean",
            # error="sem",
            # min_targets=3,
            # save_dir=out_dir,
            # filename_stub=stub + f"__avg_over_desc_max_{p_val}",
            # show=False,
           
            # )  





df_fmri = load_all_metrics(f"{base_dir}/Sep16_fmrimetrics_1")
df_fmri['type'] = 'frozen'
print("df_fmri:", df_fmri.columns.values.tolist(), df_fmri.shape)


df_fmri_tuned = load_tuned_metrics2(f"{base_dir}/Sep16_fmrifinetuned_metrics_1")
df_fmri_tuned['type'] = 'fine-tuned'
print("df_fmri_tuned:", df_fmri_tuned.columns.values.tolist(), df_fmri.shape)


issues_behavior,df_behavior_clean = audit_metrics(df_fmri, name="df_fmri", tuned=False)
issues_beh_tuned,df_behaviortuned_clean = audit_metrics(df_fmri_tuned, name="df_fmrituned", tuned=True)

