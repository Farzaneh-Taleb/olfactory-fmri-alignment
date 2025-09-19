import os, sys, re, math, ast
from pathlib import Path
from typing import Dict, List, Optional, Union
from pandas.api.types import is_categorical_dtype


from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", None)  # don't truncate long strings
pd.set_option("display.width", 0)            # don't wrap to fit console width
from .model_config import MODELS, LAYERS_END, ROIS, P_VALUES  # noqa: F401  (imported but not used)
from collections import OrderedDict
from scipy.stats import wilcoxon, ttest_rel
# --- Project imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
USECOLS = [
    "target_id","correlation","p_value_correlation","model","ds","participant_id",
    "layer","n_fold","n_components","target","roi","tr","run_id","source_file",
    "row_idx","unfreeze_last_n","behavior_embeddings"
    # NOTE: no "date"
    # add "unfreeze_last_n", "behavior_embeddings" for tuned dir if not already in CSVs
]
DTYPES = {
    "target_id": "int32",
    "correlation": "float32",
    "p_value_correlation": "float32",
    "model": "string",
    "ds": "string",
    "participant_id": "int16",
    "layer": "int16",
    "n_fold": "int8",
    "target": "string",
    "roi": "string",
    "tr": "float32",
    # tuned-only:
    
    "behavior_embeddings": "string",
}
def print_unobserved_categories(df: pd.DataFrame, cols):
    """
    For each categorical column in `cols`, print the category levels
    that are defined but not observed in the data.
    """
    any_unobserved = False
    for c in cols:
        if pd.api.types.is_categorical_dtype(df[c]):
            cats = df[c].cat.categories
            obs = pd.Index(df[c].dropna().unique())
            unobs = cats.difference(obs)
            if len(unobs) > 0:
                any_unobserved = True
                print(f"[DEBUG] Unobserved categories in '{c}': {list(unobs)}")
    if not any_unobserved:
        print("[DEBUG] No unobserved categories detected among", list(cols))
def print_unobserved_group_combinations(df: pd.DataFrame, cols):
    """
    Show unobserved *combinations* induced by categorical columns.
    Only categorical columns contribute to the cartesian product.
    """
    cat_cols = [c for c in cols if pd.api.types.is_categorical_dtype(df[c])]
    if not cat_cols:
        print("[DEBUG] No categorical group-by columns; no unobserved combinations.")
        return
    levels = [df[c].cat.categories for c in cat_cols]
    full = pd.MultiIndex.from_product(levels, names=cat_cols)
    observed = pd.MultiIndex.from_frame(df[cat_cols].dropna().drop_duplicates())
    missing = full.difference(observed)
    if len(missing) > 0:
        print(f"[DEBUG] {len(missing)} unobserved categorical group combinations (showing up to 20):")
        for tup in list(missing)[:20]:
            print("   ", dict(zip(cat_cols, tup)))
    else:
        print("[DEBUG] No unobserved categorical group combinations.")
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
    # <<< NEW: include participant_source_id and n_components
    for icol in ["participant_id","layer","n_fold","target_id","n_components","participant_source_id"]:
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
                            "unfreeze_last_n","behavior_embeddings","n_components","target_id","cid","z_score",
                            "participant_source_id"] if c in d.columns]  # <<< NEW
    

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
def audit_metrics_fmri(df: pd.DataFrame, name="df_fmri", tuned=False):
    """
    Like audit_metrics, but fMRI-aware:
      • 'z_score' is expected ALSO for untuned fMRI.
      • Checks ROI membership (if ROIS is available).
      • 'tr' must be > 0 (if provided).
      • Prints original row numbers for bad rows.
    """
    from collections import OrderedDict
    issues: dict[str, pd.DataFrame] = OrderedDict()

    base_expected = {
        "target_id","correlation","mse","p_value_correlation","p_value_mse",
        "model","ds","participant_id","layer","n_fold",
        "target","type","z_score","roi","tr"
    }
    tuned_extra = {"behavior_embeddings","unfreeze_last_n"}
    expected = base_expected | (tuned_extra if tuned else set())

    missing = sorted(expected - set(df.columns))
    if missing:
        print(f"[{name}] Missing columns: {missing}")

    d = df.copy()

    # --- Keep original row numbers (0-based) and CSV line numbers (header counted as line 1) ---
    if "row_idx" in d.columns:
        base_row = pd.to_numeric(d["row_idx"], errors="coerce")
        fallback = pd.Series(np.arange(len(d)), index=d.index)
        d["_row0"] = base_row.fillna(fallback).astype(int)
    else:
        # Position-based index (0,1,2,...) even if df.index is custom
        d["_row0"] = np.arange(len(d)).astype(int)

    d["_csv_row"] = d["_row0"] + 2  # +1 to make 1-based, +1 for header line

    def bad_numeric(col, *, min_val=None, max_val=None, positive_only=False, allow_nan=False):
        if col not in d.columns:
            return pd.DataFrame(index=[])
        vals = pd.to_numeric(d[col], errors="coerce")
        bad = ~np.isfinite(vals)
        if allow_nan:
            bad &= ~vals.isna()
        if min_val is not None:
            bad |= vals < min_val
        if max_val is not None:
            bad |= vals > max_val
        if positive_only:
            bad |= vals <= 0
        return d[bad]

    # Core metric sanity
    issues["bad_correlation"] = bad_numeric("correlation", min_val=-1.0, max_val=1.0)
    issues["bad_mse"]         = bad_numeric("mse", min_val=0.0)
    for pcol in ["p_value_correlation","p_value_mse"]:
        if pcol in d.columns:
            issues[f"bad_{pcol}"] = bad_numeric(pcol, min_val=0.0, max_val=1.0)

    # IDs
    for icol in ["participant_id","layer","n_fold","target_id"]:
        if icol in d.columns:
            issues[f"bad_{icol}"] = bad_numeric(icol)

    # Model + layer range
    if "model" in d.columns:
        issues["bad_model_nan_or_blank"] = d[d["model"].isna() | d["model"].map(_is_blank_str)]
        if MODEL_MAX:
            layers = pd.to_numeric(d.get("layer", np.nan), errors="coerce")
            max_allowed = d["model"].map(MODEL_MAX).astype("float")
            out_of_range = d[(~layers.isna()) & (~max_allowed.isna()) & ((layers < 0) | (layers > max_allowed))]
            issues["bad_layer_out_of_range"] = out_of_range

    # Text sanity
    for scol in ["ds","target","type"]:
        if scol in d.columns:
            issues[f"bad_{scol}_empty"] = d[d[scol].isna() | d[scol].map(_is_blank_str)]

    # fMRI-specific: roi, tr
    if "roi" in d.columns:
        try:
            valid_rois = set(ROIS)
            issues["bad_roi_value"] = d[~d["roi"].isin(valid_rois)]
        except Exception:
            issues["bad_roi_empty"] = d[d["roi"].isna() | d["roi"].map(_is_blank_str)]

    if "tr" in d.columns:
        issues["bad_tr"] = bad_numeric("tr")

    # Tuned-specific checks
    if tuned:
        if "unfreeze_last_n" in d.columns:
            coerced_unf = d["unfreeze_last_n"].apply(_coerce_int)
            bad_unf = d[
                coerced_unf.isna()
                & ~d["unfreeze_last_n"].isna()
                & ~(d["unfreeze_last_n"].astype(str).str.strip().str.lower()
                      .isin(["adaptive","all","none","nan","null",""]))
            ]
            issues["bad_unfreeze_last_n"] = bad_unf

        if "behavior_embeddings" in d.columns:
            parsed = d["behavior_embeddings"].apply(_beh_tuple)
            src = d["behavior_embeddings"].astype(str).str.strip().str.lower()
            bad_beh_parse = d[(parsed.isna()) & ~(src.isin(["", "none", "nan", "null"]))]  # parse failed but not blank
            issues["bad_behavior_embeddings_parse"] = bad_beh_parse

    # Keep only non-empty issue tables
    issues = {k: v for k, v in issues.items() if not v.empty}

    # Summary + preview of where bad rows came from
    print(f"\n=== Audit summary (fMRI): {name} ===")
    if not issues:
        print("No obvious bad rows found ✅")
    else:
        for k, baddf in issues.items():
            print(f"- {k}: {len(baddf)} rows")
            cols = [c for c in ["source_file","_row0","_csv_row","model","participant_id","target","layer"] if c in baddf.columns]
            if cols:
                print(baddf[cols].head())

    return issues, d

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
    for col in ("layer", "participant_id", "n_fold", "target_id", "participant_source_id"):  # <<< NEW
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors="coerce")

    if "z_score" in f.columns:
        f["z_score"] = f["z_score"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    # if "behavior_embeddings" in f.columns 
    #     f["_beh"] = f["behavior_embeddings"].map(_norm_beh)

    for key, val in (filters or {}).items():
        # if key in ("behavior_embeddings",):
        #     col = "_beh"
        #     if _is_na_value(val):
        #         f = f[f[col].apply(lambda t: len(t) == 0)]
        #     else:
        #         f = f[f[col] == _norm_beh(val)]
        #     continue

        if key not in f.columns:
            raise KeyError(f"Filter column '{key}' not found.")

        vals = val if isinstance(val, (list, tuple, set)) else [val]
        want_na = any(_is_na_value(v) for v in vals)
        non_na_vals = [v for v in vals if not _is_na_value(v)]

        # Force numeric coercion for numeric-like cols (including n_components) even if dtype is object
        numeric_like = {"layer", "participant_id", "n_fold", "target_id", "participant_source_id"}  # <<< NEW
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


def load_metrics_generic(path: Union[str, Path], **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a metrics CSV with columns present in the file.
    Adds source_file + row_idx. Does NOT add model/ds/run_id from filename.
    Use when filenames don't follow the old patterns or when new columns appear.

    Extra kwargs are passed directly to pandas.read_csv, e.g.:
        dtype={...}, usecols=[...], low_memory=False
    """
    path = Path(path)
    df = pd.read_csv(path, **read_csv_kwargs)
    df["source_file"] = path.name
    df["row_idx"] = df.index
    return df
def load_any_metrics_in_dir(directory: Union[str, Path], **read_csv_kwargs) -> pd.DataFrame:
    """
    Load all CSVs in a directory with control over pd.read_csv options.
    Example:
        load_any_metrics_in_dir("dir", usecols=[...], dtype={...}, low_memory=False)
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {directory}")
    dfs = [load_metrics_generic(f, **read_csv_kwargs) for f in files]
    return pd.concat(dfs, ignore_index=True)
def load_metrics_fast(dirpath: str) -> pd.DataFrame:
    # Peek header to decide which columns are present
    sample = next(iter(sorted(Path(dirpath).glob("*.csv"))))
    header = pd.read_csv(sample, nrows=0).columns.tolist()
    print("header", header)
    cols_to_read = [c for c in USECOLS if c in header]  # keep tuned cols if present
    dtypes_local = {k: v for k, v in DTYPES.items() if k in cols_to_read}

    df = load_any_metrics_in_dir(
        dirpath,
        usecols=cols_to_read,
        dtype=dtypes_local,
        low_memory=False,
    )

    print(df.columns.values.tolist())

    # Basic cleaning
    df["tr"] = pd.to_numeric(df.get("tr"), errors="coerce").astype("float32")
    if "z_score" in df.columns:
        df["z_score"] = (df["z_score"].astype(str).str.strip().str.lower()
                         .map({"true": True, "1": True, "t": True,
                               "false": False, "0": False, "f": False}))
    # # Keep ds now or later; up to you. Doing it here reduces downstream memory:
    # df = df[df["ds"] == "sagar2023"].copy()

    # Categorical compression
    for col in ("model","ds","roi","target","behavior_embeddings"):
        if col in df.columns and df[col].dtype.name == "string":
            df[col] = df[col].astype("category")

    # Optional: coerce unfreeze to numeric where possible


    return df
def slug(x) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(x)).strip("-").lower()
def add_bar_group_column(df: pd.DataFrame,
                         type_col: str = "type",
                         source_col: str = "participant_source_id",
                         out_col: str = "bar_group") -> pd.DataFrame:
    d = df.copy()
    if source_col in d.columns:
        src = pd.to_numeric(d[source_col], errors="coerce")
        labels = np.where(src.notna(), "source-" + src.astype(int).astype(str), d.get(type_col))
        d[out_col] = labels
    else:
        d[out_col] = d.get(type_col)
    return d

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
    print_unobserved_categories(df, group_cols)
    print_unobserved_group_combinations(df, group_cols)
    # idx of max corr per group
    idx = df.groupby(list(group_cols), dropna=False,observed=True)[corr_col].idxmax()
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
    bar_group_col: str | None = None,    # <<< NEW
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
    cat_col = bar_group_col if bar_group_col else type_col  # <<< NEW
    needed_cols = {corr_col, target_col, pid_col, model_col}
    if cat_col:
        needed_cols.add(cat_col)
    missing = needed_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_all: {missing}")

    # Orders
    type_order = sorted(df[cat_col].dropna().unique().tolist()) if cat_col in df.columns else []  # <<< NEW
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

            # Draw grouped bars by cat_col (type/source)
            if n_types > 0:
                for i, tname in enumerate(type_order):
                    heights = []
                    for tgt in targets:
                        val = (d[(d[target_col] == tgt) & (d[cat_col] == tname)][corr_col]
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
            legend_title = bar_group_col or type_col  # <<< NEW
            fig.legend(handles, labels, title=legend_title, loc="upper center",
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
        out = Path(save_dir) / f"{slug(stub)}{ptag}.png"
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
    bar_group_col: str | None = None,           # <<< NEW
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

    cat_col = bar_group_col if bar_group_col else type_col  # <<< NEW
    needed = {corr_col, target_col, pid_col, model_col, layer_at_max_col}
    if cat_col: needed.add(cat_col)
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_max: {missing}")

    type_order = sorted(df[cat_col].dropna().unique().tolist())  # <<< NEW
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

            # grouped bars by cat_col
            for i, tname in enumerate(type_order):
                heights, layers = [], []
                for tgt in targets:
                    row = d[(d[target_col] == tgt) & (d[cat_col] == tname)]
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
            fig.legend(handles, labels, title=(bar_group_col or type_col), loc="upper center", ncol=max(1, len(labels)))  # <<< NEW
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
        out = Path(save_dir) / f"{slug(stub)}{ptag}.png"
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
from pandas.api.types import is_categorical_dtype
from scipy.stats import ttest_rel, wilcoxon

def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in str(s)).strip("-")

def plot_avg_over_descriptors_by_participant_grid(
    df_all: pd.DataFrame,
    *,
    value_col: str = "correlation",
    target_col: str = "target",
    pid_col: str = "participant_id",
    model_col: str = "model",
    type_col: str = "type",
    bar_group_col: str | None = None,     # bars split by this (default = type_col)
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
    # testing options
    compare_types: tuple[str, str] | None = None,  # e.g., ("frozen","fine-tuned")
    test_kind: str = "wilcoxon",                   # "wilcoxon" or "ttest"
    fisher_z: bool = True,                         # apply Fisher z before averaging & tests
    star_levels: tuple[float, float, float] = (0.05, 0.01, 0.001),
    alternative: str = "greater"
):
    """
    Plot average metric over all descriptors (targets) for each (participant, model),
    with bars split by `bar_group_col` (default: `type_col`). Adds per-subject paired
    significance between two `type_col` levels, and a group-level test across subjects
    per model on (after - before) deltas.

    Assumes exactly two comparison types are provided via `compare_types` (levels of `type_col`).
    """

    # ---------- Validate inputs ----------
    d = df_all.copy()

    # Optional p-value filtering
    if pval_col is not None and pval_col in d.columns and pval_thresh is not None:
        d = d[d[pval_col] <= pval_thresh].copy()

    cat_col = bar_group_col if bar_group_col else type_col
    needed = {value_col, target_col, pid_col, model_col}
    if type_col: needed.add(type_col)   # tests pivot on type_col
    if cat_col: needed.add(cat_col)     # drawing split
    missing = needed - set(d.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Clean numeric series only (avoid frame-wide downcasting warning)
    s = pd.to_numeric(d[value_col], errors="coerce")
    s = s.where(np.isfinite(s))
    d[value_col] = s.astype("float32")
    d = d.dropna(subset=[value_col])

    # Normalize categoricals: remove unused categories on grouping keys
    for k in {pid_col, model_col, type_col, target_col, cat_col}:
        if k in d and is_categorical_dtype(d[k]):
            d[k] = d[k].cat.remove_unused_categories()
    # Make model a plain string (avoids categorical expansion noise)
    d[model_col] = d[model_col].astype("string")

    # Decide which two types to compare (levels of type_col)
    all_types = d[type_col].dropna().unique().tolist()
    if compare_types is None:
        if len(all_types) != 2:
            raise ValueError(
                f"Expected exactly two {type_col} levels for comparison, found {len(all_types)}: {all_types}. "
                f"Pass them via compare_types=('before','after'), etc."
            )
        compare_types = (all_types[0], all_types[1])
    else:
        for t in compare_types:
            if t not in all_types:
                raise ValueError(f"compare_types includes '{t}' which is not in {type_col} levels: {all_types}")
    t_before, t_after = compare_types

    # Fisher z transform
    def fisher_z_fn(r: pd.Series | np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    # ---------- Build two views ----------
    # d_plot: collapse to one row per (pid, model, cat_col, target) for drawing bars
    d_plot = (
        d.groupby([pid_col, model_col, cat_col, target_col], observed=True, dropna=False, as_index=False)[value_col]
         .mean()
    )
    # d_test: collapse to one row per (pid, model, type_col, target) for tests
    d_test = (
        d.groupby([pid_col, model_col, type_col, target_col], observed=True, dropna=False, as_index=False)[value_col]
         .mean()
    )

    # Add stat values (Fisher z or raw)
    if fisher_z:
        d_plot["_val_for_stats"] = fisher_z_fn(d_plot[value_col].values)
        d_test["_val_for_stats"] = fisher_z_fn(d_test[value_col].values)
        val_for_plot = "_val_for_stats"
        y_label_metric = f"{agg}(Fisher-z({value_col}))"
    else:
        d_plot["_val_for_stats"] = d_plot[value_col].values
        d_test["_val_for_stats"] = d_test[value_col].values
        val_for_plot = value_col
        y_label_metric = f"{agg}({value_col})"

    # Aggregate across targets for drawing (per subject×model×cat_col)
    def _sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        n = len(x)
        return float(x.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")

    agg_fun = "mean" if agg == "mean" else "median"
    summary = (
        d_plot.groupby([pid_col, model_col, cat_col], observed=True, dropna=False)[val_for_plot]
              .agg(n_targets="count",
                   val=agg_fun,
                   std=lambda x: pd.to_numeric(x, errors="coerce").std(ddof=1),
                   sem=_sem)
              .reset_index()
    )
    # Filter groups with too few targets
    summary = summary[summary["n_targets"] >= int(min_targets)].reset_index(drop=True)

    # Grid axes
    models = sorted(summary[model_col].dropna().unique().tolist())
    pids   = sorted(summary[pid_col].dropna().unique().tolist())
    cat_order = sorted(summary[cat_col].dropna().unique().tolist())

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

    x = np.arange(len(cat_order))
    bar_width = 0.7
    err_key = error if error in {"std", "sem"} else None

    # ---------- Group-level test per model across subjects ----------
    # For each model: compute per-subject mean delta (after - before) over targets, then Wilcoxon/ttest vs 0
    group_pvals_by_model: dict[str, float | None] = {}
    for model in models:
        deltas = []
        for pid in pids:
            sub = d_test[(d_test[pid_col] == pid) & (d_test[model_col] == model)].copy()
            # paired by target on type_col
            piv = sub.pivot(index=target_col, columns=type_col, values="_val_for_stats")
            if t_before in piv.columns and t_after in piv.columns:
                paired = piv[[t_before, t_after]].dropna()
                if len(paired) >= max(2, min_targets):
                    deltas.append((paired[t_after] - paired[t_before]).mean())
        if len(deltas) >= 2:
            if test_kind.lower() == "ttest":
                _, p = ttest_rel(deltas, np.zeros_like(deltas), alternative=alternative)
            else:
                try:
                    _, p = wilcoxon(deltas, zero_method="wilcox", alternative=alternative, mode="auto")
                except ValueError:
                    p = np.nan
        else:
            p = np.nan
        group_pvals_by_model[model] = p

    def p_to_stars(p: float | None) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)): return ""
        a, b, c = star_levels
        if p <= c: return "***"
        if p <= b: return "**"
        if p <= a: return "*"
        return ""

    # ---------- Draw bars + per-subject paired stars ----------
    for c, model in enumerate(models):
        for r, pid in enumerate(pids):
            ax = axes[r, c]
            sl = summary[(summary[pid_col] == pid) & (summary[model_col] == model)]
            if sl.empty:
                ax.axis("off")
                ax.set_title(f"pid={pid}, model={model}\n(no data)")
                continue

            y, yerr = [], []
            for t in cat_order:
                row = sl[sl[cat_col] == t]
                if row.empty:
                    y.append(0.0); yerr.append(0.0)
                else:
                    y.append(float(row["val"].iloc[0]))
                    yerr.append(float(row[err_key].iloc[0]) if err_key else 0.0)

            bars = ax.bar(x, y, width=bar_width,
                          yerr=(yerr if err_key else None),
                          capsize=3 if err_key else 0)

            # annotate
            for rect, val in zip(bars, y):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height(),
                        f"{val:.2f}",
                        ha="center", va="bottom", fontsize=9)
                
                ax.legend()

            # per-subject paired test (uses d_test on type_col)
            sub_full = d_test[(d_test[pid_col] == pid) & (d_test[model_col] == model)]
            p_pair = np.nan
            piv = sub_full.pivot(index=target_col, columns=type_col, values="_val_for_stats")
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

            star = p_to_stars(p_pair)
            if star and len(cat_order) >= 2 and (compare_types[0] in cat_order) and (compare_types[1] in cat_order):
                i0 = cat_order.index(compare_types[0])
                i1 = cat_order.index(compare_types[1])
                y_max = max(y[i0], y[i1])
                y_min = min(y[i0], y[i1])
                h = 0.04 * (abs(y_max) + abs(y_min) + 1e-6)
                y_bar = y_max + 1.5 * h
                ax.plot([x[i0], x[i0], x[i1], x[i1]],
                        [y_bar, y_bar + h, y_bar + h, y_bar], linewidth=1.0)
                ax.text((x[i0] + x[i1]) / 2, y_bar + h, star, ha="center", va="bottom", fontsize=11)

            # titles/labels
            ax.set_title(f"pid={pid}")
            ax.set_xticks(x, [str(t) for t in cat_order], rotation=0)
            if c == 0:
                ax.set_ylabel(f"{y_label_metric} over {target_col}s")

        # group-level star in column header
        gstar = p_to_stars(group_pvals_by_model.get(model, np.nan))
        axes[0, c].set_title(f"{model} {gstar}".rstrip(), fontweight="bold")

    fig.suptitle(
        f"Average over descriptors — bars={(bar_group_col or type_col)} • agg={agg}",
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


# def plot_avg_over_descriptors_by_participant_grid(
#     df_all: pd.DataFrame,
#     *,
#     value_col: str = "correlation",
#     target_col: str = "target",
#     pid_col: str = "participant_id",
#     model_col: str = "model",
#     type_col: str = "type",
#     bar_group_col: str | None = None,     # <<< NEW
#     pval_col: str | None = None,
#     pval_thresh: float | None = None,
#     agg: str = "mean",           # "mean" or "median"
#     error: str | None = "sem",   # "sem", "std", or None
#     min_targets: int = 1,
#     figsize_per_sub=(4.8, 3.2),
#     sharey: bool = True,
#     save_dir: str | None = None,
#     filename_stub: str | None = None,
#     show: bool = False,
#     # --- NEW: testing options ---
#     compare_types: tuple[str, str] | None = None,  # e.g., ("frozen","behavior_tuned")
#     test_kind: str = "wilcoxon",                   # "wilcoxon" or "ttest"
#     fisher_z: bool = True,                         # apply Fisher z to correlations for tests & summaries
#     star_levels: tuple[float, float, float] = (0.05, 0.01, 0.001),
#     alternative="greater"
# ):
#     """
#     Plot average metric over all descriptors (targets) for each (participant, model),
#     with bars split by `type`. Each bar is annotated with its average value.
#     Additionally:
#       • Per-subject paired test across descriptors between two `type`s (before vs after),
#         with significance stars drawn between the two bars.
#       • Group-level test across subjects per model on subject-wise mean improvements (after - before)
#         with stars appended to the column (model) title.

#     Assumes exactly two comparison types are provided via `compare_types`.
#     """

#     d = df_all.copy()

#     # Optional p-value filtering
#     if pval_col is not None and pval_col in d.columns and pval_thresh is not None:
#         d = d[d[pval_col] <= pval_thresh].copy()

#     # Basic checks
#     cat_col = bar_group_col if bar_group_col else type_col  # <<< NEW
#     needed = {value_col, target_col, pid_col, model_col}
#     if type_col: needed.add(type_col)   # for tests
#     if cat_col: needed.add(cat_col)     # for drawing
#     missing = needed - set(d.columns)
#     if missing:
#         raise KeyError(f"Missing required columns: {missing}")

#     # Ensure numeric correlations and within [-1, 1]
#     d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
#     d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

#     # Decide which two types to compare
#     all_types = d[type_col].dropna().unique().tolist()
#     if compare_types is None:
#         if len(all_types) != 2:
#             raise ValueError(
#                 f"Expected exactly two {type_col} levels for comparison, found {len(all_types)}: {all_types}. "
#                 f"Pass them via compare_types=('before','after') etc."
#             )
#         compare_types = (all_types[0], all_types[1])
#     else:
#         for t in compare_types:
#             if t not in all_types:
#                 raise ValueError(f"compare_types includes '{t}' which is not in {type_col} levels: {all_types}")

#     t_before, t_after = compare_types

#     # Fisher z transform helper
#     def fisher_z_fn(r: pd.Series | np.ndarray) -> np.ndarray:
#         r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
#         return 0.5 * np.log((1 + r) / (1 - r))

#     # Collapse duplicates to one row per (pid, model, cat_col, target) by mean
#     base_keys = [pid_col, model_col, cat_col, target_col]  # <<< NEW
#     d = (
#         d.groupby(base_keys, dropna=False, as_index=False)[value_col]
#          .mean()
#     )

#     # If Fisher z is requested, compute it for testing and for averaging
#     if fisher_z:
#         d["_val_for_stats"] = fisher_z_fn(d[value_col].values)
#         val_for_plot = "_val_for_stats"
#         y_label_metric = f"{agg}(Fisher-z({value_col}))"
#     else:
#         d["_val_for_stats"] = d[value_col].values
#         val_for_plot = value_col
#         y_label_metric = f"{agg}({value_col})"

#     # Aggregate across targets
#     def _sem(x: pd.Series) -> float:
#         x = pd.to_numeric(x, errors="coerce").dropna()
#         n = len(x)
#         return float(x.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")

#     agg_fun = "mean" if agg == "mean" else "median"
#     grouped = d.groupby([pid_col, model_col, cat_col], dropna=False)  # <<< NEW
#     summary = grouped[val_for_plot].agg(
#         n_targets="count",
#         val=agg_fun,
#         std=lambda x: pd.to_numeric(x, errors="coerce").std(ddof=1),
#         sem=_sem
#     ).reset_index()

#     # Filter groups with too few targets
#     summary = summary[summary["n_targets"] >= int(min_targets)].reset_index(drop=True)

#     # Setup grid
#     models = sorted(summary[model_col].dropna().unique().tolist())
#     pids   = sorted(summary[pid_col].dropna().unique().tolist())
#     cat_order = sorted(summary[cat_col].dropna().unique().tolist())  # <<< NEW

#     if not models or not pids:
#         print("[plot] Nothing to plot.")
#         return

#     nrows, ncols = len(pids), len(models)
#     fig_w = figsize_per_sub[0] * ncols
#     fig_h = figsize_per_sub[1] * nrows
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=sharey)

#     if nrows == 1 and ncols == 1:
#         axes = np.array([[axes]])
#     elif nrows == 1:
#         axes = np.array([axes])
#     elif ncols == 1:
#         axes = np.array([[ax] for ax in axes])

#     x = np.arange(len(cat_order))  # <<< NEW
#     bar_width = 0.7
#     err_key = error if error in {"std", "sem"} else None

#     # --- Group-level test per model (across subjects) ---
#     # Build subject-wise mean improvement (after - before) per model (tests use original type_col only)
#     group_pvals_by_model: dict[str, float | None] = {}
#     key4 = [pid_col, model_col, type_col, target_col]
#     key3 = [target_col, type_col, pid_col]
#     for model in models:
#         print(f"\n[Testing] model: {model}")    
#         deltas = []
#         for pid in pids:
#             # pivot by target to ensure pairing, using the original type_col
#             keys = [pid_col, model_col, type_col, target_col]
#             sub = df_all[(df_all[pid_col] == pid) & (df_all[model_col] == model)].copy()
#             print(sub[model_col].unique(), "sub shape:", sub.shape)
#             print("models present (counts):\n", sub[model_col].value_counts(dropna=False))

#             # 2) Kill categorical surprises on ALL grouping keys
#             for k in {pid_col, model_col, type_col, target_col}:
#                 if k in sub and is_categorical_dtype(sub[k]):
#                     sub[k] = sub[k].cat.remove_unused_categories()

#             # (Optional) make sure model is a plain string, not category
#             sub[model_col] = sub[model_col].astype("string")

#             print("models present (counts):\n", sub[model_col].value_counts(dropna=False))

#             # 3) Clean + aggregate with observed=True
#             sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
#             sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

#             sub = (
#                 sub.groupby(key4, observed=True, dropna=False, as_index=False)[value_col]
#                    .mean()
#             )

#             # 4) Sanity: how many models per (target,type,pid)?
#             nm = (
#                 sub.groupby(key3, observed=True, dropna=False)[model_col]
#                    .nunique()
#                    .rename("n_models")
#                    .reset_index()
#             )
#             print("Any combos with >1 model?\n", nm.query("n_models > 1").head())

#             # 5) Your dup checks (now with observed=True)
#             dups = (
#                 sub.groupby(key4, observed=True, dropna=False)
#                    .size().reset_index(name="n").query("n > 1")
#             )
#             print("Duplicate (pid, model, type, target) combos:\n", dups.head(10))

#             pivot_dups = (
#                 sub.groupby([target_col, type_col, pid_col, model_col], observed=True, dropna=False)
#                    .size().reset_index(name="n").query("n > 1")
#             )
#             print("Duplicate (target, type, pid, model) for pivot:\n", pivot_dups.head(10))

#             pivot_dups_3 = (
#                 sub.groupby(key3, observed=True, dropna=False)
#                    .size().reset_index(name="n").query("n > 1")
#             )
#             print("Duplicate (target, type, pid) for pivot:\n", pivot_dups_3.head(10))
#             ##############


       



#             # Collapse duplicates within (pid, model, type, target)
#             sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
#             # sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
#             sub = (sub.groupby([pid_col, model_col, type_col, target_col], dropna=False, as_index=False)[value_col]
#                         .mean())
#             if fisher_z:
#                 sub["_val_for_stats"] = fisher_z_fn(sub[value_col].values)
#             else:
#                 sub["_val_for_stats"] = sub[value_col].values
            
#                  ####
#             dups = (sub.groupby(keys, dropna=False)
#             .size()
#             .reset_index(name="n")
#             .query("n > 1"))
#             print("Duplicate (pid, model, type, target) combos:\n", dups.head(20))

#             pivot_keys = [target_col, type_col, pid_col,model_col]
#             pivot_dups = (sub.groupby(pivot_keys, dropna=False)
#               .size()
#               .reset_index(name="n")
#               .query("n > 1"))
            
#             print(f"Duplicate (target, type, pid_col, model_col) combos for pivot:\n{pivot_dups.head(10)}")



#             pivot_keys = [target_col, type_col, pid_col]
#             pivot_dups = (sub.groupby(pivot_keys, dropna=False)
#               .size()
#               .reset_index(name="n")
#               .query("n > 1"))
            
#             print(f"Duplicate (target, type, pid_col) combos for pivot:\n{pivot_dups.head(10)}")


            

#             # For the first offending combo, show which *other* columns vary
#             if not dups.empty:
#                 k = dups.iloc[0][keys].to_dict()
#                 rows = sub[(sub[pid_col] == k[pid_col]) &
#                            (sub[model_col] == k[model_col]) &
#                            (sub[type_col] == k[type_col]) &
#                            (sub[target_col] == k[target_col])]
#                 varying = {c: rows[c].nunique() for c in rows.columns if c not in keys}
#                 print("Columns varying within that key:\n",
#                       {c: n for c, n in varying.items() if n > 1})
#             ####
#             piv = sub.pivot(index=target_col, columns=type_col, values="_val_for_stats")
#             if t_before in piv.columns and t_after in piv.columns:
#                 paired = piv[[t_before, t_after]].dropna()
#                 if len(paired) >= max(2, min_targets):
#                     delta = (paired[t_after] - paired[t_before]).mean()
#                     deltas.append(delta)
#         if len(deltas) >= 2:
#             if test_kind.lower() == "ttest":
#                 _, p = ttest_rel(deltas, np.zeros_like(deltas), alternative=alternative)
#             else:
#                 # Wilcoxon vs zero
#                 try:
#                     _, p = wilcoxon(deltas, zero_method="wilcox", alternative=alternative, mode="auto")
#                 except ValueError:
#                     p = np.nan
#         else:
#             p = np.nan
#         group_pvals_by_model[model] = p

#     def p_to_stars(p: float | None) -> str:
#         if p is None or np.isnan(p):
#             return ""
#         a, b, c = star_levels
#         if p <= c: return "***"
#         if p <= b: return "**"
#         if p <= a: return "*"
#         return ""

#     # Draw bars + per-subject test stars
#     for c, model in enumerate(models):
#         for r, pid in enumerate(pids):
#             ax = axes[r, c]
#             sl = summary[(summary[pid_col] == pid) & (summary[model_col] == model)]
#             if sl.empty:
#                 ax.axis("off")
#                 ax.set_title(f"pid={pid}, model={model}\n(no data)")
#                 continue

#             y, yerr = [], []
#             for t in cat_order:  # <<< NEW
#                 row = sl[sl[cat_col] == t]
#                 if row.empty:
#                     y.append(0.0); yerr.append(0.0)
#                 else:
#                     y.append(float(row["val"].iloc[0]))
#                     yerr.append(float(row[err_key].iloc[0]) if err_key else 0.0)

#             bars = ax.bar(x, y, width=bar_width,
#                           yerr=(yerr if err_key else None),
#                           capsize=3 if err_key else 0)

#             # Annotate each bar with its value
#             for rect, val in zip(bars, y):
#                 ax.text(
#                     rect.get_x() + rect.get_width() / 2,
#                     rect.get_height(),
#                     f"{val:.2f}",
#                     ha="center", va="bottom", fontsize=9
#                 )

#             # --- Per-subject paired test across descriptors (on type_col only) ---
#             sub_full = df_all[(df_all[pid_col] == pid) & (df_all[model_col] == model)].copy()
#             sub_full[value_col] = pd.to_numeric(sub_full[value_col], errors="coerce")
#             sub_full = sub_full.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
#             sub_full = (sub_full.groupby([pid_col, model_col, type_col, target_col], dropna=False, as_index=False)[value_col]
#                                 .mean())
#             if fisher_z:
#                 sub_full["_val_for_stats"] = fisher_z_fn(sub_full[value_col].values)
#             else:
#                 sub_full["_val_for_stats"] = sub_full[value_col]

#             piv = sub_full.pivot(index=target_col, columns=type_col, values="_val_for_stats")
#             p_pair = np.nan
#             if t_before in piv.columns and t_after in piv.columns:
#                 paired = piv[[t_before, t_after]].dropna()
#                 if len(paired) >= max(2, min_targets):
#                     if test_kind.lower() == "ttest":
#                         _, p_pair = ttest_rel(paired[t_after].values, paired[t_before].values, alternative=alternative)
#                     else:
#                         try:
#                             _, p_pair = wilcoxon(
#                                 paired[t_after].values,
#                                 paired[t_before].values,
#                                 zero_method="wilcox",
#                                 alternative=alternative,
#                                 mode="auto"
#                             )
#                         except ValueError:
#                             p_pair = np.nan

#             # Draw significance bar if significant
#             star = p_to_stars(p_pair)
#             if star and len(cat_order) >= 2 and compare_types[0] in cat_order and compare_types[1] in cat_order:
#                 # position above the taller of the two bars we compare (map indices by label)
#                 i0 = cat_order.index(compare_types[0])
#                 i1 = cat_order.index(compare_types[1])
#                 y_max = max(y[i0], y[i1])
#                 y_min = min(y[i0], y[i1])
#                 h = 0.04 * (abs(y_max) + abs(y_min) + 1e-6)  # small height
#                 y_bar = y_max + 1.5 * h
#                 ax.plot([x[i0], x[i0], x[i1], x[i1]], [y_bar, y_bar + h, y_bar + h, y_bar], linewidth=1.0)
#                 ax.text(np.mean([x[i0], x[i1]]), y_bar + h, star, ha="center", va="bottom", fontsize=11)

#             # Titles & labels
#             ax.set_title(f"pid={pid}")
#             ax.set_xticks(x, [str(t) for t in cat_order], rotation=0)  # <<< NEW
#             if c == 0:
#                 ax.set_ylabel(f"{y_label_metric} over {target_col}s")

#         # Add group-level star to the column header (model)
#         gstar = p_to_stars(group_pvals_by_model.get(model, np.nan))
#         # Put a bold model title with group-level stars centered above the column
#         axes[0, c].set_title(f"{model} {gstar}".rstrip(), fontweight="bold")

#     fig.suptitle(
#         f"Average over descriptors — bars={(bar_group_col or type_col)} • agg={agg}",  # <<< keep info
#         y=0.98
#     )
#     fig.tight_layout(rect=[0, 0, 1, 0.95])

#     if save_dir:
#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         stub = filename_stub or "avg_over_descriptors_participant_x_model_with_tests"
#         out = Path(save_dir) / f"{slug(stub)}.png"
#         fig.savefig(out, dpi=200, bbox_inches="tight")
#         print("saved:", out)
#         plt.close(fig)
#     elif show:
#         plt.show()
#     else:
#         plt.show()


def plot_fmri_tr_lines_by_roi_subject(
    df_all: pd.DataFrame,
    *,
    value_col: str = "correlation",
    target_col: str = "target",
    pid_col: str = "participant_id",
    roi_col: str = "roi",
    tr_col: str = "tr",
    type_col: str = "type",
    bar_group_col: str | None = None,
    model_col: str = "model",
    pval_col: str | None = None,
    pval_thresh: float | None = None,
    fisher_z: bool = True,
    min_targets: int = 1,
    sharey: bool = True,
    figsize_per_sub=(4.2, 3.0),
    save_dir: str | None = None,
    filename_stub: str | None = None,
    show: bool = False,
):
    """
    Rows = ROIs, Cols = subjects. X-axis = TR (positive TRs only).
    Each subplot: n lines (one per type/bar_group), y = average over targets at each TR.
    If multiple models are present, it averages across models first for each (pid, roi, tr, type, target).
    """
    d = df_all.copy()

    # Keep positive TRs only
    d[tr_col] = pd.to_numeric(d[tr_col], errors="coerce")
    d = d[(d[tr_col] > 0) & d[tr_col].notna()].copy()

    # Optional p-value filter
    if pval_col is not None and pval_col in d.columns and pval_thresh is not None:
        d = d[d[pval_col] <= pval_thresh].copy()

    # Required cols check
    cat_col = bar_group_col if bar_group_col else type_col
    needed = {value_col, target_col, pid_col, roi_col, tr_col, cat_col}
    missing = needed - set(d.columns)
    if missing:
        raise KeyError(f"Missing columns for fMRI TR plot: {missing}")

    # Make numeric + sanitize
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    # If multiple models exist, average within (pid, roi, tr, cat, target) across models
    base_keys = [pid_col, roi_col, tr_col, cat_col, target_col]
    if model_col in d.columns:
        d = (d.groupby(base_keys, dropna=False, as_index=False)[value_col]
               .mean())

    # Fisher-z if requested (before averaging over targets)
    def fisher_z_fn(r: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    if fisher_z:
        d["_val_for_stats"] = fisher_z_fn(d[value_col].values)
        y_label_metric = "mean Fisher-z(correlation)"
        val_for_plot = "_val_for_stats"
    else:
        d["_val_for_stats"] = d[value_col].values
        y_label_metric = "mean correlation"
        val_for_plot = value_col

    # Average over targets at each (pid, roi, tr, cat)
    agg = (d.groupby([pid_col, roi_col, tr_col, cat_col], dropna=False)[val_for_plot]
             .agg(n_targets="count", mean="mean")
             .reset_index())
    agg = agg[agg["n_targets"] >= int(min_targets)].copy()

    # Grid axes
    rois = (list(ROIS) if "ROIS" in globals() and isinstance(ROIS, (list, tuple)) and len(ROIS) > 0
            else sorted(agg[roi_col].dropna().unique().tolist()))
    pids = sorted(agg[pid_col].dropna().unique().tolist())
    cats = sorted(agg[cat_col].dropna().unique().tolist())

    if not rois or not pids or not cats:
        print("[plot_fmri_tr_lines] Nothing to plot (check filters).")
        return

    nrows, ncols = len(rois), len(pids)
    fig_w = figsize_per_sub[0] * ncols
    fig_h = figsize_per_sub[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=sharey)

    # normalize axes shape
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # Global x ticks (all TRs present)
    all_trs = sorted(pd.to_numeric(agg[tr_col], errors="coerce").dropna().unique().tolist())

    # Plot lines
    for r, roi in enumerate(rois):
        for c, pid in enumerate(pids):
            ax = axes[r, c]
            sl = agg[(agg[roi_col] == roi) & (agg[pid_col] == pid)]
            if sl.empty:
                ax.axis("off")
                ax.set_title(f"{roi} | pid={pid}\n(no data)")
                continue

            # Ensure TRs are numeric + sorted
            trs_here = sorted(sl[tr_col].dropna().unique().tolist())
            x = trs_here  # TR values directly on x-axis

            # One line per category
            for cat in cats:
                dcat = sl[sl[cat_col] == cat].sort_values(tr_col)
                if dcat.empty:
                    continue
                ax.plot(dcat[tr_col].values, dcat["mean"].values, marker="o", label=str(cat))

            # cosmetics
            if r == nrows - 1:
                ax.set_xlabel("TR")
            if c == 0:
                ax.set_ylabel(y_label_metric)
            ax.set_title(f"{roi} | pid={pid}")
            ax.set_xticks(all_trs)

    # single legend
    first = axes[0, 0]
    handles, labels = first.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title=(bar_group_col or type_col), loc="upper center", ncol=max(1, len(labels)))
        fig.subplots_adjust(top=0.88)

    fig.suptitle("fMRI: average over targets across TRs — rows: ROI, cols: subject", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save / show
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stub = filename_stub or "fmri_tr_lines_by_roi_subject"
        out = Path(save_dir) / f"{slug(stub)}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.show()


def plot_fmri_layer_lines_trpairs_by_roi_subject(
    df_all: pd.DataFrame,
    *,
    value_col: str = "correlation",
    target_col: str = "target",
    pid_col: str = "participant_id",
    roi_col: str = "roi",
    tr_col: str = "tr",
    type_col: str = "type",               # expects at least {'frozen','fine-tuned'}
    model_col: str = "model",
    pval_col: str | None = None,
    pval_thresh: float | None = None,
    fisher_z: bool = True,                # Fisher-z before averaging over targets
    min_targets: int = 1,
    sharey: bool = True,
    figsize_per_sub=(4.2, 3.0),
    save_dir: str | None = None,
    filename_stub: str | None = None,
    show: bool = False,
):
    """
    Rows = ROIs, Cols = subjects. X-axis = LAYER (sorted, integers).
    For each TR in a subplot, draw TWO lines across layers:
       solid  = fine-tuned
       dashed = frozen
    Y is the mean (over targets) at each (pid, roi, tr, type, layer).
    If multiple models exist, averages across models first per (pid, roi, tr, type, layer, target).
    """
    d = df_all.copy()

    # Keep positive TRs
    d[tr_col] = pd.to_numeric(d[tr_col], errors="coerce")
    # d = d[(d[tr_col] > 0) & d[tr_col].notna()].copy()

    # Optional p-value filter
    if pval_col is not None and pval_col in d.columns and pval_thresh is not None:
        d = d[d[pval_col] <= pval_thresh].copy()

    # Required columns
    needed = {value_col, target_col, pid_col, roi_col, tr_col, type_col, "layer"}
    missing = needed - set(d.columns)
    if missing:
        raise KeyError(f"Missing columns for layer×TR plot: {missing}")

    # Sanitize numerics
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["layer"] = pd.to_numeric(d["layer"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col, "layer"])

    # Average across models if present so each sample is (pid, roi, tr, type, layer, target)
    base_keys = [pid_col, roi_col, tr_col, type_col, "layer", target_col]
    if model_col in d.columns:
        d = (d.groupby(base_keys, dropna=False, as_index=False)[value_col]
               .mean())

    # Fisher-z (before target averaging)
    def fisher_z_fn(r: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    if fisher_z:
        d["_val_for_stats"] = fisher_z_fn(d[value_col].values)
        y_label_metric = "mean Fisher-z(correlation)"
        val_for_plot = "_val_for_stats"
    else:
        d["_val_for_stats"] = d[value_col].values
        y_label_metric = "mean correlation"
        val_for_plot = value_col

    # Average over targets at each (pid, roi, tr, type, layer)
    agg = (d.groupby([pid_col, roi_col, tr_col, type_col, "layer"], dropna=False)[val_for_plot]
             .agg(n_targets="count", mean="mean")
             .reset_index())
    agg = agg[agg["n_targets"] >= int(min_targets)].copy()

    # Grid axes
    rois = (list(ROIS) if "ROIS" in globals() and isinstance(ROIS, (list, tuple)) and len(ROIS) > 0
            else sorted(agg[roi_col].dropna().unique().tolist()))
    pids = sorted(agg[pid_col].dropna().unique().tolist())
    trs_all = sorted(agg[tr_col].dropna().unique().tolist())

    if not rois or not pids or not trs_all:
        print("[plot_fmri_layer_lines_trpairs] Nothing to plot (check filters).")
        return

    # Prefer standard labels for style mapping
    types_present = sorted(agg[type_col].dropna().unique().tolist())
    # Map linestyles
    ls_map = {t: "--" for t in types_present}
    if "fine-tuned" in ls_map: ls_map["fine-tuned"] = "-"
    if "frozen" in ls_map:     ls_map["frozen"] = "--"

    nrows, ncols = len(rois), len(pids)
    fig_w = figsize_per_sub[0] * ncols
    fig_h = figsize_per_sub[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=sharey)

    # normalize axes shape
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # Colors cycle by TR
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    tr_to_color = {tr: color_cycle[i % len(color_cycle)] if color_cycle else None
                   for i, tr in enumerate(trs_all)}

    # Global x ticks (all layers present)
    all_layers = sorted(pd.to_numeric(agg["layer"], errors="coerce").dropna().unique().tolist())

    # Plot
    for r, roi in enumerate(rois):
        for c, pid in enumerate(pids):
            ax = axes[r, c]
            sl = agg[(agg[roi_col] == roi) & (agg[pid_col] == pid)]
            if sl.empty:
                ax.axis("off")
                ax.set_title(f"{roi} | pid={pid}\n(no data)")
                continue

            for tr in trs_all:
                dtr = sl[sl[tr_col] == tr]
                if dtr.empty:
                    continue
                for t in types_present:
                    dt = dtr[dtr[type_col] == t].sort_values("layer")
                    if dt.empty:
                        continue
                    ax.plot(
                        dt["layer"].values,
                        dt["mean"].values,
                        linestyle=ls_map.get(t, "-"),
                        marker="o",
                        label=f"{t} (TR={tr})",
                        color=tr_to_color[tr]
                    )

            # cosmetics
            ax.set_title(f"{roi} | pid={pid}")
            if r == nrows - 1:
                ax.set_xlabel("Layer")
            if c == 0:
                ax.set_ylabel(y_label_metric)
            ax.set_xticks(all_layers)

    # Build two legends: (1) TR colors, (2) line styles for type
    # TR legend
    tr_handles = [Line2D([0], [0], color=tr_to_color[tr], lw=2) for tr in trs_all]
    tr_labels  = [f"TR={tr}" for tr in trs_all]

    # Style legend (type)
    type_handles = []
    type_labels  = []
    for t in types_present:
        type_handles.append(Line2D([0], [0], color="black", lw=2, linestyle=ls_map.get(t, "-")))
        type_labels.append(t)

    # Place legends
    # Put TR legend on top, type legend below it
    if tr_handles:
        fig.legend(tr_handles, tr_labels, title="TR (color)", loc="upper center", ncol=min(6, len(tr_handles)))
        fig.subplots_adjust(top=0.86)
    if type_handles:
        fig.legend(type_handles, type_labels, title=f"{type_col} (line style)",
                   loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=len(type_handles))

    fig.suptitle("fMRI: layer-wise mean over targets — rows: ROI, cols: subject\n"
                 "solid = fine-tuned, dashed = frozen; color = TR", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Save / show
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stub = filename_stub or "fmri_layer_tr_pairs_by_roi_subject"
        out = Path(save_dir) / f"{slug(stub)}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.show()




def plot_maxavg_over_layer_tr_by_model_grid(
    df_all: pd.DataFrame,
    *,
    value_col: str = "correlation",
    target_col: str = "target",
    pid_col: str = "participant_id",
    roi_col: str = "roi",
    tr_col: str = "tr",
    type_col: str = "type",
    model_col: str = "model",
    # optional columns that may exist — if present, we will collapse across folds/runs, but HOLD these constant
    hold_constant: tuple[str, ...] = ("n_components","unfreeze_last_n","behavior_embeddings","z_score","ds"),
    # filtering
    filters: dict | None = None,          # e.g. {"n_components": 200, "unfreeze_last_n": 4}
    pval_col: str | None = "p_value_correlation",
    pval_thresh: float | None = None,     # e.g. 0.05 or None
    # averaging options
    fisher_z: bool = True,                # z-transform before averaging
    min_targets: int = 3,                 # require ≥ this many targets to plot a bar
    # figure
    figsize_per_sub=(4.6, 3.0),
    sharey: bool = True,
    save_dir: str | None = None,
    filename_stub: str | None = None,
    show: bool = False,
):
    d = df_all.copy()

    # --- basic checks ---
    needed = {value_col, target_col, pid_col, roi_col, tr_col, type_col, model_col, "layer"}
    missing = needed - set(d.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # --- optional filtering ---
    if filters:
        for k, v in filters.items():
            if k not in d.columns: 
                continue
            if isinstance(v, (list, tuple, set)):
                d = d[d[k].isin(list(v))]
            else:
                d = d[d[k] == v]

    # positive TRs only (recommended for fMRI)
    d[tr_col] = pd.to_numeric(d[tr_col], errors="coerce")
    d = d[(d[tr_col] > 0) & d[tr_col].notna()].copy()

    # p-value filter
    if pval_col and pval_col in d.columns and pval_thresh is not None:
        d = d[pd.to_numeric(d[pval_col], errors="coerce") <= float(pval_thresh)].copy()

    # sanitize metric
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    # --- collapse folds/runs FIRST so they don’t bias the layer/TR max step ---
    # keep these if present, otherwise they’re ignored
    base_keys = [pid_col, roi_col, model_col, type_col, target_col, "layer", tr_col]
    extras_to_hold = [c for c in hold_constant if c in d.columns]
    pre_keys = base_keys + extras_to_hold

    # Average across n_fold/run_id (and any other duplicates) inside the same (pid, roi, model, type, target, layer, tr, *held extras*)
    d_pre = (d.groupby(pre_keys, dropna=False, as_index=False)[value_col]
               .mean())

    # --- now take MAX over layer×TR per target (with held extras constant) ---
    max_keys = [pid_col, roi_col, model_col, type_col, target_col] + extras_to_hold
    d_max = (d_pre.groupby(max_keys, dropna=False, as_index=False)[value_col]
                  .max())   # this is the max across all layers and TRs for that target

    # --- Fisher z before averaging across targets (safer for correlations) ---
    def fisher_z_fn(x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, -0.999999, 0.999999)
        return 0.5 * np.log((1 + x) / (1 - x))

    if fisher_z:
        d_max["_val"] = fisher_z_fn(d_max[value_col].values)
        y_label = "mean Fisher-z(corr) of target-wise max(L,TR)"
    else:
        d_max["_val"] = d_max[value_col].values
        y_label = "mean corr of target-wise max(L,TR)"

    # --- average across targets → one number per (pid, roi, model, type, *held extras*) ---
    avg_keys = [pid_col, roi_col, model_col, type_col] + extras_to_hold
    agg = (d_max.groupby(avg_keys, dropna=False)["_val"]
               .agg(n_targets="count", mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>=2 else np.nan)
               .reset_index())

    # require enough targets
    agg = agg[agg["n_targets"] >= int(min_targets)].copy()

    # --- grid: rows = ROIs, cols = participants; x-axis = models; grouped bars = types ---
    rois = sorted(agg[roi_col].dropna().unique().tolist())
    pids = sorted(agg[pid_col].dropna().unique().tolist())
    types = sorted(agg[type_col].dropna().unique().tolist())
    models = sorted(agg[model_col].dropna().unique().tolist())

    if not rois or not pids or not types or not models:
        print("[plot] Nothing to plot after filtering.")
        return

    nrows, ncols = len(rois), len(pids)
    fig_w = figsize_per_sub[0] * ncols
    fig_h = figsize_per_sub[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=sharey)

    # normalize axes shape
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    x = np.arange(len(models))
    total_width = 0.8
    bar_w = total_width / max(len(types), 1)

    for r, roi in enumerate(rois):
        for c, pid in enumerate(pids):
            ax = axes[r, c]
            sl = agg[(agg[roi_col] == roi) & (agg[pid_col] == pid)]
            if sl.empty:
                ax.axis("off")
                ax.set_title(f"{roi} | pid={pid}\n(no data)")
                continue

            # bars grouped by type, across models on x-axis
            for i, t in enumerate(types):
                y = []; yerr = []
                for m in models:
                    row = sl[(sl[model_col] == m) & (sl[type_col] == t)]
                    if row.empty:
                        y.append(0.0); yerr.append(0.0)
                    else:
                        y.append(float(row["mean"].iloc[0]))
                        yerr.append(float(row["sem"].iloc[0]) if not np.isnan(row["sem"].iloc[0]) else 0.0)
                offs = x - total_width/2 + i*bar_w + bar_w/2
                ax.bar(offs, y, width=bar_w, yerr=yerr if any(yerr) else None, capsize=3 if any(yerr) else 0, label=str(t))
                # annotate
                for xx, yy in zip(offs, y):
                    ax.text(xx, yy, f"{yy:.2f}", ha="center", va="bottom", fontsize=8)

            ax.set_title(f"{roi} | pid={pid}")
            ax.set_xticks(x, [str(m) for m in models], rotation=30, ha="right")
            if c == 0:
                ax.set_ylabel(y_label)

    # single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title=type_col, loc="upper center", ncol=len(types))
        fig.subplots_adjust(top=0.88)

    fig.suptitle("Max over layers×TR per target → average across targets\nrows=ROI, cols=participant; x=model; bars=type", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

        # Save / show
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stub = filename_stub or "fmri_max_over_LxTR_avg_by_model_"
        out = Path(save_dir) / f"{slug(stub)}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.show()
# Build model -> max_layer map if available


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_max_travg_over_layers_by_model_grid(
    df_all: pd.DataFrame,
    *,
    value_col: str = "correlation",
    target_col: str = "target",
    pid_col: str = "participant_id",
    roi_col: str = "roi",
    tr_col: str = "tr",
    type_col: str = "type",
    model_col: str = "model",
    # keep settings constant (if present) or filter them via `filters`
    hold_constant: tuple[str, ...] = ("n_components","unfreeze_last_n","behavior_embeddings","z_score","ds"),
    filters: dict | None = None,              # e.g. {"ds":"sagar2023","n_components":200,"unfreeze_last_n":4}
    pval_col: str | None = "p_value_correlation",
    pval_thresh: float | None = None,         # e.g. 0.05 or None
    fisher_z: bool = True,                    # z-transform before averaging over targets/TRs
    min_targets: int = 3,                     # require ≥ this many targets per (pid,roi,model,type,layer,TR)
    figsize_per_sub=(4.6, 3.0),
    sharey: bool = True,
    save_dir: str | None = None,
    filename_stub: str | None = None,
    show: bool = False,
):
    d = df_all.copy()

    # --- required columns ---
    needed = {value_col, target_col, pid_col, roi_col, tr_col, type_col, model_col, "layer"}
    missing = needed - set(d.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # --- filters (keep constants fixed so you don't mix configs) ---
    if filters:
        for k, v in filters.items():
            if k not in d.columns: 
                continue
            d = d[d[k].isin(v)] if isinstance(v, (list, tuple, set)) else d[d[k] == v]

    # --- TR>0 only (fMRI) ---
    d[tr_col] = pd.to_numeric(d[tr_col], errors="coerce")
    d = d[(d[tr_col] > 0) & d[tr_col].notna()].copy()
    d["layer"] = pd.to_numeric(d["layer"], errors="coerce")

    # --- p-value filter (optional) ---
    if pval_col and pval_col in d.columns and pval_thresh is not None:
        d = d[pd.to_numeric(d[pval_col], errors="coerce") <= float(pval_thresh)].copy()

    # --- metric sanitize ---
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])

    # --- collapse duplicates (folds/runs/etc.) BEFORE any averaging/max ---
    base_keys = [pid_col, roi_col, model_col, type_col, target_col, "layer", tr_col]
    extras_to_hold = [c for c in hold_constant if c in d.columns]
    pre_keys = base_keys + extras_to_hold

    d_pre = (d.groupby(pre_keys, dropna=False, as_index=False)[value_col]
               .mean())

    # --- Fisher-z before averaging (safer for correlations) ---
    def fisher_z_fn(x):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, -0.999999, 0.999999)
        return 0.5 * np.log((1 + x) / (1 - x))

    val_col = value_col
    if fisher_z:
        d_pre["_val"] = fisher_z_fn(d_pre[value_col].values)
        val_col = "_val"
        y_label = "mean Fisher-z(corr) of ⟨targets⟩→⟨TR⟩, max layer"
    else:
        y_label = "mean corr of ⟨targets⟩→⟨TR⟩, max layer"

    # --- step 1: average over TARGETS at each (pid,roi,model,type,layer,TR) ---
    tavg_keys = [pid_col, roi_col, model_col, type_col, "layer", tr_col] + extras_to_hold
    per_tr = (d_pre.groupby(tavg_keys, dropna=False)[val_col]
                    .agg(n_targets="count", mean_over_targets="mean")
                    .reset_index())
    # enforce enough targets
    per_tr = per_tr[per_tr["n_targets"] >= int(min_targets)].copy()

    # --- step 2: average over TRs → one value per (pid,roi,model,type,layer) ---
    lavg_keys = [pid_col, roi_col, model_col, type_col, "layer"] + extras_to_hold
    per_layer = (per_tr.groupby(lavg_keys, dropna=False)["mean_over_targets"]
                       .mean()            # mean over TRs
                       .reset_index()
                       .rename(columns={"mean_over_targets":"tr_avg"}))

    # --- step 3: pick the max layer per (pid,roi,model,type,held-extras) ---
    max_keys = [pid_col, roi_col, model_col, type_col] + extras_to_hold
    idx = per_layer.groupby(max_keys, dropna=False)["tr_avg"].idxmax()
    best = per_layer.loc[idx].copy()  # has columns: ... + ["layer","tr_avg"]

    # --- grid data ---
    rois   = sorted(best[roi_col].dropna().unique().tolist())
    pids   = sorted(best[pid_col].dropna().unique().tolist())
    types  = sorted(best[type_col].dropna().unique().tolist())
    models = sorted(best[model_col].dropna().unique().tolist())

    if not rois or not pids or not types or not models:
        print("[plot] Nothing to plot after filtering.")
        return

    # --- plot: rows=ROIs, cols=participants; x=models; grouped bars=types ---
    nrows, ncols = len(rois), len(pids)
    fig_w = figsize_per_sub[0] * ncols
    fig_h = figsize_per_sub[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=sharey)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    x = np.arange(len(models))
    total_w = 0.8
    bar_w = total_w / max(len(types), 1)

    for r, roi in enumerate(rois):
        for c, pid in enumerate(pids):
            ax = axes[r, c]
            sl = best[(best[roi_col] == roi) & (best[pid_col] == pid)]
            if sl.empty:
                ax.axis("off")
                ax.set_title(f"{roi} | pid={pid}\n(no data)")
                continue

            for i, t in enumerate(types):
                y = []; layers = []
                for m in models:
                    row = sl[(sl[model_col] == m) & (sl[type_col] == t)]
                    if row.empty:
                        y.append(0.0); layers.append(None)
                    else:
                        y.append(float(row["tr_avg"].iloc[0]))
                        layers.append(int(row["layer"].iloc[0]) if pd.notna(row["layer"].iloc[0]) else None)

                offs = x - total_w/2 + i*bar_w + bar_w/2
                bars = ax.bar(offs, y, width=bar_w, label=str(t))

                # annotate value and chosen layer
                for xx, yy, lay in zip(offs, y, layers):
                    ax.text(xx, yy, f"{yy:.2f}" + (f"\nL{lay}" if lay is not None else ""),
                            ha="center", va="bottom", fontsize=8)

            ax.set_title(f"{roi} | pid={pid}")
            ax.set_xticks(x, [str(m) for m in models], rotation=30, ha="right")
            if c == 0:
                ax.set_ylabel(y_label)

    # legend + layout
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title=type_col, loc="upper center", ncol=len(types))
        fig.subplots_adjust(top=0.88)

    fig.suptitle("Avg over targets per TR → avg over TRs → choose max layer\nrows=ROI, cols=participant; x=model; bars=type", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # save/show
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stub = filename_stub or "fmri_travg_maxlayer_by_model"
        out = Path(save_dir) / f"{stub}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.show()


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


# -------------- File naming utilities --------------
# def parse_metrics_filename(path: Union[str, Path]) -> Dict[str, str]:
#     """
#     Extract model, dataset, and runid from filename:
#     behavior_metrics_model-<MODEL>_ds-<DATASET>_runid-<RUNID>.csv
#     """
#     fname = Path(path).name
#     m = re.match(r".*metrics.*_model-(.+?)_ds-(.+?)\.csv", fname)
#     if not m:
#         raise ValueError(f"Filename does not match pattern: {fname}")
#     return {"model": m.group(1), "ds": m.group(2)}

# def parse_metrics_filename_tuned(path: Union[str, Path]) -> Dict[str, str]:
#     """
#     Extract model, dataset, and runid from filename:
#     behavior_metrics_model-<MODEL>_ds-<DATASET>_runid-<RUNID>.csv
#     """
#     fname = Path(path).name
#     m = re.match(r".*metrics_model-(.+?)_ds-(.+?)_runid-(.+?)_unfreeze-(.+?)_behembd-(.+?)\.csv", fname)
#     if not m:
#         raise ValueError(f"Filename does not match pattern: {fname}")
#     return {"model": m.group(1), "ds": m.group(2), "run_id": m.group(3), "unfreeze_last_n": m.group(4), "behavior_embeddings": m.group(5)}


# def parse_metrics_filename_tuned2(path: Union[str, Path]) -> Dict[str, str]:
#     """
#     Extract model, dataset, and runid from filename:
#     behavior_metrics_model-<MODEL>_ds-<DATASET>_runid-<RUNID>.csv
#     """
#     fname = Path(path).name
#     m = re.match(r".*metrics.*_model-(.+?)_ds-(.+?)_unfreeze-(.+?)_behembd-(.+?)\.csv", fname)
#     if not m:
#         raise ValueError(f"Filename does not match pattern: {fname}")
#     return {"model": m.group(1), "ds": m.group(2), "unfreeze_last_n": m.group(3), "behavior_embeddings": m.group(4)}


# def load_metrics_file(path: Union[str, Path]) -> pd.DataFrame:
#     meta = parse_metrics_filename(path)
#     df = pd.read_csv(path)
#     df["source_file"] = Path(path).name
#     df["row_idx"] = df.index   # preserve original index from CSV
#     for k, v in meta.items():
#         df[k] = v
#     return df


# def load_metrics_tuned_file(path: Union[str, Path]) -> pd.DataFrame:
#     meta = parse_metrics_filename_tuned(path)
#     df = pd.read_csv(path)
#     df["source_file"] = Path(path).name
#     df["row_idx"] = df.index
#     for k, v in meta.items():
#         df[k] = v
#     return df

# def load_metrics_tuned_file2(path: Union[str, Path]) -> pd.DataFrame:
#     meta = parse_metrics_filename_tuned2(path)
#     df = pd.read_csv(path)
#     df["source_file"] = Path(path).name
#     df["row_idx"] = df.index
#     for k, v in meta.items():
#         df[k] = v
#     return df

# <<< NEW: generic loader for new CSVs that may include participant_source_id and don't follow filename patterns
# def load_metrics_generic(path: Union[str, Path]) -> pd.DataFrame:
#     """
#     Read a metrics CSV with columns present in the file (no filename parsing).
#     Adds source_file + row_idx. Does NOT add model/ds/run_id from filename.
#     Use when filenames don't follow the old patterns or when new columns appear.
#     """
#     path = Path(path)
#     df = pd.read_csv(path)
#     df["source_file"] = path.name
#     df["row_idx"] = df.index
#     return df


# <<< NEW: directory loader using the generic reader
# def load_any_metrics_in_dir(directory: Union[str, Path]) -> pd.DataFrame:
#     """
#     Load all CSVs in a directory using the generic loader; do not parse filename.
#     """
#     directory = Path(directory)
#     files = sorted(directory.glob("*.csv"))
#     if not files:
#         raise ValueError(f"No CSV files found in {directory}")
#     dfs = [load_metrics_generic(f) for f in files]
#     return pd.concat(dfs, ignore_index=True)

# def load_all_metrics(directory: Union[str, Path]) -> pd.DataFrame:
#     directory = Path(directory)
#     files = list(directory.glob("*metrics*_model-*_ds-*.csv"))
#     if not files:
#         raise ValueError(f"No metrics CSVs found in {directory}")
#     dfs = [load_metrics_file(f) for f in files]
#     return pd.concat(dfs, ignore_index=True)

# def load_tuned_metrics(directory: Union[str, Path]) -> pd.DataFrame:
#     directory = Path(directory)
#     files = list(directory.glob("*metrics_model-*_ds-*_runid-*_unfreeze-*_behembd-*.csv"))
#     if not files:
#         raise ValueError(f"No metrics CSVs found in {directory}")
#     dfs = [load_metrics_tuned_file(f) for f in files]
#     return pd.concat(dfs, ignore_index=True)

# def load_tuned_metrics2(directory: Union[str, Path]) -> pd.DataFrame:
#     directory = Path(directory)
#     files = list(directory.glob("*metrics_model-*_ds-*_unfreeze-*_behembd-*.csv"))
#     if not files:
#         raise ValueError(f"No metrics CSVs found in {directory}")
#     dfs = [load_metrics_tuned_file2(f) for f in files]
#     return pd.concat(dfs, ignore_index=True)