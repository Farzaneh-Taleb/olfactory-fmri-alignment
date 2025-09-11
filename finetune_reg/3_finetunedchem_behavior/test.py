#!/usr/bin/env python3
import re
import shlex
from pathlib import Path
import pandas as pd

LOG_DIR = Path("logs")  # change if needed

# --- Error signatures to flag ---
ERR_PATTERNS = [
    re.compile(r'module\(s\).+cannot be loaded.+miniconda3', re.IGNORECASE | re.DOTALL),
    re.compile(r'ValueError:\s*None of the requested CIDs are present in the embeddings file\.', re.IGNORECASE),
]

def parse_kv_line(line: str) -> dict:
    """
    Parse a line of shell-style key=value tokens.
    Uses shlex to respect quotes.
    Returns dict of parsed key/values, leaves as strings.
    """
    out = {}
    for token in shlex.split(line.strip()):
        if '=' in token:
            k, v = token.split('=', 1)
            out[k.strip()] = v.strip()
    return out

def detect_error(err_text: str) -> str | None:
    """Return a short label for the first matched error pattern, else None."""
    for pat in ERR_PATTERNS:
        if pat.search(err_text):
            if 'requested CIDs' in pat.pattern:
                return 'MISSING_CIDS'
    return None

def find_out_for_err(err_path: Path) -> Path:
    """
    Given logs/error_<JOBID>_<ARRAY>.err -> logs/output_<JOBID>_<ARRAY>.out
    """
    name = err_path.name
    m = re.match(r"error_(\d+)_([0-9]+)\.err$", name)
    if not m:
        return None
    jobid, arr = m.group(1), m.group(2)
    out_name = f"output_{jobid}_{arr}.out"
    return err_path.parent / out_name

def process_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for err_path in sorted(log_dir.glob("error_*_*.err")):
        try:
            err_text = err_path.read_text(errors="ignore")
        except Exception as e:
            print(f"Could not read {err_path}: {e}")
            continue

        err_type = detect_error(err_text)
        if not err_type:
            continue  # skip non-matching errors

        out_path = find_out_for_err(err_path)
        run_id = None
        params = {}

        if out_path and out_path.exists():
            try:
                lines = out_path.read_text(errors="ignore").splitlines()
            except Exception as e:
                print(f"Could not read {out_path}: {e}")
                lines = []

            # Grab RUN_ID from any line that starts with 'RUN_ID='
            for ln in lines[:5]:  # check a few early lines
                if ln.strip().startswith("RUN_ID="):
                    rid_kv = parse_kv_line(ln)
                    run_id = rid_kv.get("RUN_ID")
                    break

            # “Line 2” per your example is the second non-empty line (index 1).
            # We’ll be literal: if at least 2 lines exist, take lines[1].
            # If it’s empty, try to find the first line that looks like key=val pairs.
            param_line = None
            if len(lines) >= 2 and lines[1].strip():
                param_line = lines[1]
            else:
                # Fallback: find the first line that has at least one '='
                for ln in lines:
                    if "=" in ln:
                        param_line = ln
                        break

            if param_line:
                params = parse_kv_line(param_line)
        else:
            print(f"No matching .out for {err_path.name}")

        # Pull jobid/array from filename
        m = re.match(r"error_(\d+)_([0-9]+)\.err$", err_path.name)
        jobid = m.group(1) if m else None
        array_id = m.group(2) if m else None

        row = {
            "err_file": str(err_path),
            "out_file": str(out_path) if out_path else None,
            "job_id": jobid,
            "array_id": array_id,
            "error_type": err_type,
            "RUN_ID": run_id,
        }

        # Common fields you showed; include if present in parsed params
        for key in [
            "ds", "participant_id", "n_fold", "n_components", "model",
            "behavior_embeddings", "z_score", "unfreeze_last_n", "out_dir"
        ]:
            row[key] = params.get(key)

        rows.append(row)

    return pd.DataFrame(rows)

def main():
    df = process_logs(LOG_DIR)
    if df.empty:
        print("No matching errors found.")
        return
    # Display & save
    print(df.to_string(index=False))
    out_csv = LOG_DIR / "flagged_errors.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
