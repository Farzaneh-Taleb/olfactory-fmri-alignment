import sys
import os
from pathlib import Path

# --- Make imports work from anywhere ---
REPO_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_DIR))  # so `config` is importable
from utils.config import BASE_DIR as CONFIG_BASE_DIR  # do not shadow later
sys.path.append(str(REPO_DIR))  # ensure utils is on path
from utils.helpers import save_fold_indices

import argparse
from utils.helpers import save_fold_indices

def main():

    BASE_DIR = '../DATASETS'
    for ds in  ["keller2016","sagar2023"]:
        for n_fold in [5,10]:
            save_fold_indices(BASE_DIR, n_fold,ds)

if __name__ == "__main__":
    main()