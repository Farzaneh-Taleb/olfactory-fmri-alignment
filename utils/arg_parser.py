
"""
Common argument parsing utilities for regression scripts.
"""
import argparse
import os
def str_list_or_empty(v: str) -> list[str]:
    if not v or v.strip() == "":
        return []
    return [c.strip() for c in v.split(",") if c.strip()]

def none_or_int(value):
    if value.lower() in ("none", ""):
        return None
    elif value == "adaptive":
        return "adaptive"
    return int(value)



def none_or_float(value):
    if value.lower() in ("none", ""):
        return None
    return float(value)
def create_base_parser(description='chem_exploration'):
    """
    Create base argument parser with common arguments.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser: Configured parser with common arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Common arguments
    parser.add_argument('--participant_id', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ds', type=str, required=True)
    parser.add_argument('--n_components', type=str, default="None")

    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_fold', type=int, required=True)
    parser.add_argument('--z_score', type=str, default="false")
    parser.add_argument("--run_id", default=os.environ.get("RUN_ID", "unknown"))

    
    return parser
def create_extract_rep_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract finetuned representations")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--participant_id", required=True, type=int)
    parser.add_argument("--n_fold", required=True, type=int)
    parser.add_argument("--behavior_embeddings",type=str_list_or_empty,default=[],
    help="Comma-separated list of behavior embedding columns (or empty)."
)
    parser.add_argument("--unfreeze_last_n", type=none_int_or_keywords)
    parser.add_argument("--ds", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--run_id", required=True, type=str)
    return parser


def create_fmri_parser(description='chem_exploration'):
    """
    Create argument parser for fMRI-specific scripts.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser: Configured parser with fMRI-specific arguments
    """
    parser = create_base_parser(description)
    
    # fMRI-specific arguments
    parser.add_argument('--roi', type=str, required=True)
    parser.add_argument('--tr', type=int, default=-1)
    parser.add_argument("--behavior_embeddings",type=str_list_or_empty,default=[],
    help="Comma-separated list of behavior embedding columns (or empty)."
)
    parser.add_argument("--unfreeze_last_n", type=none_int_or_keywords)
    return parser

def create_fmri_finetune_parser(description='chem_exploration'):
    """
    Create argument parser for fMRI-specific scripts.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser: Configured parser with fMRI-specific arguments
    """
    parser = create_base_parser(description)
    
    # fMRI-specific arguments
    parser.add_argument('--roi', type=str, required=True)
    parser.add_argument('--tr', type=int, default=-1)

    
    return parser


def create_behavior_parser(description='chem_exploration'):
    """
    Create argument parser for fMRI-specific scripts.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser: Configured parser with fMRI-specific arguments
    """
    parser = create_base_parser(description)
    
    # fMRI-specific arguments
    parser.add_argument("--behavior_embeddings",type=str_list_or_empty,default=[],
    help="Comma-separated list of behavior embedding columns (or empty)."
)
    
    return parser


def parse_common_args(args):
    """
    Parse and convert common arguments to appropriate types.
    
    Args:
        args: Parsed arguments object
        
    Returns:
        Parsed and converted arguments
    """
    # Convert string boolean arguments
    args.z_score = str(args.z_score).lower() == 'true'
    
    # Convert n_components
    if args.n_components == "None":
        args.n_components = None
    else:
        args.n_components = float(args.n_components)
    
    return args


# def parse_fmri_args(args):
#     """
#     Parse and convert fMRI-specific arguments.
    
#     Args:
#         args: Parsed arguments object
        
#     Returns:
#         Parsed and converted arguments
#     """
#     args = parse_common_args(args)
    
#     return args



def create_finetune_parser(description='finetune_by_behavior'):
    """
    Parser for fine-tuning on string inputs (SMILES/SELFIES) with multi-target
    behavior regression. Matches flags used by finetune_by_behavior.py and the runner.
    """
    # start from base + behavior embeddings helper
    parser = create_behavior_parser(description)

    # fine-tuning specific args

    parser.add_argument(
    '--unfreeze_last_n',
    type=none_int_or_keywords,
    default=None,
    help="Unfreeze last N encoder layers (None/empty = freeze backbone; heads always trainable)."
)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--embed_type', type=str, default='can', choices=['can', 'iso'],
                        help="Which text field to use from the dataset CSV.")
    parser.add_argument('--i_fold', type=int, default=0,
                        help="Token index to extract from each sequence (default=0 for CLS).")
    

        # ---------- LoRA (optional; defaults keep current behavior) ----------
    # LoRA flags (default OFF -> current behavior unchanged)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters (default: off)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="auto",
        help=(
            "LoRA target selection. "
            "'auto' = attention only (q/k/v/o, out_proj, etc.); "
            "'auto_all' or 'all' = attention + common MLP (fc1/fc2, intermediate/output, up/down/gate). "
            "Or pass a comma list (e.g., 'q_proj,k_proj,v_proj,o_proj')."
        ),
    )

    return parser



def create_hparm_parser(description='best params parser'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ds",                  required=True)
    parser.add_argument("--model",          required=True)
    parser.add_argument("--participant_id",             required=True, type=int)
    parser.add_argument("--behavior_embeddings",type=str_list_or_empty,default=[],
    help="Comma-separated list of behavior embedding columns (or empty)."
)
    
    parser.add_argument("--unfreeze_last_n",     type=none_int_or_keywords,
    default=None,
    help="Unfreeze last N encoder layers (None/empty = freeze backbone; heads always trainable).")
    parser.add_argument("--run_id",            required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_fold', type=int, required=True)
    parser.add_argument('--metrics_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    
    
    
    return parser


def create_regression_behavior_parser(description="regression_behavior"):
    """
    Parser for regression_behavior_refactored.py.

    Args:
        description: Description for the parser

    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)

    # Required args
    parser.add_argument("--ds", required=True, type=str,
                        help="Dataset identifier (e.g., sagar2023).")
    parser.add_argument("--participant_id", required=True, type=int,
                        help="Subject/participant ID.")
    parser.add_argument("--model", required=True, type=str,
                        help="Model name/path.")
    parser.add_argument("--n_fold", required=True, type=int,
                        help="Fold index for CV.")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="Output directory path.")

    # Optional args
    parser.add_argument("--n_components", type=none_or_float, default=None,
                        help="Number of PCA components (None = skip PCA).")
    parser.add_argument("--behavior_embeddings", type=str_list_or_empty, default=[],
                        help="Comma-separated list of behavior embedding columns (or empty = default).")
    parser.add_argument("--unfreeze_last_n", type=none_int_or_keywords, default=None,
                        help="Unfreeze last N encoder layers (None = freeze backbone).")
    parser.add_argument("--z_score", type=lambda v: str(v).lower() == "true", default=False,
                        help="Apply z-scoring (true/false).")
    parser.add_argument("--run_id", type=str,
                        help="Unique run identifier (default from RUN_ID env).")

    return parser


# def parse_unfreeze_last_n(x):
#     if x is None:
#         return "all"                      # None ⇒ unfreeze all
#     if isinstance(x, str):
#         s = x.strip().lower()
#         if s in {"", "none", "all"}:      return "all"
#         if s in {"adaptive"}: return "adaptive"
#         return int(s)
#     if isinstance(x, (int, float)):
#         return int(x)
#     raise ValueError(f"Invalid --unfreeze_last_n: {x!r}")


def none_int_or_keywords(value):
    """
    Accepts:
      - empty/none  -> None
      - integers    -> int
      - 'all'       -> 'all'
      - 'adaptive'  -> 'adaptive'
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"", "none"}:
        return None
    if s in {"all", "adaptive"}:
        return s
    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid value: must be an integer, 'all', or 'adaptive'"
        )