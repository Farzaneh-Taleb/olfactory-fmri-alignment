
"""
Common argument parsing utilities for regression scripts.
"""
import argparse
import os
def str_list_or_empty(v: str) -> list[str]:
    if not v or v.strip() == "":
        return []
    return [c.strip() for c in v.split(",") if c.strip()]

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
        args.n_components = int(args.n_components)
    
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

    parser.add_argument('--unfreeze_last_n', type=int, default=0,
                        help="Unfreeze last N encoder layers (0 = freeze backbone; heads always trainable).")
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--input_type', type=str, default='smiles', choices=['smiles', 'selfies'],
                        help="Which text field to use from the dataset CSV.")

    # keep --model for labeling/metadata (separate from model_name_path)

    return parser