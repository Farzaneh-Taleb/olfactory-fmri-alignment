import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse
from pathlib import Path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = '/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment'
sys.path.insert(0, project_root)

from utils.config import BASE_DIR, SEED

def construct_filename(BASE_DIR, model_name, n_fold, num_train_epochs, subject, 
                      behavior_embedding, unfreeze_last_n, ds="", lr=None, 
                      batch_size=None, weight_decay=None, out_dir="read_orig_avg"):
    """
    Construct the CSV filename based on parameters
    
    Args:
        BASE_DIR (str): Base directory path
        model_name (str): Model name (e.g., 'ChemBERTa-77M-MLM')
        n_fold (int): Number of folds
        num_train_epochs (int): Number of training epochs
        subject (int): Subject number
        behavior_embedding (str): Behavior embedding indices (e.g., '0' or '0,1,2')
        unfreeze_last_n (int): Number of last layers to unfreeze
        ds (str): Dataset suffix (default: "")
        lr (float): Learning rate (optional)
        batch_size (int): Batch size (optional)
        weight_decay (float): Weight decay (optional)
        out_dir (str): Output directory name (default: "read_orig_avg")
    
    Returns:
        str: Full path to the CSV file
    """
    # Base filename pattern: mean_mse_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embedding}_{unfreeze_last_n}{ds}
    filename = f"mean_mse_{model_name}_{n_fold}_{num_train_epochs}_{subject}_{behavior_embedding}_{unfreeze_last_n}{ds}"
    
    # Add optional parameters if provided
    if lr is not None:
        filename += f"_{lr}"
    if batch_size is not None:
        filename += f"_{batch_size}"
    if weight_decay is not None:
        filename += f"_{weight_decay}"
    
    filename += ".csv"
    
    # Construct full path
    full_path = os.path.join(BASE_DIR, f"{out_dir}_metrics", filename)
    print("full path", full_path)
    return full_path

def find_matching_files(BASE_DIR, model_name=None, n_fold=None, num_train_epochs=None, 
                       subject=None, behavior_embedding=None, unfreeze_last_n=None, 
                       ds="", out_dir="read_orig_avg"):
    """
    Find CSV files matching the given parameters (with wildcards for None values)
    """
    # Build pattern with wildcards for None values
    pattern_parts = ["mean_mse"]
    
    pattern_parts.append(model_name if model_name is not None else "*")
    pattern_parts.append(str(n_fold) if n_fold is not None else "*")
    pattern_parts.append(str(num_train_epochs) if num_train_epochs is not None else "*")
    pattern_parts.append(str(subject) if subject is not None else "*")
    pattern_parts.append(str(behavior_embedding) if behavior_embedding is not None else "*")
    pattern_parts.append(str(unfreeze_last_n) if unfreeze_last_n is not None else "*")
    
    pattern = "_".join(pattern_parts) + f"{ds}*.csv"
    search_path = os.path.join(BASE_DIR, f"{out_dir}_metrics", pattern)
    print(search_path)
    
    return glob.glob(search_path)

def plot_mse_losses(csv_file_path, save_dir=None, show_plot=True):
    """
    Plot MSE losses from mean_mse CSV files
    
    Args:
        csv_file_path (str): Path to the mean_mse CSV file
        save_dir (str): Directory to save plots (optional)
        show_plot (bool): Whether to display the plot
    """
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found: {csv_file_path}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Extract filename components for titles
    filename = Path(csv_file_path).stem
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Get unique folds and target columns
    folds = df['fold'].unique()
    n_folds = len(folds)
    
    # Get target columns (mse_0, mse_1, etc.)
    # target_cols = [col for col in df.columns if col.startswith('mse_') or col == 'mean_mse']
    target_cols = [col for col in df.columns if col == 'mean_mse']
    
    n_targets = len(target_cols)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Plot mean MSE across all folds and targets
    plt.subplot(2, 3, 1)
    for fold in folds:
        fold_data = df[df['fold'] == fold]
        print(fold_data['epoch'].values.tolist(), fold_data[target_cols].values.tolist())
        plt.plot(fold_data['epoch'], fold_data[target_cols], 
                marker='o', linewidth=2, markersize=4, label=f'Fold {fold}')
    
    # Plot overall mean across folds
    # mean_across_folds = df.groupby('epoch')[target_cols].mean()
    # plt.plot(mean_across_folds.index, mean_across_folds.values, 
    #         'k--', linewidth=3, marker='s', markersize=6, label='Mean Across Folds')
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean MSE')
    plt.title('Mean MSE Across All Targets and Folds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Plot MSE for each target (averaged across folds)
    # plt.subplot(2, 3, 2)
    for target_col in target_cols:
        target_mean = df.groupby('epoch')[target_col].mean()
        target_num = target_col.split('_')[1]
        plt.plot(target_mean.index, target_mean.values, 
                marker='*', linewidth=2, markersize=4, label=f'Avg fold {target_num}')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE per Target (Mean Across Folds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # # 3. Heatmap of MSE across epochs and folds (mean MSE)
    # plt.subplot(2, 3, 3)
    # pivot_df = df.pivot(index='epoch', columns='fold', values='mean_mse')
    # sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='viridis_r', 
    #             cbar_kws={'label': 'Mean MSE'})
    # plt.title('Mean MSE Heatmap (Epochs vs Folds)')
    # plt.ylabel('Epoch')
    # plt.xlabel('Fold')
    
    # # 4. Box plot of MSE distribution across folds for each epoch
    # plt.subplot(2, 3, 4)
    # epoch_data = []
    # epoch_labels = []
    # for epoch in sorted(df['epoch'].unique()):
    #     epoch_mse = df[df['epoch'] == epoch]['mean_mse'].values
    #     epoch_data.append(epoch_mse)
    #     epoch_labels.append(f'E{int(epoch)}')
    
    # plt.boxplot(epoch_data, labels=epoch_labels)
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean MSE')
    # plt.title('MSE Distribution Across Folds per Epoch')
    # plt.xticks(rotation=45)
    # plt.grid(True, alpha=0.3)
    
    # # 5. Target-wise MSE progression (separate lines for each fold)
    # plt.subplot(2, 3, 5)
    # colors = plt.cm.Set3(np.linspace(0, 1, n_targets))
    
    # for i, target_col in enumerate(target_cols):
    #     target_num = target_col.split('_')[1]
    #     for fold in folds:
    #         fold_data = df[df['fold'] == fold]
    #         alpha = 0.6 if fold != folds[0] else 1.0  # Highlight first fold
    #         plt.plot(fold_data['epoch'], fold_data[target_col], 
    #                 color=colors[i], alpha=alpha, linewidth=1.5,
    #                 label=f'Target {target_num}' if fold == folds[0] else "")
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Individual Target MSE (All Folds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # # 6. Learning curve with error bars
    # plt.subplot(2, 3, 6)
    # epochs = sorted(df['epoch'].unique())
    # mean_values = []
    # std_values = []
    
    # for epoch in epochs:
    #     epoch_data = df[df['epoch'] == epoch]['mean_mse']
    #     mean_values.append(epoch_data.mean())
    #     std_values.append(epoch_data.std())
    
    # plt.errorbar(epochs, mean_values, yerr=std_values, 
    #             marker='o', linewidth=2, capsize=5, capthick=2,
    #             label='Mean ± Std across folds')
    # plt.fill_between(epochs, 
    #                 np.array(mean_values) - np.array(std_values),
    #                 np.array(mean_values) + np.array(std_values),
    #                 alpha=0.2)
    
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean MSE')
    # plt.title('Learning Curve with Error Bars')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    
    # Save plot if save_dir is provided
    if save_dir:
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{filename}_loss_plots.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print summary statistics
    print(f"\n=== Loss Summary for {filename} ===")
    print(f"Number of folds: {n_folds}")
    print(f"Number of targets: {n_targets}")
    print(f"Number of epochs: {len(df['epoch'].unique())}")
    print(f"Final mean MSE (last epoch): {df[df['epoch'] == df['epoch'].max()]['mean_mse'].mean():.6f}")
    print(f"Best mean MSE (across all epochs): {df['mean_mse'].min():.6f}")
    print(f"Final std MSE (last epoch): {df[df['epoch'] == df['epoch'].max()]['mean_mse'].std():.6f}")

def plot_multiple_experiments(csv_files, save_dir=None, show_plot=True):
    """
    Plot MSE losses for multiple experiments
    
    Args:
        csv_files (list): List of CSV file paths
        save_dir (str): Directory to save plots (optional)
        show_plot (bool): Whether to display the plot
    """
    if not csv_files:
        print("No CSV files provided")
        return
    
    print(f"Plotting {len(csv_files)} experiments:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
    
    # Main comparison plot
    ax = axes[0, 0]
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        filename = Path(csv_file).stem
        
        # Calculate mean across folds for each epoch
        mean_across_folds = df.groupby('epoch')['mean_mse'].agg(['mean', 'std'])
        
        ax.plot(mean_across_folds.index, mean_across_folds['mean'], 
                color=colors[i], marker='o', linewidth=2, label=filename)
        ax.fill_between(mean_across_folds.index,
                        mean_across_folds['mean'] - mean_across_folds['std'],
                        mean_across_folds['mean'] + mean_across_folds['std'],
                        color=colors[i], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean MSE')
    ax.set_title('Comparison of Mean MSE Across Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot individual experiments in remaining subplots
    subplot_positions = [(0, 1), (1, 0), (1, 1)]
    for i, csv_file in enumerate(csv_files[:3]):  # Only plot first 3 for space
        if i < len(subplot_positions):
            ax = axes[subplot_positions[i]]
            plot_single_experiment_subplot(csv_file, ax)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "experiment_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_single_experiment_subplot(csv_file, ax):
    """Helper function to plot a single experiment in a subplot"""
    df = pd.read_csv(csv_file)
    filename = Path(csv_file).stem
    
    folds = df['fold'].unique()
    for fold in folds:
        fold_data = df[df['fold'] == fold]
        ax.plot(fold_data['epoch'], fold_data['mean_mse'], 
                marker='o', linewidth=1, markersize=3, alpha=0.7, label=f'Fold {fold}')
    
    mean_across_folds = df.groupby('epoch')['mean_mse'].mean()
    ax.plot(mean_across_folds.index, mean_across_folds.values, 
            'k--', linewidth=2, label='Mean')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean MSE')
    ax.set_title(f'{filename}')
    ax.legend()
    ax.grid(True, alpha=0.3)

def main():
    parser = argparse.ArgumentParser(description='Plot MSE losses from mean_mse CSV files')
    
    # File specification options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--csv_file', type=str, help='Direct path to CSV file')
    group.add_argument('--csv_pattern', type=str, help='Glob pattern for CSV files')
    
    # Filename component arguments
    parser.add_argument('--BASE_DIR', type=str, 
                       default='/cfs/klemming/projects/supr/olfactory_alignment',
                       help='Base directory path')
    parser.add_argument('--model_name', type=str, help='Model name (e.g., ChemBERTa-77M-MLM)')
    parser.add_argument('--n_fold', type=int, help='Number of folds')
    parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--subject', type=int, help='Subject number')
    parser.add_argument('--behavior_embedding', type=str, help='Behavior embedding (e.g., "0" or "0,1,2")')
    parser.add_argument('--unfreeze_last_n', type=int, help='Number of last layers unfrozen')
    parser.add_argument('--ds', type=str, default='', help='Dataset suffix')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--out_dir', type=str, default='read_orig_avg', help='Output directory name')
    
    # Output options
    parser.add_argument('--save_dir', type=str, help='Directory to save plots')
    parser.add_argument('--no_show', action='store_true', help='Don\'t display plots')
    parser.add_argument('--compare', action='store_true', help='Create comparison plot for multiple files')
    
    args = parser.parse_args()
    args.save_dir =  f"{args.save_dir}/{args.subject}/{args.model_name}/"
    
    csv_files = []
    
    if args.csv_file:
        # Direct file path
        csv_files = [args.csv_file]
    elif args.csv_pattern:
        # Glob pattern
        csv_files = glob.glob(args.csv_pattern)
    else:
        print("ccc")
        # Construct filename or find matching files
        if all(param is not None for param in [args.model_name, args.n_fold, args.num_train_epochs, 
                                             args.subject, args.behavior_embedding, args.unfreeze_last_n]):
            print("aaa")
            # All parameters specified - construct exact filename
            csv_file = construct_filename(
                args.BASE_DIR, args.model_name, args.n_fold, args.num_train_epochs,
                args.subject, args.behavior_embedding, args.unfreeze_last_n,
                args.ds, args.lr, args.batch_size, args.weight_decay, args.out_dir
            )
        
            csv_files = [csv_file] if os.path.exists(csv_file) else []
        else:
            print("bbb")
            # Some parameters missing - find matching files
            csv_files = find_matching_files(
                args.BASE_DIR, args.model_name, args.n_fold, args.num_train_epochs,
                args.subject, args.behavior_embedding, args.unfreeze_last_n,
                args.ds, args.out_dir
            )
    
    if not csv_files:
        print("No CSV files found matching the criteria")
        if not args.csv_file and not args.csv_pattern:
            print("\nParameters used for search:")
            print(f"  BASE_DIR: {args.BASE_DIR}")
            print(f"  model_name: {args.model_name}")
            print(f"  n_fold: {args.n_fold}")
            print(f"  num_train_epochs: {args.num_train_epochs}")
            print(f"  subject: {args.subject}")
            print(f"  behavior_embedding: {args.behavior_embedding}")
            print(f"  unfreeze_last_n: {args.unfreeze_last_n}")
            print(f"  ds: '{args.ds}'")
            print(f"  out_dir: {args.out_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for file in csv_files:
        print(f"  - {file}")
    
    # Plot files
    if args.compare and len(csv_files) > 1:
        # Create comparison plot
        plot_multiple_experiments(csv_files, args.save_dir, not args.no_show)
    else:
        # Plot each file individually
        for csv_file in csv_files:
            print(f"\n{'='*60}")
            print(f"Plotting: {csv_file}")
            print('='*60)
            plot_mse_losses(csv_file, args.save_dir, not args.no_show)

if __name__ == "__main__":
    main()