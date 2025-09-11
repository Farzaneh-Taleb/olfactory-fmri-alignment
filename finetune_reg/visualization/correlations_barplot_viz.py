#!/usr/bin/env python3
"""
Script to filter, group, and visualize correlation results as barplots.
Supports optional p-value filtering and flexible grouping by different columns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

def load_csv_files(csv_paths: Union[str, List[str]]) -> pd.DataFrame:
    """Load and concatenate CSV files."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    dfs = []
    for path in csv_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {path}")
        else:
            print(f"Warning: File not found {path}")
    
    if not dfs:
        raise ValueError("No CSV files could be loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total combined rows: {len(combined_df)}")
    return combined_df

def parse_list_value(value_str: str) -> Union[str, List[str], int, List[int], float, List[float]]:
    """Parse comma-separated values into appropriate types."""
    if ',' in value_str:
        # List of values
        items = [item.strip() for item in value_str.split(',')]
        # Try to convert to numbers if possible
        try:
            return [int(item) for item in items]
        except ValueError:
            try:
                return [float(item) for item in items]
            except ValueError:
                return items
    else:
        # Single value
        try:
            val = float(value_str)
            return int(val) if val.is_integer() else val
        except ValueError:
            return value_str

def apply_filters(df: pd.DataFrame, p_value_threshold: Optional[float] = None, 
                 custom_filters: Optional[dict] = None) -> pd.DataFrame:
    """Apply optional p-value and custom filters to the dataframe."""
    filtered_df = df.copy()
    
    # P-value filtering
    if p_value_threshold is not None:
        if 'p_value_correlation' in filtered_df.columns:
            initial_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['p_value_correlation'] <= p_value_threshold]
            print(f"P-value filter (≤{p_value_threshold}): {initial_count} → {len(filtered_df)} rows")
        else:
            print("Warning: p_value_correlation column not found, skipping p-value filter")
    
    # Custom filters
    if custom_filters:
        for column, values in custom_filters.items():
            if column in filtered_df.columns:
                initial_count = len(filtered_df)
                if isinstance(values, (list, tuple)):
                    filtered_df = filtered_df[filtered_df[column].isin(values)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == values]
                print(f"Filter {column}={values}: {initial_count} → {len(filtered_df)} rows")
            else:
                print(f"Warning: Column '{column}' not found, skipping filter")
    
    return filtered_df

def create_grouped_barplot(df: pd.DataFrame, 
                          x_col: str, 
                          y_col: str = 'correlation',
                          hue_col: Optional[str] = None,
                          subplot_col: Optional[str] = None,
                          agg_func: str = 'mean',
                          figsize: Optional[tuple] = None,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None,
                          show_values: bool = True) -> plt.Figure:
    """Create grouped barplot of correlations with optional subplots."""
    
    # Group and aggregate data
    group_cols = [x_col]
    if hue_col:
        group_cols.append(hue_col)
    if subplot_col:
        group_cols.append(subplot_col)
    
    if agg_func == 'mean':
        grouped_df = df.groupby(group_cols)[y_col].mean().reset_index()
        y_label = f'Mean {y_col.title()}'
    elif agg_func == 'median':
        grouped_df = df.groupby(group_cols)[y_col].median().reset_index()
        y_label = f'Median {y_col.title()}'
    elif agg_func == 'max':
        grouped_df = df.groupby(group_cols)[y_col].max().reset_index()
        y_label = f'Max {y_col.title()}'
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")
    
    # Handle subplots
    if subplot_col:
        subplot_values = sorted(grouped_df[subplot_col].unique())
        n_subplots = len(subplot_values)
        
        # Calculate subplot layout
        n_cols = min(3, n_subplots)  # Max 3 columns
        n_rows = (n_subplots + n_cols - 1) // n_cols
        
        # Set figsize if not provided
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_subplots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Create subplot for each value
        for i, subplot_val in enumerate(subplot_values):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            subplot_data = grouped_df[grouped_df[subplot_col] == subplot_val]
            
            if hue_col:
                sns.barplot(data=subplot_data, x=x_col, y=y_col, hue=hue_col, ax=ax)
                if i == 0:  # Only show legend on first subplot
                    ax.legend(title=hue_col.title(), bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.legend().remove()
            else:
                bars = sns.barplot(data=subplot_data, x=x_col, y=y_col, ax=ax)
                
                # Add value labels on bars if requested
                if show_values:
                    for bar in bars.patches:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Formatting
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_label if col == 0 else '')
            ax.set_title(f'{subplot_col.title()}: {subplot_val}')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(n_subplots, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        # Overall title
        if title is None:
            title = f'{y_label} by {x_col.replace("_", " ").title()}'
            if hue_col:
                title += f' (grouped by {hue_col.replace("_", " ").title()})'
        fig.suptitle(title, fontsize=14, y=1.02)
        
    else:
        # Single plot (original behavior)
        if figsize is None:
            figsize = (12, 6)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if hue_col:
            sns.barplot(data=grouped_df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            ax.legend(title=hue_col.title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            bars = sns.barplot(data=grouped_df, x=x_col, y=y_col, ax=ax)
            
            # Add value labels on bars if requested
            if show_values:
                for bar in bars.patches:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_label)
        
        if title is None:
            title = f'{y_label} by {x_col.replace("_", " ").title()}'
            if hue_col:
                title += f' (grouped by {hue_col.replace("_", " ").title()})'
        ax.set_title(title)
        
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize correlation results as barplots')
    parser.add_argument('csv_files', nargs='+', help='CSV file(s) containing correlation results')
    parser.add_argument('--x-col', default='model', help='Column for x-axis grouping (default: model)')
    parser.add_argument('--y-col', default='correlation', help='Column for y-axis values (default: correlation)')
    parser.add_argument('--hue-col', help='Column for color grouping (optional)')
    parser.add_argument('--subplot-col', help='Column to create subplots by (optional)')
    parser.add_argument('--agg-func', choices=['mean', 'median', 'max'], default='mean', 
                       help='Aggregation function for grouped data (default: mean)')
    parser.add_argument('--p-threshold', type=float, help='Filter by p-value threshold (≤)')
    parser.add_argument('--filter-col', action='append', help='Column to filter by (format: col=value1,value2 for lists)')
    parser.add_argument('--figsize', nargs=2, type=int, help='Figure size (width height) - auto-calculated for subplots if not specified')
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--output', help='Save plot to file')
    parser.add_argument('--show-values', action='store_true', help='Show values on bars')
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = load_csv_files(args.csv_files)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    print(f"Available columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Parse custom filters
    custom_filters = {}
    if args.filter_col:
        for filter_str in args.filter_col:
            try:
                col, val_str = filter_str.split('=', 1)
                val = parse_list_value(val_str)
                custom_filters[col] = val
            except ValueError:
                print(f"Warning: Invalid filter format '{filter_str}', should be 'column=value' or 'column=val1,val2,val3'")
    
    # Apply filters
    filtered_df = apply_filters(df, args.p_threshold, custom_filters)
    
    if len(filtered_df) == 0:
        print("No data remaining after filtering!")
        return
    
    # Check if required columns exist
    if args.x_col not in filtered_df.columns:
        print(f"Error: Column '{args.x_col}' not found in data")
        return
    
    if args.y_col not in filtered_df.columns:
        print(f"Error: Column '{args.y_col}' not found in data")
        return
    
    if args.hue_col and args.hue_col not in filtered_df.columns:
        print(f"Error: Column '{args.hue_col}' not found in data")
        return
    
    if args.subplot_col and args.subplot_col not in filtered_df.columns:
        print(f"Error: Column '{args.subplot_col}' not found in data")
        return
    
    # Create plot
    try:
        fig = create_grouped_barplot(
            filtered_df,
            x_col=args.x_col,
            y_col=args.y_col,
            hue_col=args.hue_col,
            subplot_col=args.subplot_col,
            agg_func=args.agg_func,
            figsize=tuple(args.figsize) if args.figsize else None,
            title=args.title,
            save_path=args.output,
            show_values=args.show_values
        )
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return

if __name__ == "__main__":
    main()