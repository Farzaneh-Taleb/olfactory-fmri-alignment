# Correlation Results Visualization

Script to filter, group, and visualize correlation results from your regression analyses as barplots.

## Usage

```bash
python correlations_barplot_viz.py path/to/metrics.csv [options]
```

## Examples

### Basic usage - plot correlations by model
```bash
python correlations_barplot_viz.py ../0_chem_behavior/*metrics*.csv --x-col model
```

### Group by layer with p-value filtering
```bash
python correlations_barplot_viz.py ../0_chem_fmri/*metrics*.csv --x-col layer --p-threshold 0.05
```

### Compare models across datasets with color grouping
```bash
python correlations_barplot_viz.py ../*/*metrics*.csv --x-col model --hue-col ds --agg-func median
```

### Filter specific participant and show values on bars
```bash
python correlations_barplot_viz.py ../4_finetunedchem_fmri/*metrics*.csv \
  --x-col roi --filter-col participant_id=1 --show-values
```

### Multiple filters and save plot
```bash
python correlations_barplot_viz.py ../*/*metrics*.csv \
  --x-col layer --hue-col model \
  --filter-col ds=keller --filter-col n_components=50 \
  --p-threshold 0.01 --output correlation_plot.png
```

### Create subplots by participant with list filtering
```bash
python correlations_barplot_viz.py ../*/*metrics*.csv \
  --x-col model --subplot-col participant_id \
  --filter-col layer=1,2,3 --filter-col ds=keller,ravia
```

### Filter multiple models and ROIs with subplots
```bash
python correlations_barplot_viz.py ../4_finetunedchem_fmri/*metrics*.csv \
  --x-col layer --hue-col model --subplot-col roi \
  --filter-col model=bert-base,roberta-base --filter-col participant_id=1,2,3
```

## Key Options

- `--x-col`: Column for x-axis grouping (default: model)
- `--y-col`: Column for y-axis values (default: correlation)  
- `--hue-col`: Column for color grouping (optional)
- `--subplot-col`: Column to create subplots by (optional)
- `--agg-func`: How to aggregate data (mean/median/max, default: mean)
- `--p-threshold`: Filter by p-value threshold (≤)
- `--filter-col`: Filter by specific values (format: col=value or col=val1,val2,val3 for lists)
- `--output`: Save plot to file
- `--show-values`: Show numeric values on bars
- `--figsize`: Figure size (width height) - auto-calculated for subplots if not specified

## Common Columns

Based on your regression scripts, available columns typically include:
- `target`, `correlation`, `mse`, `p_value_correlation`, `p_value_mse`
- `model`, `ds`, `participant_id`, `layer`, `n_fold`, `n_components` 
- `roi`, `tr` (fMRI only)
- `z_score`, `date`, `run_id`