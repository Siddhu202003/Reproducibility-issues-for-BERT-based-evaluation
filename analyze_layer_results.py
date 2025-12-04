#!/usr/bin/env python3
"""
Analysis and Visualization Script for BERTScore Layer Variation Experiment

This script processes the output from WMT17_layer_variation_experiment.py and generates:
- Statistical comparisons across layers
- Visualization (heatmaps, bar charts)
- Publication-ready tables
- Significance testing results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy import stats
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze BERTScore layer variation experiment results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='layer_variation_results',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='layer_analysis',
        help='Directory to save analysis outputs'
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Specific timestamp to analyze (optional)'
    )
    
    return parser.parse_args()

def load_experiment_results(results_dir, timestamp=None):
    """
    Load all result files from experiment.
    
    Args:
        results_dir: Directory containing result files
        timestamp: Specific timestamp to load (if None, uses most recent)
    
    Returns:
        Dictionary mapping layer names to DataFrames
    """
    result_files = {}
    
    # Find all .seg.score files
    for file in Path(results_dir).glob('*.seg.score'):
        filename = file.stem
        
        # Extract layer name (e.g., 'bertscore_L9_20231204_120000')
        parts = filename.split('_')
        if len(parts) >= 2:
            layer_name = parts[1]  # L1, L6, L9, L12, or Avg
            file_timestamp = '_'.join(parts[2:]) if len(parts) > 2 else ''
            
            # Filter by timestamp if specified
            if timestamp is None or file_timestamp.startswith(timestamp):
                result_files[layer_name] = file
    
    # Load data
    data_dict = {}
    for layer_name, file_path in result_files.items():
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['metric', 'lp', 'testset', 'system', 'sid', 'score']
        )
        data_dict[layer_name] = df
        print(f"Loaded {layer_name}: {len(df)} samples")
    
    if not data_dict:
        raise ValueError(f"No result files found in {results_dir}")
    
    return data_dict

def compute_system_level_scores(data_dict):
    """
    Aggregate segment-level scores to system-level.
    
    Args:
        data_dict: Dictionary mapping layer names to DataFrames
    
    Returns:
        DataFrame with system-level scores
    """
    system_scores = []
    
    for layer_name, df in data_dict.items():
        # Group by language pair and system
        grouped = df.groupby(['lp', 'system'])['score'].mean().reset_index()
        grouped['layer'] = layer_name
        system_scores.append(grouped)
    
    return pd.concat(system_scores, ignore_index=True)

def create_performance_heatmap(system_scores, output_dir):
    """
    Create heatmap showing layer performance across languages.
    
    Args:
        system_scores: DataFrame with system-level scores
        output_dir: Directory to save plot
    """
    # Compute mean score for each layer/language pair
    pivot_data = system_scores.groupby(['layer', 'lp'])['score'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='layer', columns='lp', values='score')
    
    # Order layers logically
    layer_order = ['L1', 'L6', 'L9', 'L12', 'Avg']
    pivot_table = pivot_table.reindex([l for l in layer_order if l in pivot_table.index])
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        cbar_kws={'label': 'BERTScore F1'},
        linewidths=0.5
    )
    plt.title('BERTScore Performance by Layer and Language Pair', fontsize=14, fontweight='bold')
    plt.xlabel('Language Pair', fontsize=12)
    plt.ylabel('BERT Layer', fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'layer_language_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {output_file}")
    plt.close()
    
    return pivot_table

def create_variance_analysis(system_scores, output_dir):
    """
    Analyze variance across layers for each language.
    
    Args:
        system_scores: DataFrame with system-level scores
        output_dir: Directory to save plot
    """
    # Compute statistics for each language pair
    stats_data = []
    
    for lp in system_scores['lp'].unique():
        lp_data = system_scores[system_scores['lp'] == lp]
        
        for layer in lp_data['layer'].unique():
            layer_data = lp_data[lp_data['layer'] == layer]['score']
            
            stats_data.append({
                'Language': lp,
                'Layer': layer,
                'Mean': layer_data.mean(),
                'Std': layer_data.std(),
                'Min': layer_data.min(),
                'Max': layer_data.max()
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Compute range (max - min) for each language
    range_data = []
    for lp in stats_df['Language'].unique():
        lp_stats = stats_df[stats_df['Language'] == lp]
        layer_range = lp_stats['Mean'].max() - lp_stats['Mean'].min()
        range_data.append({
            'Language': lp,
            'Range': layer_range,
            'Best_Layer': lp_stats.loc[lp_stats['Mean'].idxmax(), 'Layer'],
            'Best_Score': lp_stats['Mean'].max()
        })
    
    range_df = pd.DataFrame(range_data).sort_values('Range', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range_df['Language'], range_df['Range'], color='steelblue', alpha=0.7)
    
    # Add best layer labels on bars
    for i, (idx, row) in enumerate(range_df.iterrows()):
        plt.text(
            i, row['Range'] + 0.001,
            row['Best_Layer'],
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )
    
    plt.title('Layer Performance Variance by Language Pair', fontsize=14, fontweight='bold')
    plt.xlabel('Language Pair', fontsize=12)
    plt.ylabel('Score Range (Max - Min)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'layer_variance_by_language.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved variance plot: {output_file}")
    plt.close()
    
    return range_df, stats_df

def perform_significance_testing(data_dict, output_dir):
    """
    Perform statistical significance testing between layers.
    
    Args:
        data_dict: Dictionary mapping layer names to DataFrames
        output_dir: Directory to save results
    
    Returns:
        DataFrame with significance test results
    """
    # Compare L9 (default) against other layers for each language
    if 'L9' not in data_dict:
        print("Warning: L9 (default) not found in results. Skipping significance testing.")
        return None
    
    l9_data = data_dict['L9']
    sig_results = []
    
    for layer_name, layer_data in data_dict.items():
        if layer_name == 'L9':
            continue
        
        for lp in l9_data['lp'].unique():
            l9_scores = l9_data[l9_data['lp'] == lp]['score'].values
            layer_scores = layer_data[layer_data['lp'] == lp]['score'].values
            
            # Paired t-test (same segments compared)
            if len(l9_scores) == len(layer_scores):
                t_stat, p_value = stats.ttest_rel(l9_scores, layer_scores)
                
                # Wilcoxon signed-rank test (non-parametric alternative)
                w_stat, w_pvalue = stats.wilcoxon(l9_scores, layer_scores)
                
                mean_diff = layer_scores.mean() - l9_scores.mean()
                
                sig_results.append({
                    'Language': lp,
                    'Comparison': f'L9 vs {layer_name}',
                    'Mean_Difference': mean_diff,
                    'T_Statistic': t_stat,
                    'P_Value_TTest': p_value,
                    'Significant_TTest': p_value < 0.05,
                    'Wilcoxon_Stat': w_stat,
                    'P_Value_Wilcoxon': w_pvalue,
                    'Significant_Wilcoxon': w_pvalue < 0.05
                })
    
    sig_df = pd.DataFrame(sig_results)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'significance_tests.csv')
    sig_df.to_csv(output_file, index=False)
    print(f"Saved significance tests: {output_file}")
    
    return sig_df

def create_comparison_table(pivot_table, range_df, output_dir):
    """
    Create publication-ready comparison table.
    
    Args:
        pivot_table: Pivot table from heatmap
        range_df: DataFrame with variance analysis
        output_dir: Directory to save table
    """
    # Combine data
    table_data = []
    
    for lp in pivot_table.columns:
        row = {'Language Pair': lp}
        
        # Add scores for each layer
        for layer in pivot_table.index:
            row[layer] = f"{pivot_table.loc[layer, lp]:.4f}"
        
        # Add range and best layer
        lp_range = range_df[range_df['Language'] == lp]
        if not lp_range.empty:
            row['Range'] = f"{lp_range.iloc[0]['Range']:.4f}"
            row['Best'] = lp_range.iloc[0]['Best_Layer']
        
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    output_file = os.path.join(output_dir, 'performance_comparison_table.csv')
    table_df.to_csv(output_file, index=False)
    print(f"Saved comparison table: {output_file}")
    
    # Save as LaTeX
    latex_file = os.path.join(output_dir, 'performance_comparison_table.tex')
    with open(latex_file, 'w') as f:
        f.write(table_df.to_latex(index=False, float_format="%.4f"))
    print(f"Saved LaTeX table: {latex_file}")
    
    return table_df

def generate_analysis_report(data_dict, system_scores, range_df, sig_df, output_dir):
    """
    Generate comprehensive analysis report.
    
    Args:
        data_dict: Dictionary of raw data
        system_scores: System-level aggregated scores
        range_df: Variance analysis results
        sig_df: Significance testing results
        output_dir: Directory to save report
    """
    report_file = os.path.join(output_dir, 'detailed_analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED ANALYSIS REPORT: BERTScore Layer Variation Experiment\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("1. OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total layers tested: {len(data_dict)}\n")
        f.write(f"Language pairs evaluated: {len(system_scores['lp'].unique())}\n")
        f.write(f"Total systems analyzed: {len(system_scores['system'].unique())}\n\n")
        
        # Layer performance ranking
        f.write("2. LAYER PERFORMANCE BY LANGUAGE\n")
        f.write("-" * 80 + "\n\n")
        
        for lp in sorted(system_scores['lp'].unique()):
            lp_data = system_scores[system_scores['lp'] == lp]
            layer_means = lp_data.groupby('layer')['score'].mean().sort_values(ascending=False)
            
            f.write(f"Language Pair: {lp}\n")
            for i, (layer, score) in enumerate(layer_means.items(), 1):
                f.write(f"  {i}. {layer:5s}: {score:.6f}")
                if i == 1:
                    f.write(" ← BEST")
                elif layer == 'L9':
                    f.write(" ← DEFAULT")
                f.write("\n")
            f.write("\n")
        
        # Variance analysis
        f.write("3. LAYER SENSITIVITY (Variance) BY LANGUAGE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Language':<10} {'Range':<10} {'Best Layer':<12} {'Default (L9) Rank'}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in range_df.iterrows():
            lp = row['Language']
            lp_data = system_scores[system_scores['lp'] == lp]
            layer_means = lp_data.groupby('layer')['score'].mean().sort_values(ascending=False)
            l9_rank = list(layer_means.index).index('L9') + 1 if 'L9' in layer_means.index else 'N/A'
            
            f.write(f"{lp:<10} {row['Range']:<10.6f} {row['Best_Layer']:<12} {l9_rank}\n")
        
        f.write("\n")
        
        # Significance testing summary
        if sig_df is not None:
            f.write("4. STATISTICAL SIGNIFICANCE (L9 vs Other Layers)\n")
            f.write("-" * 80 + "\n")
            
            sig_summary = sig_df.groupby('Language').agg({
                'Significant_TTest': 'sum',
                'Significant_Wilcoxon': 'sum'
            }).reset_index()
            
            f.write(f"{'Language':<10} {'Significant Differences (T-Test)':<35} {'(Wilcoxon)'}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in sig_summary.iterrows():
                f.write(f"{row['Language']:<10} {row['Significant_TTest']}/4{'':<29} {row['Significant_Wilcoxon']}/4\n")
            
            f.write("\n")
        
        # Key findings
        f.write("5. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Finding 1: Is L9 always best?
        l9_best_count = sum(1 for _, row in range_df.iterrows() if row['Best_Layer'] == 'L9')
        f.write(f"\n• Default layer (L9) is optimal for {l9_best_count}/{len(range_df)} language pairs\n")
        
        if l9_best_count < len(range_df):
            f.write("  → This confirms our hypothesis: optimal layer varies by language!\n")
        
        # Finding 2: Highest variance language
        max_var_lang = range_df.iloc[0]['Language']
        max_var_val = range_df.iloc[0]['Range']
        f.write(f"\n• Highest layer sensitivity: {max_var_lang} (range: {max_var_val:.6f})\n")
        f.write("  → This language shows strongest layer-dependent effects\n")
        
        # Finding 3: Most stable language
        min_var_lang = range_df.iloc[-1]['Language']
        min_var_val = range_df.iloc[-1]['Range']
        f.write(f"\n• Lowest layer sensitivity: {min_var_lang} (range: {min_var_val:.6f})\n")
        f.write("  → This language shows most consistent performance across layers\n")
        
        # Finding 4: Layer averaging
        if 'Avg' in data_dict:
            f.write("\n• Layer averaging (Avg) performance:\n")
            avg_ranks = []
            for lp in system_scores['lp'].unique():
                lp_data = system_scores[system_scores['lp'] == lp]
                layer_means = lp_data.groupby('layer')['score'].mean().sort_values(ascending=False)
                if 'Avg' in layer_means.index:
                    rank = list(layer_means.index).index('Avg') + 1
                    avg_ranks.append(rank)
            
            if avg_ranks:
                mean_rank = np.mean(avg_ranks)
                f.write(f"  Mean rank across languages: {mean_rank:.2f}\n")
                f.write(f"  → {'Good' if mean_rank <= 3 else 'Moderate'} trade-off between stability and performance\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Saved detailed report: {report_file}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("BERTScore Layer Variation Analysis")
    print("="*80)
    print(f"\nResults directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load experimental results
    print("\nLoading experimental results...")
    data_dict = load_experiment_results(args.results_dir, args.timestamp)
    
    # Compute system-level scores
    print("\nComputing system-level aggregations...")
    system_scores = compute_system_level_scores(data_dict)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    pivot_table = create_performance_heatmap(system_scores, args.output_dir)
    range_df, stats_df = create_variance_analysis(system_scores, args.output_dir)
    
    # Perform significance testing
    print("\nPerforming statistical significance testing...")
    sig_df = perform_significance_testing(data_dict, args.output_dir)
    
    # Create comparison table
    print("\nCreating comparison tables...")
    table_df = create_comparison_table(pivot_table, range_df, args.output_dir)
    
    # Generate comprehensive report
    print("\nGenerating detailed analysis report...")
    generate_analysis_report(data_dict, system_scores, range_df, sig_df, args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print(f"All outputs saved to: {args.output_dir}")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    exit(main())
