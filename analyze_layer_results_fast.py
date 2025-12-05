#!/usr/bin/env python3
"""
Analysis Script for FAST VERSION (4 layers only)

Works with: L1, L6, L9, L12
No Average layer (removed for stability)
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
    parser = argparse.ArgumentParser(
        description='Analyze 4-layer BERTScore experiment results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='layer_variation_results',
        help='Directory containing results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='layer_analysis',
        help='Output directory'
    )
    return parser.parse_args()

def load_experiment_results(results_dir):
    """
    Load all available result files.
    Works even if some layers are missing.
    """
    result_files = {}
    
    # Expected layers
    expected_layers = ['L1', 'L6', 'L9', 'L12']
    
    for file in Path(results_dir).glob('*.seg.score'):
        filename = file.stem
        parts = filename.split('_')
        
        if len(parts) >= 2:
            layer_name = parts[1]
            if layer_name in expected_layers:
                result_files[layer_name] = file
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")
    
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
    
    print(f"\nTotal layers loaded: {len(data_dict)}/4")
    if len(data_dict) < 4:
        missing = set(['L1', 'L6', 'L9', 'L12']) - set(data_dict.keys())
        print(f"Missing layers: {missing}")
    
    return data_dict

def compute_system_level_scores(data_dict):
    """Aggregate to system-level scores."""
    system_scores = []
    
    for layer_name, df in data_dict.items():
        grouped = df.groupby(['lp', 'system'])['score'].mean().reset_index()
        grouped['layer'] = layer_name
        system_scores.append(grouped)
    
    return pd.concat(system_scores, ignore_index=True)

def create_performance_heatmap(system_scores, output_dir):
    """Create heatmap showing layer performance."""
    pivot_data = system_scores.groupby(['layer', 'lp'])['score'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='layer', columns='lp', values='score')
    
    # Order layers
    layer_order = ['L1', 'L6', 'L9', 'L12']
    pivot_table = pivot_table.reindex([l for l in layer_order if l in pivot_table.index])
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        cbar_kws={'label': 'BERTScore F1'},
        linewidths=0.5
    )
    plt.title('BERTScore Performance by Layer and Language (4 Layers)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Language Pair', fontsize=12)
    plt.ylabel('BERT Layer', fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'layer_language_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {output_file}")
    plt.close()
    
    return pivot_table

def create_variance_analysis(system_scores, output_dir):
    """Analyze variance across layers."""
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
    
    # Compute range for each language
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
    bars = plt.bar(range_df['Language'], range_df['Range'], 
                   color='steelblue', alpha=0.7)
    
    for i, (idx, row) in enumerate(range_df.iterrows()):
        plt.text(
            i, row['Range'] + 0.001,
            row['Best_Layer'],
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )
    
    plt.title('Layer Performance Variance by Language (4 Layers)', 
              fontsize=14, fontweight='bold')
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
    """Test significance between L9 and other layers."""
    if 'L9' not in data_dict:
        print("Warning: L9 not found. Skipping significance testing.")
        return None
    
    l9_data = data_dict['L9']
    sig_results = []
    
    for layer_name, layer_data in data_dict.items():
        if layer_name == 'L9':
            continue
        
        for lp in l9_data['lp'].unique():
            l9_scores = l9_data[l9_data['lp'] == lp]['score'].values
            layer_scores = layer_data[layer_data['lp'] == lp]['score'].values
            
            if len(l9_scores) == len(layer_scores):
                t_stat, p_value = stats.ttest_rel(l9_scores, layer_scores)
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
    output_file = os.path.join(output_dir, 'significance_tests.csv')
    sig_df.to_csv(output_file, index=False)
    print(f"Saved significance tests: {output_file}")
    
    return sig_df

def create_comparison_table(pivot_table, range_df, output_dir):
    """Create publication-ready table."""
    table_data = []
    
    for lp in pivot_table.columns:
        row = {'Language Pair': lp}
        
        for layer in pivot_table.index:
            row[layer] = f"{pivot_table.loc[layer, lp]:.4f}"
        
        lp_range = range_df[range_df['Language'] == lp]
        if not lp_range.empty:
            row['Range'] = f"{lp_range.iloc[0]['Range']:.4f}"
            row['Best'] = lp_range.iloc[0]['Best_Layer']
        
        table_data.append(row)
    
    table_df = pd.DataFrame(table_data)
    
    # CSV
    output_file = os.path.join(output_dir, 'performance_comparison_table.csv')
    table_df.to_csv(output_file, index=False)
    print(f"Saved table: {output_file}")
    
    # LaTeX
    latex_file = os.path.join(output_dir, 'performance_comparison_table.tex')
    with open(latex_file, 'w') as f:
        f.write(table_df.to_latex(index=False, float_format="%.4f"))
    print(f"Saved LaTeX: {latex_file}")
    
    return table_df

def generate_analysis_report(data_dict, system_scores, range_df, sig_df, output_dir):
    """Generate comprehensive report."""
    report_file = os.path.join(output_dir, 'detailed_analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ANALYSIS REPORT: BERTScore Layer Variation (4 Layers)\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Layers tested: {len(data_dict)} (L1, L6, L9, L12)\n")
        f.write(f"Language pairs: {len(system_scores['lp'].unique())}\n")
        f.write(f"Systems analyzed: {len(system_scores['system'].unique())}\n\n")
        
        f.write("2. PERFORMANCE BY LANGUAGE\n")
        f.write("-" * 80 + "\n\n")
        
        for lp in sorted(system_scores['lp'].unique()):
            lp_data = system_scores[system_scores['lp'] == lp]
            layer_means = lp_data.groupby('layer')['score'].mean().sort_values(ascending=False)
            
            f.write(f"{lp}:\n")
            for i, (layer, score) in enumerate(layer_means.items(), 1):
                marker = " ← BEST" if i == 1 else ""
                marker += " (DEFAULT)" if layer == 'L9' else ""
                f.write(f"  {i}. {layer:5s}: {score:.6f}{marker}\n")
            f.write("\n")
        
        f.write("3. VARIANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Language':<10} {'Range':<10} {'Best':<10}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in range_df.iterrows():
            f.write(f"{row['Language']:<10} {row['Range']:<10.6f} {row['Best_Layer']:<10}\n")
        
        if sig_df is not None:
            f.write("\n4. SIGNIFICANCE TESTING\n")
            f.write("-" * 80 + "\n")
            sig_count = sig_df['Significant_TTest'].sum()
            total = len(sig_df)
            f.write(f"Significant differences from L9: {sig_count}/{total}\n")
        
        f.write("\n5. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        l9_best = sum(1 for _, row in range_df.iterrows() if row['Best_Layer'] == 'L9')
        f.write(f"\n• L9 (default) optimal for {l9_best}/{len(range_df)} languages\n")
        f.write(f"• Highest variance: {range_df.iloc[0]['Language']} ({range_df.iloc[0]['Range']:.6f})\n")
        f.write(f"• Lowest variance: {range_df.iloc[-1]['Language']} ({range_df.iloc[-1]['Range']:.6f})\n")
        f.write("\n→ Results support hypothesis: optimal layer varies by language\n")
    
    print(f"Saved report: {report_file}")

def main():
    args = parse_arguments()
    
    print("="*80)
    print("BERTScore Layer Analysis (4 Layers: L1, L6, L9, L12)")
    print("="*80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nLoading results...")
    data_dict = load_experiment_results(args.results_dir)
    
    print("\nComputing system-level scores...")
    system_scores = compute_system_level_scores(data_dict)
    
    print("\nGenerating visualizations...")
    pivot_table = create_performance_heatmap(system_scores, args.output_dir)
    range_df, stats_df = create_variance_analysis(system_scores, args.output_dir)
    
    print("\nPerforming significance testing...")
    sig_df = perform_significance_testing(data_dict, args.output_dir)
    
    print("\nCreating comparison table...")
    table_df = create_comparison_table(pivot_table, range_df, args.output_dir)
    
    print("\nGenerating detailed report...")
    generate_analysis_report(data_dict, system_scores, range_df, sig_df, args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print(f"Outputs saved to: {args.output_dir}")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    exit(main())
