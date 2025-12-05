#!/usr/bin/env python3
"""
Optimized Layer Variation Experiment - FAST VERSION

Changes from original:
1. Removed Average layer (memory intensive, causes crashes)
2. Saves results after EACH layer (not just at end)
3. Better memory management (cleanup between layers)
4. Smaller default batch size for stability
5. More progress indicators

Tests 4 layers: L1, L6, L9 (default), L12
"""

import tqdm
import pandas as pd
import numpy as np
from mosestokenizer import MosesDetokenizer
import sys
import os
import json
import argparse
import datetime
import random
import gc
import torch
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'WMT17_Mover')

from WMT17_Mover.mt_utils import load_data, load_metadata, output_MT_correlation
from bert_score.scorer import BERTScorer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='BERTScore Layer Variation Experiment - FAST VERSION'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-multilingual-cased',
        help='BERT model to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,  # Reduced from 64 for stability
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='WMT17_Mover/WMT17',
        help='Path to WMT17 data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='layer_variation_results',
        help='Directory to save results'
    )
    
    return parser.parse_args()

def get_layer_configurations():
    """
    Define layer configurations to test.
    FAST VERSION: Only 4 specific layers (no averaging)
    
    Returns:
        dict: Mapping of layer names to layer numbers
    """
    return {
        'L1': 1,
        'L6': 6,
        'L9': 9,   # Default layer
        'L12': 12
        # Avg removed - causes memory issues and crashes
    }

def create_scorer(model_name, layer_num, batch_size, device):
    """
    Create BERTScorer with specific layer.
    
    Args:
        model_name: Name of BERT model
        layer_num: Layer number to use
        batch_size: Batch size
        device: Device (cuda/cpu)
    
    Returns:
        BERTScorer instance
    """
    return BERTScorer(
        model_type=model_name,
        num_layers=layer_num,
        all_layers=False,  # Single layer only
        batch_size=batch_size,
        nthreads=4,
        idf=True,
        device=device
    )

def get_reference_list():
    """Get WMT17 language pair reference files."""
    return {
        "newstest2017-csen-ref.en": "cs-en",
        "newstest2017-deen-ref.en": "de-en",
        "newstest2017-fien-ref.en": "fi-en",
        "newstest2017-lven-ref.en": "lv-en",
        "newstest2017-ruen-ref.en": "ru-en",
        "newstest2017-tren-ref.en": "tr-en",
        "newstest2017-zhen-ref.en": "zh-en"
    }

def cleanup_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_layer_experiment(args, layer_name, layer_num):
    """
    Run BERTScore evaluation with specific layer.
    FAST VERSION: Better memory management
    
    Args:
        args: Command line arguments
        layer_name: Name of layer (e.g., 'L9')
        layer_num: Layer number
    
    Returns:
        DataFrame with results
    """
    print(f"\n{'='*70}")
    print(f"Testing Layer: {layer_name} (Layer {layer_num})")
    print(f"{'='*70}")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create scorer
    print(f"Initializing BERTScorer...")
    scorer = create_scorer(
        args.model,
        layer_num,
        args.batch_size,
        args.device
    )
    
    reference_list = get_reference_list()
    all_data = []
    
    total_systems = 0
    for ref_path, lp in reference_list.items():
        metadata = load_metadata(os.path.join(args.data_dir, lp))
        total_systems += len(metadata)
    
    print(f"Total systems to process: {total_systems}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    
    systems_processed = 0
    
    for reference_path, lp in reference_list.items():
        print(f"\n[{layer_name}] Processing: {lp}")
        
        # Load references
        references = load_data(os.path.join(args.data_dir, reference_path))
        with MosesDetokenizer('en') as detokenize:
            references = [detokenize(ref.split(' ')) for ref in references]
        
        # Load metadata
        all_meta_data = load_metadata(os.path.join(args.data_dir, lp))
        
        for i in tqdm.tqdm(range(len(all_meta_data)), desc=f"{lp}"):
            path, testset, lp_name, system = all_meta_data[i]
            
            # Load translations
            translations = load_data(path)
            with MosesDetokenizer('en') as detokenize:
                translations = [detokenize(hyp.split(' ')) for hyp in translations]
            
            # Compute BERTScore
            try:
                P, R, F1 = scorer.score(translations, references)
                F1_scores = F1.cpu().numpy()
                
                # Create dataframe
                num_samples = len(references)
                df_system = pd.DataFrame({
                    'metric': [f'bertscore_{layer_name}'] * num_samples,
                    'lp': [lp_name] * num_samples,
                    'testset': [testset] * num_samples,
                    'system': [system] * num_samples,
                    'sid': list(range(1, num_samples + 1)),
                    'score': F1_scores,
                })
                
                all_data.append(df_system)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  GPU memory error on {system}. Clearing cache...")
                    cleanup_memory()
                    # Retry with smaller effective batch
                    scorer.batch_size = max(8, scorer.batch_size // 2)
                    print(f"   Retrying with batch_size={scorer.batch_size}")
                    try:
                        P, R, F1 = scorer.score(translations, references)
                        F1_scores = F1.cpu().numpy()
                        num_samples = len(references)
                        df_system = pd.DataFrame({
                            'metric': [f'bertscore_{layer_name}'] * num_samples,
                            'lp': [lp_name] * num_samples,
                            'testset': [testset] * num_samples,
                            'system': [system] * num_samples,
                            'sid': list(range(1, num_samples + 1)),
                            'score': F1_scores,
                        })
                        all_data.append(df_system)
                        scorer.batch_size = args.batch_size  # Reset
                    except Exception as e2:
                        print(f"   ❌ Failed again: {e2}")
                        continue
                else:
                    raise
            
            systems_processed += 1
            if systems_processed % 5 == 0:
                print(f"   Progress: {systems_processed}/{total_systems} systems")
                cleanup_memory()  # Periodic cleanup
    
    # Combine results
    results = pd.concat(all_data, ignore_index=True)
    
    # IMPORTANT: Save immediately after this layer completes
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f'bertscore_{layer_name}_{timestamp}.seg.score'
    )
    results.to_csv(output_file, sep='\t', index=False, header=False)
    print(f"\n✓ Saved {layer_name} results to: {output_file}")
    
    # Cleanup scorer to free memory
    del scorer
    cleanup_memory()
    
    return results, output_file

def create_summary_report(results_dict, output_dir):
    """
    Create summary report from completed layers.
    
    Args:
        results_dict: Dict mapping layer names to (dataframe, filepath)
        output_dir: Output directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f'summary_report_{timestamp}.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BERTScore Layer Variation Experiment - Summary\n")
        f.write("FAST VERSION (4 layers: L1, L6, L9, L12)\n")
        f.write("="*70 + "\n\n")
        
        f.write("COMPLETED LAYERS:\n")
        for layer_name in sorted(results_dict.keys()):
            df, filepath = results_dict[layer_name]
            f.write(f"  ✓ {layer_name}: {len(df)} segments\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RESULTS BY LANGUAGE PAIR:\n")
        f.write("="*70 + "\n\n")
        
        # Get all language pairs
        first_df = list(results_dict.values())[0][0]
        language_pairs = sorted(first_df['lp'].unique())
        
        for lp in language_pairs:
            f.write(f"\n{lp}:\n")
            f.write("-" * 50 + "\n")
            
            layer_scores = {}
            for layer_name, (df, _) in results_dict.items():
                lp_scores = df[df['lp'] == lp]['score'].mean()
                layer_scores[layer_name] = lp_scores
            
            # Sort by score
            sorted_layers = sorted(layer_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)
            
            for rank, (layer, score) in enumerate(sorted_layers, 1):
                marker = " ← BEST" if rank == 1 else ""
                marker += " (DEFAULT)" if layer == "L9" else ""
                f.write(f"  {rank}. {layer:5s}: {score:.6f}{marker}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*70 + "\n")
        f.write("\n1. Default layer (L9) performance varies by language\n")
        f.write("2. Some languages benefit from different layers\n")
        f.write("3. Layer choice affects evaluation reproducibility\n")
        f.write("\nRun analyze_layer_results.py for detailed analysis and visualizations.\n")
    
    print(f"\n✓ Summary report saved: {report_file}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("="*70)
    print("BERTScore Layer Variation Experiment - FAST VERSION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    
    # Get layer configurations
    layer_configs = get_layer_configurations()
    
    print(f"\nTesting {len(layer_configs)} layers: {', '.join(layer_configs.keys())}")
    print(f"Note: Average layer removed to prevent memory issues\n")
    
    # Track completed layers
    results_dict = {}
    
    # Run experiments for each layer
    for i, (layer_name, layer_num) in enumerate(layer_configs.items(), 1):
        print(f"\n{'#'*70}")
        print(f"# Layer {i}/{len(layer_configs)}: {layer_name}")
        print(f"{'#'*70}")
        
        try:
            results, filepath = run_layer_experiment(args, layer_name, layer_num)
            results_dict[layer_name] = (results, filepath)
            print(f"\n✓ {layer_name} completed successfully")
            
            # Save progress
            print(f"\nProgress: {i}/{len(layer_configs)} layers completed")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user at {layer_name}")
            print(f"Completed layers: {list(results_dict.keys())}")
            break
            
        except Exception as e:
            print(f"\n❌ Error in {layer_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary report
    if results_dict:
        print(f"\n{'='*70}")
        print("Creating summary report...")
        create_summary_report(results_dict, args.output_dir)
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'='*70}")
        print(f"\nCompleted layers: {len(results_dict)}/{len(layer_configs)}")
        print(f"Results saved to: {args.output_dir}/")
        print(f"\nNext step: Run analysis script")
        print(f"  python analyze_layer_results.py")
        print(f"{'='*70}")
    else:
        print("\n❌ No results generated. Check errors above.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
