# Quick Start Guide: BERTScore Layer Variation Experiment

This guide will walk you through running the complete layer variation experiment from start to finish.

## Prerequisites

### 1. Environment Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/Siddhu202003/Reproducibility-issues-for-BERT-based-evaluation.git
cd Reproducibility-issues-for-BERT-based-evaluation

# Install dependencies
pip install -r requirements.txt

# Verify CUDA availability (recommended but not required)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Verify Data

```bash
# Check WMT17 data is present
ls WMT17_Mover/WMT17/

# You should see reference files like:
# newstest2017-csen-ref.en
# newstest2017-deen-ref.en
# newstest2017-fien-ref.en
# ... etc.
```

## Step-by-Step Workflow

### Step 1: Run the Experiment (~30-60 min with GPU, ~3-5 hours with CPU)

```bash
# Basic usage (uses default settings)
python WMT17_layer_variation_experiment.py

# OR with custom settings:
python WMT17_layer_variation_experiment.py \
    --model bert-base-multilingual-cased \
    --batch_size 64 \
    --device cuda \
    --output_dir layer_variation_results
```

**What happens:**
- Tests BERTScore with 5 layer configurations (L1, L6, L9, L12, Average)
- Evaluates on 7 language pairs: cs-en, de-en, fi-en, lv-en, ru-en, tr-en, zh-en
- Processes all MT systems in WMT17 dataset
- Saves detailed results to `layer_variation_results/`

**Expected output files:**
```
layer_variation_results/
  ├── bertscore_L1_YYYYMMDD_HHMMSS.seg.score
  ├── bertscore_L6_YYYYMMDD_HHMMSS.seg.score
  ├── bertscore_L9_YYYYMMDD_HHMMSS.seg.score
  ├── bertscore_L12_YYYYMMDD_HHMMSS.seg.score
  ├── bertscore_Avg_YYYYMMDD_HHMMSS.seg.score
  ├── layer_correlation_analysis_YYYYMMDD_HHMMSS.csv
  └── summary_report_YYYYMMDD_HHMMSS.txt
```

### Step 2: Analyze Results (~5 min)

```bash
# Analyze and visualize results
python analyze_layer_results.py

# OR specify directories:
python analyze_layer_results.py \
    --results_dir layer_variation_results \
    --output_dir layer_analysis
```

**What happens:**
- Loads all experimental results
- Computes statistical comparisons
- Generates visualizations (heatmaps, bar charts)
- Performs significance testing
- Creates publication-ready tables

**Expected output files:**
```
layer_analysis/
  ├── layer_language_heatmap.png          # Performance heatmap
  ├── layer_variance_by_language.png      # Variance analysis
  ├── significance_tests.csv              # Statistical tests
  ├── performance_comparison_table.csv    # Summary table (CSV)
  ├── performance_comparison_table.tex    # Summary table (LaTeX)
  └── detailed_analysis_report.txt        # Full analysis report
```

### Step 3: Review Results

```bash
# View the summary report
cat layer_variation_results/summary_report_*.txt

# View detailed analysis
cat layer_analysis/detailed_analysis_report.txt

# View visualizations (opens in default image viewer)
open layer_analysis/layer_language_heatmap.png
open layer_analysis/layer_variance_by_language.png
```

## Understanding the Results

### Key Files to Examine

1. **summary_report_*.txt**: Quick overview of results by language
   - Shows which layer performed best for each language
   - Highlights key findings

2. **detailed_analysis_report.txt**: Comprehensive analysis
   - Performance rankings for each language
   - Variance analysis (which languages show strongest layer effects)
   - Statistical significance summary
   - Key findings with context

3. **layer_language_heatmap.png**: Visual comparison
   - Rows = Layers (L1, L6, L9, L12, Avg)
   - Columns = Language pairs
   - Color intensity = BERTScore F1 performance
   - Quickly see which layer-language combinations perform best

4. **layer_variance_by_language.png**: Sensitivity analysis
   - Bar height = performance range (max - min layer score)
   - Higher bars = more sensitive to layer choice
   - Labels show which layer performed best

5. **significance_tests.csv**: Statistical validation
   - Compares default layer (L9) against others
   - P-values indicate if differences are statistically significant
   - Both parametric (T-test) and non-parametric (Wilcoxon) tests

### Interpreting Results for Your Report

**Does the hypothesis hold?**
- Check if L9 (default) is optimal for all languages
- Look for languages where other layers significantly outperform L9
- Examine variance - high variance supports hypothesis

**Which languages are most affected?**
- Check variance plot - languages with high bars
- Morphologically rich languages (fi, cs, lv, tr) may show higher variance
- Logographic languages (zh) may have different optimal layer

**Quantify the effect:**
- Use performance_comparison_table.csv for exact numbers
- Report score ranges for each language
- Compare to preprocessing effects from original paper

## Writing Your Report

### Recommended Structure

1. **Title & Abstract**
   - State hypothesis clearly
   - Summarize methodology (fixed metric/dataset, varied layer)
   - Preview key findings

2. **Introduction**
   - Contextualize within original paper's findings
   - Identify gap: layer selection understudied
   - State research question

3. **Methodology**
   - **HYPOTHESIS** (in BOLD): "The optimal BERT layer for computing embeddings in BERTScore varies significantly across different languages..."
   - Experimental design (what's fixed, what varies)
   - Layer configurations tested (L1, L6, L9, L12, Avg)
   - Evaluation metrics (F1 scores, correlations)

4. **Results**
   - Include heatmap visualization
   - Present performance comparison table
   - Show variance analysis
   - Report significance test outcomes

5. **Discussion**
   - Interpret findings in light of hypothesis
   - Compare effect size to preprocessing (original paper)
   - Discuss linguistic patterns (morphology, script type)
   - Address limitations

6. **Conclusion**
   - Summarize support for hypothesis
   - Implications for metric usage
   - Recommendations for practitioners

### Key Metrics to Report

- **Default layer optimality**: "L9 optimal for X/7 language pairs"
- **Maximum variance**: "fi-en showed largest range (0.XXXX)"
- **Significant differences**: "Y/4 comparisons significantly different from L9"
- **Best alternative**: "L6 optimal for 3 languages (cs, fi, lv)"

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Reduce batch size
python WMT17_layer_variation_experiment.py --batch_size 32
# Or use CPU (slower)
python WMT17_layer_variation_experiment.py --device cpu
```

**2. Missing Dependencies**
```bash
# Install specific packages
pip install torch transformers mosestokenizer pandas scipy matplotlib seaborn
```

**3. Data Not Found**
```bash
# Verify data directory structure
ls -R WMT17_Mover/WMT17/

# If data missing, may need to download separately
# (Check original repository for data download instructions)
```

**4. Import Errors**
```bash
# Make sure you're in repository root
pwd  # Should end with Reproducibility-issues-for-BERT-based-evaluation

# Run from root directory
python WMT17_layer_variation_experiment.py
```

**5. Analysis Script Can't Find Results**
```bash
# Check if experiment completed successfully
ls layer_variation_results/*.seg.score

# If files exist but script can't find them:
python analyze_layer_results.py --results_dir layer_variation_results
```

## Timeline Estimates

### With GPU (recommended)
- **Experiment runtime**: 30-60 minutes
- **Analysis**: 5 minutes
- **Report writing**: 4-6 hours
- **Total**: ~5-7 hours

### With CPU only
- **Experiment runtime**: 3-5 hours
- **Analysis**: 5 minutes
- **Report writing**: 4-6 hours
- **Total**: ~7-11 hours

## Additional Resources

- **Full documentation**: [LAYER_EXPERIMENT_README.md](LAYER_EXPERIMENT_README.md)
- **Original paper**: Chen et al., EMNLP 2022
- **BERTScore paper**: Zhang et al., ICLR 2020
- **Code repository**: https://github.com/Siddhu202003/Reproducibility-issues-for-BERT-based-evaluation

## Need Help?

1. Check [LAYER_EXPERIMENT_README.md](LAYER_EXPERIMENT_README.md) for detailed documentation
2. Review troubleshooting section above
3. Check GitHub Issues for similar problems
4. Contact course instructor or TA

## Next Steps After Completion

1. ✅ Run experiment
2. ✅ Analyze results
3. ✅ Generate visualizations
4. ✅ Write report following structure above
5. ✅ Include code and results in GitHub repository
6. ✅ Verify individual contributions are visible in Git history
7. ✅ Submit report before deadline

**Good luck with your final project!** 🚀
