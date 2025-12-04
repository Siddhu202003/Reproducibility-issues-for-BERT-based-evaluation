# BERTScore Layer Variation Experiment

## **Hypothesis**

**The optimal BERT layer for computing embeddings in BERTScore varies significantly across different languages, revealing an additional and previously under-explored source of reproducibility issues in BERT-based evaluation metrics.**

While the original paper focused on preprocessing effects (stopwords, IDF weighting) for MoverScore, we hypothesize that BERTScore—despite its apparent simplicity—has its own critical sensitivity: **the choice of which BERT layer to use for extracting embeddings**.

## Background

### Gap in Literature

The paper "Reproducibility Issues for BERT-based Evaluation Metrics" (Chen et al., 2022) demonstrated that:
- Preprocessing choices significantly impact MoverScore performance
- Different BERT-based metrics have varying sensitivities to implementation details
- Reproducibility issues extend beyond just BLEU to modern neural metrics

However, the paper did not extensively investigate:
- **Layer-specific sensitivity in BERTScore across languages**
- Whether the default layer choice (Layer 9 for BERT-base-multilingual) is optimal for all language pairs
- How much performance variation exists when using different layers

### Our Contribution

We address this gap by:
1. Systematically testing BERTScore with different BERT layers (L1, L6, L9, L12, Average)
2. Measuring performance variation across 7 language pairs in WMT17
3. Demonstrating that **optimal layer selection is language-dependent**
4. Quantifying the magnitude of this effect compared to other reproducibility factors

## Methodology

### Experimental Design

**Fixed Variables:**
- Metric: BERTScore (with IDF weighting)
- Dataset: WMT17 (7 language pairs: cs-en, de-en, fi-en, lv-en, ru-en, tr-en, zh-en)
- Model: bert-base-multilingual-cased
- Aggregation: Greedy matching with cosine similarity

**Variable:**
- BERT layer used for embedding extraction: {1, 6, 9, 12, Average}

**Metrics:**
- Segment-level F1 scores
- System-level mean scores
- Correlation with human judgments (where available)
- Performance variance across layers

### Layer Configurations

1. **L1**: First layer (surface-level, token-based features)
2. **L6**: Mid-layer (transitional representations)
3. **L9**: Default layer (paper's recommended layer for bert-base-multilingual-cased)
4. **L12**: Final layer (task-specific, contextualized)
5. **Avg**: Average embeddings across all layers

### Rationale

- Different layers capture different linguistic phenomena
- Lower layers: syntactic features, POS tags
- Middle layers: semantic relationships
- Upper layers: task-specific features, discourse
- Language complexity may affect which layer is optimal

## Installation & Setup

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt

# Ensure CUDA is available (recommended for faster processing)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Data Verification

```bash
# Verify WMT17 data is present
ls WMT17_Mover/WMT17/

# Expected files:
# - newstest2017-csen-ref.en
# - newstest2017-deen-ref.en
# - newstest2017-fien-ref.en
# - newstest2017-lven-ref.en
# - newstest2017-ruen-ref.en
# - newstest2017-tren-ref.en
# - newstest2017-zhen-ref.en
# - (plus system translations and metadata)
```

## Running the Experiment

### Basic Usage

```bash
# Run with default settings (CUDA, batch_size=64)
python WMT17_layer_variation_experiment.py
```

### Advanced Options

```bash
# Use CPU (if CUDA unavailable)
python WMT17_layer_variation_experiment.py --device cpu

# Adjust batch size for memory constraints
python WMT17_layer_variation_experiment.py --batch_size 32

# Use different BERT model
python WMT17_layer_variation_experiment.py --model roberta-large

# Specify custom data directory
python WMT17_layer_variation_experiment.py --data_dir /path/to/WMT17

# Specify output directory
python WMT17_layer_variation_experiment.py --output_dir my_results
```

### Expected Runtime

- **With GPU**: ~30-60 minutes (depending on GPU)
- **With CPU**: ~3-5 hours
- **Memory**: 4-8 GB GPU RAM (batch_size=64), 16+ GB system RAM

## Output Files

The experiment generates several output files in the `layer_variation_results/` directory:

### 1. Individual Layer Results

```
bertscore_L1_YYYYMMDD_HHMMSS.seg.score
bertscore_L6_YYYYMMDD_HHMMSS.seg.score
bertscore_L9_YYYYMMDD_HHMMSS.seg.score
bertscore_L12_YYYYMMDD_HHMMSS.seg.score
bertscore_Avg_YYYYMMDD_HHMMSS.seg.score
```

Format: Tab-separated values with columns:
- `metric`: Metric name (e.g., 'bertscore_L9')
- `lp`: Language pair (e.g., 'cs-en')
- `testset`: Test set name
- `system`: MT system name
- `sid`: Segment ID
- `score`: BERTScore F1 for this segment

### 2. Correlation Analysis

```
layer_correlation_analysis_YYYYMMDD_HHMMSS.csv
```

Contains statistical analysis for each layer/language pair:
- Number of systems evaluated
- Mean scores and standard deviations
- Min/max scores
- (Future: Correlation with human judgments)

### 3. Summary Report

```
summary_report_YYYYMMDD_HHMMSS.txt
```

Human-readable summary including:
- Best performing layer for each language pair
- Performance differences across layers
- Key findings and observations

## Analysis & Results Interpretation

### Expected Patterns

Based on linguistic theory and prior BERT research:

1. **Morphologically Rich Languages** (fi-en, cs-en, lv-en, tr-en):
   - May benefit from middle layers (L6-L9)
   - Higher variance across layers
   - Surface features less informative

2. **Logographic Languages** (zh-en):
   - Potentially different optimal layer
   - Character-level vs. word-level representations matter

3. **Germanic Languages** (de-en):
   - May show more stable performance across layers
   - Compound word handling affects results

### Key Metrics to Examine

1. **Layer Ranking by Language**:
   - Does L9 (default) perform best for all languages?
   - Which languages deviate most from default?

2. **Performance Variance**:
   - Range between best and worst layer for each language
   - Standard deviation across layers

3. **Layer Averaging vs. Single Layer**:
   - Does averaging improve consistency?
   - Trade-off: stability vs. peak performance

## Reproducing Original Paper Results

To compare with the original paper's results:

```bash
# Run original reproduce_17.py with BERTScore
cd WMT17_Mover
python reproduce_17.py --metric bertscore --model bert-base-multilingual-cased
```

This will use the default layer (L9) as specified in the original implementation.

## Extending the Experiment

### Additional Analyses

1. **Statistical Significance Testing**:
```python
# Add to analysis script
from scipy.stats import wilcoxon
# Compare L9 vs. best layer for each language
```

2. **Correlation with Human Judgments**:
```python
# If human scores available
from scipy.stats import pearsonr, spearmanr
# Compute correlation for each layer
```

3. **Error Analysis**:
```python
# Identify segments where layer choice matters most
# Analyze linguistic characteristics of these segments
```

### Different Models

Test hypothesis with other BERT variants:
```bash
python WMT17_layer_variation_experiment.py --model roberta-large
python WMT17_layer_variation_experiment.py --model xlm-roberta-base
python WMT17_layer_variation_experiment.py --model bert-base-uncased  # English only
```

## Final Report Structure

Your final project report should include:

### 1. Title & Abstract
- Clear statement of hypothesis
- Brief methodology description
- Key findings summary

### 2. Introduction
- Problem statement: Reproducibility in NLP evaluation
- Gap: Layer selection sensitivity understudied
- Research question: Does optimal layer vary by language?

### 3. Background
- Original paper's findings on preprocessing
- BERT layer representations (prior work)
- BERTScore metric overview

### 4. Methodology
- **Hypothesis** (BOLD): State clearly with measurable goals
- Experimental design (fixed vs. variable)
- Dataset description (WMT17)
- Layer configurations tested
- Evaluation metrics

### 5. Results
- Performance tables by layer and language
- Visualization: Heatmap of layer × language performance
- Statistical analysis of variance
- Identification of optimal layers per language

### 6. Discussion
- Do results support hypothesis?
- Which languages show strongest layer effects?
- Comparison to preprocessing effects (original paper)
- Implications for metric reproducibility
- Limitations of current study

### 7. Conclusion
- Summary of findings
- Contribution to reproducibility research
- Recommendations for BERTScore users
- Future work directions

### 8. References
- Original paper (Chen et al., 2022)
- BERTScore paper (Zhang et al., 2020)
- BERT paper (Devlin et al., 2019)
- WMT17 shared task papers

## Citation

If you use this code, please cite both the original paper and BERTScore:

```bibtex
@inproceedings{chen-etal-2022-reproducibility,
    title = "Reproducibility Issues for {BERT}-based Evaluation Metrics",
    author = "Chen, Yanran and Belouadi, Jonas and Eger, Steffen",
    booktitle = "Proceedings of EMNLP 2022",
    year = "2022",
}

@inproceedings{zhang2020bertscore,
    title = "{BERT}Score: Evaluating Text Generation with {BERT}",
    author = "Zhang, Tianyi and Kishore, Varsha and Wu, Felix and Weinberger, Kilian Q. and Artzi, Yoav",
    booktitle = "Proceedings of ICLR 2020",
    year = "2020",
}
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python WMT17_layer_variation_experiment.py --batch_size 16
```

**2. Missing WMT17 Data**
```bash
# Verify data path
ls WMT17_Mover/WMT17/
# Update path if needed
python WMT17_layer_variation_experiment.py --data_dir /correct/path
```

**3. Tokenization Errors**
```bash
# Ensure mosestokenizer is installed
pip install mosestokenizer
```

**4. Import Errors**
```bash
# Run from repository root
cd /path/to/Reproducibility-issues-for-BERT-based-evaluation
python WMT17_layer_variation_experiment.py
```

## Contact

For questions about this experiment:
- Check original paper: Chen et al., EMNLP 2022
- Refer to BERTScore documentation: https://github.com/Tiiiger/bert_score

## License

This code extends the original reproducibility study. Please respect the original repository's license terms.
