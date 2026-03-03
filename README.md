# AI Music Detection Under Generator Shift

This repository presents a fully reproducible demonstration of an AI music detection pipeline designed to evaluate generalization under generator distribution shift.

The project mirrors a larger capstone conducted with Sound Ethics, where the core finding was that detection models often fail when exposed to audio from unseen generators.

This reproducible version implements:
- Baseline CNN training on SONICS fake + FMA real
- Evaluation on unseen generators
- Generator-level holdout fine-tuning experiment
- Threshold policy analysis for deployment tradeoffs

# Repository Structure
```
data/
  raw/                  # local audio files (gitignored)
  interim/              # processed spectrograms + splits
  external/             # external generator processed data + splits

results/
  baseline_cnn.pt
  baseline_cnn_finetuned.pt
  figures/
  *.json
  *.csv

src/
  sample_dataset.py
  make_manifest.py
  preprocess.py
  preprocess_external.py
  train_baseline.py
  evaluate.py
  evaluate_external.py
  make_external_splits.py
  finetune_external.py

notebooks/
  generator_shift_analysis.ipynb
```

# Set up

Install dependencies:
```
pip install -r requirements.txt
```

Python 3.10+ recommended

# Reproducible Quickstart

### 1. Train baseline model
```
python -m src.train_baseline
```

### 2. Evaluate baseline (in-distribution)
```
python -m src.evaluate --model_path results/baseline_cnn.pt
```

### 3. Preprocess external generators
```
python -m src.preprocess_external
```

### 4. Evaluate baseline on unseen generators
```
python -m src.evaluate_external --model_path results/baseline_cnn.pt
```

### 5. Create generator holdout split
```
python -m src.make_external_splits
```

### 6. Fine tune model
```
python -m src.finetune_external
```

### 7. Evaluate fine tuned model on held out generators
```
python -m src.evaluate_external \
  --external_manifest data/external/splits/external_holdout.csv \
  --model_path results/baseline_cnn_finetuned.pt
```

# Key Findings

- Baseline performs strongly in-distribution (~93% accuracy)
- Baseline fails under generator shift (~0% recall on most unseen generators)
- Fine tuning on a small set of generators dramatically improves robustness
- Fine tuning introduces measurable tradeoffs in original validation performance
- Threshold selection meaningfully affects deployment risk
- Strong validation accuracy does not guarantee robustness under distribution shift

# Notebook Walkthrough

For a structured explanation of the experiment design, results, and product implications:
```
notebooks/generator_shift_analysis.ipynb
```

# Notes on Original Project
The original project used the SpecTTTra model from the SONICS paper and was trained on a GPU enabled VM with a substantially larger dataset.

This repository provides a lightweight but structurally equivalent experimental framework designed for clarity and reproducibility.