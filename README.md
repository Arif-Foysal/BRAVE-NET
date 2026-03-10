# BRAVE-Net

**Burg Residual Augmented Vision Transformer for Parkinson's Disease Detection from Dysarthric Speech**

| | |
|---|---|
| **Student** | MD Arif Faysal Nayem |
| **Supervisor** | Dr. Md. Ohidujjaman |
| **Target** | IEEE Access (Q1) · Biomedical Signal Processing and Control (Elsevier, Q1) |

---

## Overview

BRAVE-Net introduces a **"Signal-First"** paradigm for PD detection. Standard deep learning pipelines treat Parkinsonian vocal irregularities as noise and remove them — discarding the very biomarkers that carry diagnostic value.

BRAVE-Net instead applies the **Burg-Based Linear Prediction with Residual Estimation method** (Ohidujjaman et al., 2025) to *mathematically characterise and preserve* these irregularities before classification.

```
Raw Audio
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Burg Residual Restoration    │
│  • Compute LPC via Burg's method        │
│  • Extract residual e[n] = A(z)·s[n]   │
│  • Amplify 100–300 Hz (glottal band)   │
│  • Amplify 4–8 Hz (tremor modulation)  │
│  • Reconstruct via LP synthesis 1/A(z) │
└────────────────┬────────────────────────┘
                 │  Restored Signal
                 ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Mel-Spectrogram              │
│  128 mel bands · hop=256 · win=1024     │
│  → 224×224 RGB image                   │
└────────────────┬────────────────────────┘
                 │  Spectrogram Image
                 ▼
┌─────────────────────────────────────────┐
│  Stage 3 — ViT-B/16 (ImageNet-21k)     │
│  • All attention blocks frozen          │
│  • Last 2 encoder blocks unfrozen      │
│  • Custom head: Linear(768→256→2)      │
└────────────────┬────────────────────────┘
                 │
                 ▼
           PD / HC prediction
```

### Research Hypothesis

The **residual error term** in Linear Predictive Coding — discarded by standard models as unpredictable noise — encodes the primary biomarkers of PD-related vocal dysfunction (glottal tremors, hypophonia). Amplifying this residue before classification should yield a measurable improvement over raw-audio baselines.

---

## Ablation Study Design

| Configuration | Signal Input | Classifier | Role |
|:---|:---|:---|:---|
| **Baseline 1** | Raw Audio | MFCC + Random Forest | Replicates Hassan et al., 2019 |
| **Baseline 2** | Raw Audio | Mel-Spec + ResNet-18 | Standard deep learning |
| **Baseline 3** | Raw Audio | Mel-Spec + ViT-B/16 | Isolates transformer contribution |
| **BRAVE-Net** | **Burg-Restored** | **Mel-Spec + ViT-B/16** | **Full framework** |

The critical comparison is Baseline 3 vs BRAVE-Net — any F1 gain isolates the Burg restoration contribution.

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd BRAVE-NET

# Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

---

## Datasets

### TORGO Database (Primary)
- 7 dysarthric speakers + 7 healthy controls
- University of Toronto: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
- Place files at: `data/raw/TORGO/`

Expected structure:
```
data/raw/TORGO/
├── F01/           ← Dysarthric female
│   ├── Session1/
│   │   ├── wav_headMic/
│   │   └── wav_arrayMic/
├── FC1/           ← Female control
└── M01/ ...
```

### MDVR-KCL (Cross-Dataset Validation)
- King's College London PD voice dataset
- Place files at: `data/raw/MDVR-KCL/PD/` and `data/raw/MDVR-KCL/HC/`

---

## Quick Start

### Step 1 — Preprocess datasets

```bash
# Generate Burg-restored Mel-Spectrograms (for BRAVE-Net)
python scripts/prepare_torgo.py \
    --dataset torgo \
    --raw_dir data/raw/TORGO \
    --out_dir data/processed/TORGO \
    --apply_restoration \
    --n_jobs 8

# Generate raw Mel-Spectrograms (for ablation baselines 2 & 3)
python scripts/prepare_torgo.py \
    --dataset torgo \
    --raw_dir data/raw/TORGO \
    --out_dir data/processed/TORGO \
    --n_jobs 8
```

### Step 2 — Train models

```bash
# BRAVE-Net (proposed)
python scripts/train.py --model brave_net \
    --processed_dir data/processed/TORGO/brave_net \
    --run_name brave_net

# Baseline 1: Random Forest
python scripts/train.py --model rf \
    --mfcc_path data/processed/TORGO/mfcc/features.npz \
    --run_name baseline_rf

# Baseline 2: ResNet-18
python scripts/train.py --model resnet18 \
    --processed_dir data/processed/TORGO/raw_mel \
    --run_name baseline_resnet18

# Baseline 3: ViT-B/16 (raw)
python scripts/train.py --model vit_raw \
    --processed_dir data/processed/TORGO/raw_mel \
    --run_name baseline_vit_raw
```

### Step 3 — Evaluate and generate figures

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/brave_net_loso_M01_best.pt \
    --manifest   data/processed/TORGO/brave_net/test_M01_manifest.json \
    --model      brave_net \
    --output_dir results/brave_net_eval \
    --gradcam \
    --n_gradcam  20
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test suites
pytest tests/test_burg_lp.py -v        # Burg algorithm
pytest tests/test_model.py -v          # Model architecture
pytest tests/test_integration.py -v   # End-to-end pipeline

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Project Structure

```
BRAVE-NET/
├── configs/
│   └── brave_net_config.yaml      ← All hyperparameters
├── data/
│   ├── raw/                       ← Download datasets here
│   ├── processed/                 ← Generated by prepare_torgo.py
│   └── splits/
├── src/
│   ├── features/
│   │   ├── burg_lp.py             ← Burg's method (core algorithm)
│   │   ├── residual_error.py      ← Residual estimation & restoration
│   │   └── feature_pipeline.py   ← Full feature extraction pipeline
│   ├── models/
│   │   ├── brave_net.py           ← BRAVE-Net (ViT-B/16 + restoration)
│   │   └── baselines.py          ← RF, ResNet-18, ViT-raw baselines
│   ├── training/
│   │   ├── trainer.py             ← LOSO-CV training loop
│   │   ├── losses.py              ← Weighted CE loss
│   │   └── metrics.py             ← Sensitivity, Specificity, F1, AUC, MCC
│   ├── evaluation/
│   │   ├── evaluate.py            ← Inference + CI computation
│   │   └── visualize.py          ← Grad-CAM, confusion matrix, ROC
│   └── utils/
│       ├── audio_utils.py         ← Audio I/O and preprocessing
│       ├── dataset.py             ← TORGO / MDVR-KCL dataset classes
│       └── config.py              ← Config loading
├── scripts/
│   ├── prepare_torgo.py           ← Dataset preprocessing
│   ├── train.py                   ← Training entry point
│   └── evaluate.py                ← Evaluation entry point
└── tests/
    ├── test_burg_lp.py
    ├── test_model.py
    └── test_integration.py
```

---

## Evaluation Protocol

All results are reported under **Leave-One-Speaker-Out Cross-Validation (LOSO-CV)** — speaker identity is strictly held out to prevent data leakage.

Metrics reported (with 95% bootstrap CI):
- **Sensitivity** (Recall for PD class) — primary clinical metric
- **Specificity** (Recall for HC class) — primary clinical metric
- **F1-Score** — primary comparative metric for ablation study
- **AUC-ROC**
- **Matthew's Correlation Coefficient (MCC)** — robust to class imbalance
- **McNemar's Test** — statistical significance of BRAVE-Net vs Baseline 3

---

## Scientific Contributions

1. **Cross-Domain Algorithm Repurposing.** First application of a telecommunications-derived Burg Residual Estimation algorithm (Ohidujjaman et al., 2025) to preserve pathological vocal biomarkers for PD diagnosis.

2. **Interpretable Clinical AI.** Grad-CAM validation confirms the model attends to glottal tremor frequency bands (100–300 Hz) and tremor modulation (4–8 Hz), not to speaker identity artefacts.

3. **Rigorous Small-Data Benchmarking.** Four-condition ablation with LOSO-CV and statistical significance testing establishes a reproducible benchmark for future dysarthric PD detection research.

---

## Citation

```bibtex
@article{nayem2026bravenet,
  title   = {{BRAVE-Net}: Burg Residual Augmented Vision Transformer
             for Interpretable Parkinson's Disease Detection from Dysarthric Speech},
  author  = {Nayem, MD Arif Faysal and Ohidujjaman, Md.},
  journal = {IEEE Access},
  year    = {2026},
}
```

---

## License

MIT License © 2026 MD Arif Faysal Nayem
