# Robust Multimodal Learning for Neurological Disease Prediction Under Missing Modalities

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-OASIS--1-green.svg)](https://www.oasis-brains.org/)

---

## 📌 Table of Contents
- [1. Executive Overview](#1-executive-overview)
- [2. Research Insights & Methodological Rigor](#2-research-insights--methodological-rigor)
  - [The MMSE Target-Proxy Leakage](#the-mmse-target-proxy-leakage)
  - [The Age Distribution Confound](#the-age-distribution-confound)
  - [Leak-Free Cross-Validation & The Value of a Rigorous Null Result](#leak-free-cross-validation--the-value-of-a-rigorous-null-result)
- [3. System Architecture](#3-system-architecture)
  - [Vision Encoder (3D CNN)](#vision-encoder-3d-cnn)
  - [Clinical Encoder (MLP)](#clinical-encoder-mlp)
  - [Late Fusion & Missing-Modality Bernoulli Masking](#late-fusion--missing-modality-bernoulli-masking)
- [4. Repository Structure](#4-repository-structure)
- [5. Dataset & Preprocessing](#5-dataset--preprocessing)
- [6. Installation & Environment Setup](#6-installation--environment-setup)
- [7. How to Run the Project](#7-how-to-run-the-project)
  - [Step 1: Dataset Acquisition](#step-1-dataset-acquisition)
  - [Step 2: Exploratory Data Analysis & Integrity Checks](#step-2-exploratory-data-analysis--integrity-checks)
  - [Step 3: Training & Modality Ablation Experiments](#step-3-training--modality-ablation-experiments)
  - [Step 4: 5-Fold Stratified Cross-Validation & Statistical Testing](#step-4-5-fold-stratified-cross-validation--statistical-testing)
- [8. Experimental Results & Benchmarks](#8-experimental-results--benchmarks)
- [9. Pretrained Checkpoints & Inference](#9-pretrained-checkpoints--inference)
- [10. Citation & Acknowledgments](#10-citation--acknowledgments)

---

## 1. Executive Overview

This repository implements a **robust multimodal deep learning framework** for classifying cognitive impairment (Healthy vs. Impaired, defined via the Clinical Dementia Rating `CDR > 0`) using paired **3D structural brain T1 MRI volumes** and **tabular clinical/demographic markers** from the **OASIS-1** cross-sectional dataset (436 subjects).

### Core Research Questions
1. **Multimodal Synergy**: Does fusing 3D volumetric MRI with tabular demographic markers provide a statistically significant diagnostic advantage over single-modality baselines?
2. **Missing-Modality Robustness**: How can late-fusion networks be trained to remain robust and functional when an imaging scan or clinical survey is missing at inference time?
3. **Clinical Trustworthiness & Leakage Auditing**: Why do medical AI models often exhibit inflated performance (>0.90 AUC), and how do target-proxy leakage and demographic confounds compromise clinical generalization?

```
                               ┌────────────────────────────────────────────────┐
                               │                 Input Modalities               │
                               └──────────────────────┬─────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │ 3D Structural MRI Volume  │                                 │    Tabular Clinical Data  │
         │      (96 x 128 x 96)      │                                 │     (Age, Education, Sex) │
         └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                       │                                                             │
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │     3D CNN (4 Blocks)     │                                 │     2-Layer MLP (32-d)    │
         │  16 -> 32 -> 64 -> 128-d  │                                 │    Linear -> ReLU -> Linear│
         └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                       │                                                             │
                       │ 128-d Embedding                                             │ 32-d Embedding
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   Bernoulli Modality Mask   │  <-- Robust to Missing Modalities
                                       └──────────────┬──────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │    Concatenation (160-d)    │
                                       └──────────────┬──────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   Classification Head (MLP) │
                                       │ Linear(160, 64) -> Dropout  │
                                       │       -> Linear(64, 2)      │
                                       └──────────────┬──────────────┘
                                                      ▼
                                         [ Healthy vs. Impaired ]
```

---

## 2. Research Insights & Methodological Rigor

Unlike standard deep learning benchmarks that report raw test accuracy, this project was developed as a rigorous investigation into clinical AI validity:

### The MMSE Target-Proxy Leakage
- In early prototypes, a naive clinical baseline achieved **>0.90 AUC** when including the **Mini-Mental State Examination (MMSE)** score.
- **Root Cause**: MMSE is a primary cognitive assessment instrument used by clinicians when *assigning* the Clinical Dementia Rating (CDR) target label. Including MMSE creates a **diagnostic-proxy leakage loop**—the model is not diagnosing pathology from independent indicators, but reconstructing its own label.
- **Correction**: MMSE was strictly removed from all feature vectors, reducing the clinical feature set to independent demographics: `Age`, `Education Level (Educ)`, and `Sex`.

### The Age Distribution Confound
- Group analysis reveals an intrinsic confound in the OASIS-1 cross-sectional cohort design:
  - **Healthy Group**: Mean age = $43.80 \pm 23.75$ years (range: 18 – 94).
  - **Impaired Group**: Mean age = $76.76 \pm 7.12$ years (range: 62 – 96).
- The healthy cohort contains a disproportionate number of young college-aged control subjects, allowing demographic variables (principally `Age`) to achieve an AUC of ~0.83 without learning true neuropathology.

### Leak-Free Cross-Validation & The Value of a Rigorous Null Result
- **Isolated Fold Transformations**: Imputation (`SimpleImputer(strategy='median')`) and standardization (`StandardScaler()`) are fit **strictly inside each training fold** and transformed onto validation/test folds to eliminate data contamination.
- **Fixed-Epoch Blind Evaluation**: Avoided validation-set double-dipping by using a fixed training schedule (20 epochs) evaluated once, blind, on the held-out fold.
- **5-Fold Cross-Validation Result**:
  - Clinical-only Baseline: **Mean AUC = 0.8871**
  - Multimodal Late Fusion: **Mean AUC = 0.8890**
  - Paired t-test: $t = -0.4465, p = 0.6784$ (no statistically significant advantage from MRI imaging over demographics alone in this cohort size and architecture).
- **Takeaway**: Training a 3D CNN from scratch on ~350 volumes per fold without pretraining provides limited spatial diagnostic power beyond the strong demographic signal. Reporting this null result prevents overclaiming and highlights the necessity of pretrained medical backbones and age-matched cohorts.

---

## 3. System Architecture

### Vision Encoder (`MRI_CNN`)
A 4-stage 3D Convolutional Neural Network processing normalized 3D volumes of shape $(1, 96, 128, 96)$:
1. **Block 1**: `Conv3d(1, 16, kernel_size=3, padding=1)` $\to$ `ReLU()` $\to$ `MaxPool3d(2)`
2. **Block 2**: `Conv3d(16, 32, kernel_size=3, padding=1)` $\to$ `ReLU()` $\to$ `MaxPool3d(2)`
3. **Block 3**: `Conv3d(32, 64, kernel_size=3, padding=1)` $\to$ `ReLU()` $\to$ `MaxPool3d(2)`
4. **Block 4**: `Conv3d(64, 128, kernel_size=3, padding=1)` $\to$ `ReLU()` $\to$ `AdaptiveAvgPool3d(1)`
5. **Projection**: `Linear(128, 128)` output embedding.

### Clinical Encoder (`Clinical_MLP`)
A compact Multi-Layer Perceptron for 3-dimensional normalized tabular inputs (`[Age, Educ, Sex_encoded]`):
- `Linear(3, 32)` $\to$ `ReLU()` $\to$ `Linear(32, 32)` $\to$ `ReLU()`

### Late Fusion & Missing-Modality Bernoulli Masking
- **Late Fusion**: Embeddings are concatenated: $\mathbf{h} = [\mathbf{z}_{\text{MRI}} \mathbin{\Vert} \mathbf{z}_{\text{Clinical}}] \in \mathbb{R}^{160}$.
- **Classification Head**: `Linear(160, 64)` $\to$ `ReLU()` $\to$ `Dropout(p=0.3)` $\to$ `Linear(64, 2)`.
- **Missing-Modality Masking**:
  - MRI modality masking: $\tilde{\mathbf{x}}_{\text{MRI}} = \mathbf{x}_{\text{MRI}} \odot \text{Bernoulli}(p_{\text{mri}})$. If masked, the volume is zeroed out.
  - Clinical modality masking: $\tilde{\mathbf{x}}_{\text{clin}} = m \cdot \mathbf{x}_{\text{clin}} + (1 - m) \cdot \boldsymbol{\mu}_{\text{train}}$, where $m \sim \text{Bernoulli}(p_{\text{clin}})$. If masked, features are imputed using training fold feature means.

---

## 4. Repository Structure

```text
Robust-Multimodal-Learning/
├── README.md                      # Comprehensive project documentation
├── article.md                     # Scientific writeup on clinical trustworthiness & leakage
├── download.ipynb                 # Automated HuggingFace downloader for OASIS-1 dataset
├── EDA.ipynb                      # Exploratory Data Analysis & confound verification
├── MultiModalFusion.ipynb         # Complete PyTorch multimodal training & CV pipeline
├── oasis_cross-sectional.csv      # Raw OASIS-1 cross-sectional clinical metadata
├── oasis_validated.csv            # Cleaned & verified dataset mapping 436 subjects to MRI files
├── fold_comparison.png            # 5-fold CV comparison visualization
├── Exp1_MRI.pth                   # Trained weights: MRI-Only single split
├── Exp1_MRI_Only_best.pth         # Best checkpoint: MRI-Only encoder
├── Exp2_Clinical.pth              # Trained weights: Clinical-Only single split
├── Exp3_Multi.pth                 # Trained weights: Multimodal Fusion model
└── oasis_images/
    └── oasis_preprocessed/        # 436 Skull-stripped, masked T1 MRI volumes (*.npy)
```

---

## 5. Dataset & Preprocessing

### Dataset Source
- **OASIS-1**: Open Access Series of Imaging Studies (Cross-Sectional Dataset in Young, Middle Aged, Nondemented, and Demented Older Adults).
- Preprocessed 3D T1 masked volumes hosted at Hugging Face: [`pzarzycki/mri-oasis-1-ixi-pre`](https://huggingface.co/datasets/pzarzycki/mri-oasis-1-ixi-pre).

### Preprocessing Protocol
1. **Target Formulation**: Binary target derived from CDR:
   $$\text{target} = \begin{cases} 0 & (\text{CDR} = 0, \text{ Healthy}) \\ 1 & (\text{CDR} > 0, \text{ Impaired}) \end{cases}$$
2. **MRI Resampling**: Volumes are resliced to uniform target dimensions $(96, 128, 96)$ using 1st-order spline interpolation (`scipy.ndimage.zoom`).
3. **MRI Intensity Normalization**: Min-Max scaled per volume to $[0.0, 1.0]$.
4. **Clinical Imputation & Scaling**: Missing values in `Age` and `Educ` are imputed with training median; all continuous features are standard-scaled ($\mu=0, \sigma=1$).
5. **Class Imbalance Handling**: Inverse class weighting in PyTorch `CrossEntropyLoss`:
   $$w_c = \frac{N}{2 \cdot N_c}$$

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with $\ge 8\text{ GB}$ VRAM recommended (CUDA 11.8 or 12.x)

### 1. Clone the Repository
```bash
git clone https://github.com/Uman-66/Robust-Multimodal-Learning-for-Neurological-Disease-Prediction-Under-Missing-Modalities.git
cd Robust-Multimodal-Learning-for-Neurological-Disease-Prediction-Under-Missing-Modalities
```

### 2. Create and Activate a Virtual Environment
```bash
# Using conda
conda create -n robust-mri python=3.10 -y
conda activate robust-mri

# Or using venv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scipy scikit-learn matplotlib seaborn huggingface_hub tqdm ipykernel
```

---

## 7. How to Run the Project

### Step 1: Dataset Acquisition
Open and execute `download.ipynb` (or run via Python) to automatically download all 436 preprocessed `.npy` MRI scans from Hugging Face into `oasis_images/oasis_preprocessed/`:

```python
import os
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download

os.makedirs("oasis_images/oasis_preprocessed", exist_ok=True)

# 1. Download clinical CSV
csv_cache = hf_hub_download(
    repo_id="pzarzycki/mri-oasis-1-ixi-pre",
    repo_type="dataset",
    filename="oasis_preprocessed/oasis_cross-sectional.csv"
)
shutil.copyfile(csv_cache, "oasis_cross-sectional.csv")

# 2. Download all 436 3D MRI scans
df = pd.read_csv("oasis_cross-sectional.csv")
for sub in df['ID']:
    hf_hub_download(
        repo_id="pzarzycki/mri-oasis-1-ixi-pre",
        repo_type="dataset",
        filename=f"oasis_preprocessed/{sub}_t1_masked.npy",
        local_dir="./oasis_images",
        local_dir_use_symlinks=False
    )
```

### Step 2: Exploratory Data Analysis & Integrity Checks
Open and run `EDA.ipynb`. This notebook:
- Confirms all 436 MRI files exist on disk and outputs `oasis_validated.csv`.
- Generates demographic breakdown plots (Healthy vs. Impaired).
- Produces the feature correlation matrix illustrating the `Age` confound.

### Step 3: Training & Modality Ablation Experiments
Open `MultiModalFusion.ipynb` and run the single-split controlled ablation experiments:

```python
# Experiment 1: MRI-Only (Vision Branch Only)
run_experiment("Exp1_MRI", mri_present=1.0, clinical_present=0.0)

# Experiment 2: Clinical-Only (Demographics Only)
run_experiment("Exp2_Clinical", mri_present=0.0, clinical_present=1.0)

# Experiment 3: Full Multimodal Late Fusion
run_experiment("Exp3_Multi", mri_present=1.0, clinical_present=1.0)
```

### Step 4: 5-Fold Stratified Cross-Validation & Statistical Testing
Execute the final cell in `MultiModalFusion.ipynb` to run the 5-fold cross-validation pipeline:
- Strictly fits `SimpleImputer` and `StandardScaler` on training folds.
- Trains `model_c` (Clinical) and `model_m` (Multimodal) for 20 fixed epochs per fold.
- Computes out-of-fold validation AUC and performs a two-tailed **paired t-test** (`scipy.stats.ttest_rel`).

---

## 8. Experimental Results & Benchmarks

### Single Held-Out Test Set (20% Split, $N=44$)

| Experiment | Modality Inputs | Features / Architecture | Test Accuracy | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Exp 1: MRI-Only** | 3D T1 MRI Scans | 4-Block 3D CNN (128-d) | **77.27%** | 0.6737 | 0.8055 |
| **Exp 2: Clinical-Only** | Tabular Demographics | 2-Layer MLP (Age, Educ, Sex) | 73.86% | **0.7581** | **0.8257** |
| **Exp 3: Multimodal** | MRI + Clinical | Late Fusion (160-d) Head | 73.86% | 0.7569 | 0.8228 |

### 5-Fold Stratified Cross-Validation (All 436 Subjects)

| Fold | Clinical-Only AUC | Multimodal Fusion AUC | Winning Modality |
| :---: | :---: | :---: | :---: |
| **Fold 1** | **0.8904** | 0.8816 | Clinical (+0.0088) |
| **Fold 2** | 0.8761 | **0.8903** | Multimodal (+0.0142) |
| **Fold 3** | 0.8519 | **0.8545** | Multimodal (+0.0026) |
| **Fold 4** | 0.8937 | **0.9011** | Multimodal (+0.0074) |
| **Fold 5** | **0.9235** | 0.9175 | Clinical (+0.0060) |
| **Mean $\pm$ Std** | **0.8871 $\pm$ 0.026** | **0.8890 $\pm$ 0.023** | **Multimodal (+0.0019)** |

$$\text{Paired } t\text{-Test}: \quad t = -0.4465, \quad p = 0.6784 \quad (\text{Not Statistically Significant})$$

---

## 9. Pretrained Checkpoints & Inference

The repository includes trained weights saved during ablation experiments:
- `Exp1_MRI.pth` / `Exp1_MRI_Only_best.pth`: Vision encoder weights.
- `Exp2_Clinical.pth`: Clinical MLP weights.
- `Exp3_Multi.pth`: Full multimodal fusion weights.

### Sample Inference Script

```python
import torch
import numpy as np

# Define classes as in MultiModalFusion.ipynb
# (MRI_CNN, Clinical_MLP, MultimodalFusion)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MultimodalFusion(mri_dim=128, clin_dim=32, num_classes=2).to(device)
model.load_state_dict(torch.load("Exp3_Multi.pth", map_location=device))
model.eval()

# Dummy input example: 1 MRI volume (1, 1, 96, 128, 96) and 1 Clinical vector (1, 3)
mri_sample = torch.randn(1, 1, 96, 128, 96).to(device)
clin_sample = torch.tensor([[0.45, -0.20, 1.0]], dtype=torch.float32).to(device)  # [Age, Educ, Sex]

with torch.no_grad():
    # 1. Full Multimodal Prediction
    logits_full = model(mri_sample, clin_sample)
    prob_impaired_full = torch.softmax(logits_full, dim=1)[0, 1].item()

    # 2. Missing MRI Scenario (Zero-masking)
    logits_missing_mri = model(torch.zeros_like(mri_sample), clin_sample)
    prob_missing_mri = torch.softmax(logits_missing_mri, dim=1)[0, 1].item()

    # 3. Missing Clinical Scenario (Mean imputation)
    clin_mean = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32).to(device)
    logits_missing_clin = model(mri_sample, clin_mean)
    prob_missing_clin = torch.softmax(logits_missing_clin, dim=1)[0, 1].item()

print(f"P(Impaired | Full Multimodal) : {prob_impaired_full:.4f}")
print(f"P(Impaired | Missing MRI)      : {prob_missing_mri:.4f}")
print(f"P(Impaired | Missing Clinical) : {prob_missing_clin:.4f}")
```

---

## 10. Citation & Acknowledgments

### OASIS Dataset Citation
> Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). *Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults.* Journal of Cognitive Neuroscience, 19(9), 1498-1507.

### Preprocessed Hugging Face Dataset
> Zarzycki, P. *MRI OASIS-1 & IXI Preprocessed Dataset*. Hugging Face Hub: `pzarzycki/mri-oasis-1-ixi-pre`.

---

