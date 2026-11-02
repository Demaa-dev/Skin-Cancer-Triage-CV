# Skin Cancer Triage: Automated Dermatological Screening with Hybrid CV + Deep Learning

> **Can a computer help a doctor decide which mole needs urgent attention and which is safe to monitor?** This project builds a **binary skin cancer classifier** that distinguishes **malignant melanoma from benign nevus** on 7,818 dermoscopic images, using a **hybrid approach** — classical computer vision for feature intuition, and a **fine-tuned MobileNetV2 transfer learning pipeline** for the final decision.

---

## Overview

Skin cancer is one of the most common cancers worldwide. Early detection of **melanoma** dramatically improves survival rates — but screening is expensive, slow, and unevenly distributed.

This project builds a full pipeline that:

- Applies **classical CV techniques** (histogram equalization, Gaussian, Laplacian) to understand what the model "sees"
- Trains a **MobileNetV2 transfer learning model** on the HAM10000 dataset
- Detects the **silent killer of medical ML** — class imbalance bias — where the model learns to say "benign" because it's *usually right*
- Fixes it with **class weighting** and **fine-tuning**, then proves the fix with **sensitivity**, the metric that actually matters in medicine

---

## The Problem This Project Solves

The dataset is **86% benign, 14% melanoma**. A naive model that always predicts "benign" achieves **86% accuracy** — and **catches zero cancers**.

That's exactly what happened in the first training run:

| Model Version | Accuracy | Sensitivity (Recall for Melanoma) |
|---------------|:--------:|:---------------------------------:|
| Baseline MobileNetV2 | **88.68%** | **35.83%** ❌ |
| After class weighting + fine-tuning | **89.19%** | **54.17%** ✅ |

> **This is the core lesson of the project:** accuracy lied. Sensitivity told the truth. A model that misses **2 out of 3 melanomas** is not a medical tool — it's a hazard.

The fix used `sklearn`'s balanced class weights (melanoma weighted **~6× more** than benign) plus fine-tuning the last 20 layers of MobileNetV2 at a reduced learning rate (`1e-5`).

---

## Methodology

### 1. Dataset Preparation
- **Dataset:** HAM10000 (`kmader/skin-cancer-mnist-ham10000`)
- **Filtered** to two classes: `mel` (malignant) → `1`, `nv` (benign) → `0`
- **Final size:** 7,818 images (6,705 benign / 1,113 melanoma)
- **Split strategy:** `GroupShuffleSplit` on `lesion_id` — **the same mole never appears in both train and test**. This prevents data leakage across augmentations of the same lesion.

| Split | Images |
|-------|-------:|
| Train | 6,257 |
| Val   | 775   |
| Test  | 786   |

### 2. Classical CV Analysis (Syllabus Concepts)
Applied to a sample lesion to make the model's input transformations **visible**:
- **Histogram Equalization** — point operator for contrast
- **Gaussian Filtering** — low-pass, suppresses texture noise
- **Laplacian Filtering** — high-pass, highlights the mole boundary

### 3. Deep Learning Pipeline
- **Base model:** MobileNetV2 (ImageNet weights, frozen)
- **Custom head:** `GlobalAveragePooling → Dropout(0.4) → Dense(64, relu) → Dense(1, sigmoid)`
- **Augmentation:** rotation (±20°), zoom (±20%), horizontal flip
- **Callbacks:** `EarlyStopping` + `ReduceLROnPlateau`
- **Fine-tuning phase:** last 20 layers unfrozen, `Adam` at `lr=1e-5`

### 4. The Fix (the important part)
- `compute_class_weight('balanced')` → benign `0.58`, melanoma `3.52`
- Retrained with `class_weight=class_weights` → **sensitivity nearly doubled** with a small accuracy boost

---

## Results

| Metric | Baseline | Improved |
|--------|---------:|---------:|
| Test Accuracy | 88.68% | **89.19%** |
| Melanoma Sensitivity | 35.83% | **54.17%** |

A confusion matrix comparison (blue = baseline, green = improved) is included in the notebook, showing exactly which predictions shifted after class weighting.

---

## What Makes This Project Different

- **Grouped splitting by `lesion_id`** — prevents a classic HAM10000 leak where the same lesion is in train *and* test
- **Metric honesty** — the project reports **sensitivity** as the primary KPI, not accuracy
- **Class imbalance is treated as the central problem**, not an afterthought
- **CV theory made visible** — histograms and filters are plotted against the raw lesion
- **Reproducible export** — model saved as `skin_cancer_model.keras` for Streamlit deployment

---

## How to Run

### Notebook
```bash
pip install kaggle tensorflow opencv-python scikit-learn seaborn matplotlib
# place kaggle.json in ~/.kaggle/ first
jupyter notebook SkinCancer.ipynb
```
The notebook downloads the dataset, trains the model, and saves `skin_cancer_model.keras`.

### Streamlit App
```bash
streamlit run app.py
```
Upload a dermoscopic image and get a **benign/malignant prediction** with a confidence score.

---

## Limitations & Honest Caveats

- **54% sensitivity is not clinically deployable.** A model that misses ~46% of melanomas is a **screening assistant at best**, not a diagnostic tool.
- **Dermoscopic images only** — not smartphone photos.
- **Two-class problem** — real dermatology involves many lesion types (BCC, AKIEC, BKL, DF, VASC).
- **Further work:** threshold tuning, heavier augmentation, Focal Loss, cross-validation, and eventually testing on an external dataset.

---

## Files

```
├── SkinCancer.ipynb             # Full pipeline
├── skin_cancer_model.keras      # Trained MobileNetV2 model
├── app.py                       # Streamlit demo
├── HAM10000_metadata.csv        # Dataset metadata
└── README.md
```


---
NOT MY DATASET
