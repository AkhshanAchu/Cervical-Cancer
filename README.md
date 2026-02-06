# 🔬 Cervical Cancer Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A deep learning-based dual-stage pipeline for automated cervical cancer detection using medical imaging. This system combines dual head **image segmentation** and feauture attention **classification** models to analyze cervical cell images.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Installation](#-installation)
- [Training Pipeline](#-training-pipeline)
- [Evaluation](#-evaluation)
- [Requirements](#-requirements)

---

## Overview

This project implements a **two-stage deep learning pipeline** for cervical cancer detection:

1. **Stage 1: Segmentation** - Identifies and segments cytoplasm and nucleus regions in cervical cell images
2. **Stage 2: Classification** - Classifies cells into different cancer stages based on segmented features

### Key Features
**Dual-Head Attention U-Net** for precise cell segmentation  
**ConvNeXt-based Classifier** with attention mechanisms  
**Multi-class Classification** (5 cancer stages)  
**Channel Attention (CBAM)** for improved feature extraction  
**Attention Gates** for better skip connections  

---

## Architecture

### Pipeline Overview

```
Input Image (Cervical Cell)
         ↓
    ┌────────────────────┐
    │  SEGMENTATION      │
    │  (Stage 1)         │
    └────────────────────┘
         ↓
    Segmented Masks:
    - Cytoplasm Mask
    - Nucleus Mask
         ↓
    ┌────────────────────┐
    │  CLASSIFICATION    │
    │  (Stage 2)         │
    └────────────────────┘
         ↓
    Cancer Stage Prediction
    (Class 0-4)
```

---

## 📁 Project Structure

```
Cervical-Cancer/
│
├── 📂 model/                      # Model architecture definitions
│   ├── segmentation_model.py     # Dual-Head Attention U-Net
│   └── classification_model.py   # ConvNeXt Attention Classifier
│
├── 📂 tools/                      # Utility functions and tools
│   ├── data_loader.py            # Data loading and augmentation
│   └── metrics.py                # Evaluation metrics (IoU, Dice, etc.)
│
├── 📂 utils/                      # Helper functions
│   ├── visualization.py          # Visualization utilities
│   └── preprocessing.py          # Image preprocessing
│
├── 📂 __pycache__/               # Python cache files
│
├── 📄 main_segment.py            # Segmentation training entry point
├── 📄 train_segment.py           # Segmentation training logic
├── 📄 main_classify.py           # Classification training entry point
├── 📄 train_classify.py          # Classification training logic
├── 📄 evaluate.py                # Model evaluation script
├── 📄 .gitattributes             # Git attributes
└── 📄 README.md                  # Project documentation
```

---

## 🤖 Models

### 1. Segmentation Model: Dual-Head Attention U-Net

**Architecture:** `DualHeadAttConvNeXtUNet`

#### Key Components:

- **Encoder**: ConvNeXt-Tiny (pretrained on ImageNet)
  - Extracts hierarchical features at multiple scales
  - Feature maps: [96, 192, 384, 768] channels

- **Decoder**: Attention U-Net with CBAM
  - **Attention Gates**: Focus on relevant regions during upsampling
  - **CBAM Modules**: Channel + Spatial attention for refined features
  - Symmetric decoder path: [256, 128, 64] channels

- **Dual Heads**: Separate outputs for:
  - **Cytoplasm Segmentation** (cyt)
  - **Nucleus Segmentation** (nuc)

#### Architecture Details:

```python
Input: [B, 3, 1024, 1024]
         ↓
    ConvNeXt Encoder
         ↓
    Features: [96, 192, 384, 768] @ [64, 32, 16, 8]
         ↓
    ┌─────────────────┬─────────────────┐
    │   Cyt Decoder   │   Nuc Decoder   │
    │   + Attention   │   + Attention   │
    └─────────────────┴─────────────────┘
         ↓                    ↓
    Cyt Mask              Nuc Mask
    [B, 1, 1024, 1024]   [B, 1, 1024, 1024]
```

#### Modules:

**CBAM (Convolutional Block Attention Module)**
- Channel Attention: Uses both avg and max pooling
- Spatial Attention: Focuses on important spatial locations
- Reduction ratio: 16

**Attention Gate**
- Gates skip connections from encoder
- Suppresses irrelevant features
- Enhances relevant spatial regions

---

### 2. Classification Model: ConvNeXt Attention Classifier

**Architecture:** `ConvNeXtAttentionClassifier`

#### Key Components:

- **Backbone**: ConvNeXt-Tiny (modified input)
  - Custom first layer for **5-channel input**:
    - 3 RGB channels (original image)
    - 2 segmentation masks (cytoplasm + nucleus)
  - Output: 768-dimensional features

- **SE Block**: Squeeze-and-Excitation
  - Channel-wise attention
  - Reduction ratio: 16

- **Attention MLP**: Multi-layer classifier
  - Self-attention on features (4 heads)
  - Layer normalization
  - Dropout regularization (0.4, 0.3)

#### Architecture Details:

```python
Input: [B, 5, 256, 256]
  (3 RGB + 2 Masks)
         ↓
    ConvNeXt Features
    [B, 768, H, W]
         ↓
    SE Block (Channel Attention)
         ↓
    Global Average Pooling
    [B, 768]
         ↓
    ┌─────────────────────┐
    │   Attention MLP     │
    │   768 → 512 → 256   │
    │   + Self-Attention  │
    └─────────────────────┘
         ↓
    Class Predictions
    [B, num_classes]
```

#### Classification Layers:

1. **FC Layer 1**: 768 → 512 (BatchNorm + ReLU + Dropout 0.4)
2. **Self-Attention**: 4-head multi-head attention
3. **FC Layer 2**: 512 → 256 (BatchNorm + ReLU + Dropout 0.3)
4. **Output Layer**: 256 → num_classes

**Number of Classes**: 5 (cervical cancer stages)

---

1. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm
pip install opencv-python
pip install albumentations
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install tensorboard
```

---

## 💻 Usage

### Quick Start

#### 1️⃣ **Stage 1: Train Segmentation Model**

```bash
python main_segment.py
```

**What it does:**
- Trains the Dual-Head Attention U-Net
- Segments cytoplasm and nucleus regions
- Saves trained model weights
- Generates segmentation masks for dataset

**Output:**
- Model checkpoint: `checkpoints/segmentation_best.pth`
- Segmentation masks: `outputs/masks/`

---

#### 2️⃣ **Stage 2: Train Classification Model**

```bash
python main_classify.py
```

**What it does:**
- Loads pre-trained segmentation model
- Generates masks for training images
- Trains ConvNeXt classifier on RGB + masks (5 channels)
- Classifies cells into cancer stages

**Output:**
- Model checkpoint: `checkpoints/classification_best.pth`
- Training logs: `logs/classification/`

---

#### 3️⃣ **Evaluate Models**

```bash
python evaluate.py
```

**What it does:**
- Evaluates segmentation performance (IoU, Dice score)
- Evaluates classification performance (Accuracy, F1-score)
- Generates confusion matrices
- Saves evaluation metrics

---

## 🎓 Training Pipeline

1. **Data Preparation**
   - Organize images in appropriate directories
   - Apply preprocessing (resize, normalize)
   - Create train/validation/test splits

2. **Segmentation Training**
   ```bash
   python main_segment.py --epochs 100 --batch-size 8 --lr 1e-4
   ```
   - Input: RGB images (1024×1024)
   - Output: Cytoplasm + Nucleus masks
   - Loss: Dice Loss + BCE Loss
   - Optimizer: AdamW

3. **Mask Generation**
   - Use trained segmentation model
   - Generate masks for entire dataset
   - Save masks for classification stage

4. **Classification Training**
   ```bash
   python main_classify.py --epochs 50 --batch-size 16 --lr 1e-4
   ```
   - Input: 5-channel images (RGB + 2 masks, 256×256)
   - Output: Class probabilities (5 classes)
   - Loss: Cross-Entropy Loss
   - Optimizer: AdamW with scheduler
<img width="1575" height="409" alt="SEG" src="https://github.com/user-attachments/assets/dce9bb49-155c-4386-a8d8-6e18464e831a" />

---

## 📊 Evaluation

### Segmentation Metrics

- **IoU (Intersection over Union)**: Overlap between predicted and ground truth masks
- **Dice Coefficient**: 2×overlap / (area1 + area2)
- **Pixel Accuracy**: Correctly classified pixels

### Classification Metrics

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: Class-wise predictions
- **ROC-AUC**: Area under ROC curve

## 🎯 Model Performance

### Expected Performance Metrics

| Model | Metric | Score |
|-------|--------|-------|
| Segmentation | Dice (Cytoplasm) | ~0.9511 |
| Segmentation | Dice (Nucleus) | ~0.9329 |
| Classification | Accuracy | ~99.26% |
| Classification | F1-Score (macro) | ~0.9926 |

*Note: Actual performance depends on dataset quality and training configuration*

---
Made with ❤️ from NiceGuy
