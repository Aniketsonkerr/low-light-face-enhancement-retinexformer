# Low-Light Face Image Enhancement using RetinexFormer

## Overview

Low-light conditions significantly degrade image quality by introducing poor illumination, low contrast, noise, and loss of important facial features. These issues affect both human perception and the performance of downstream computer vision applications such as face recognition, surveillance, biometric authentication, and object detection.

This project presents a **Face-Aware Low-Light Image Enhancement Pipeline** built upon **RetinexFormer (ICCV 2023)**. Instead of enhancing the entire image uniformly, the proposed system first detects facial regions and then applies RetinexFormer-based enhancement to restore facial details while suppressing noise and illumination artifacts.

The project combines classical computer vision techniques with state-of-the-art transformer-based image restoration to achieve efficient and high-quality low-light face enhancement.

---

## Problem Statement

Traditional low-light image enhancement methods suffer from several limitations:

- Multi-stage pipelines that require separate enhancement, denoising, and color correction modules.
- Noise amplification after brightening dark images.
- CNN-based approaches struggle to model long-range dependencies.
- Standard vision transformers are computationally expensive.
- Whole-image enhancement wastes resources on irrelevant background regions.

In low-light facial images, these limitations often result in:

- Loss of facial details
- Distorted skin tones
- Poor visibility
- Increased computational cost

---

## Proposed Solution

The proposed system introduces a **two-stage face-aware enhancement pipeline**.

### Stage 1: Face Detection & ROI Extraction

Input images are processed using an OpenCV Haar Cascade Face Detector.

The detector:

- Identifies facial regions.
- Creates bounding boxes.
- Applies 20% contextual padding around detected faces.
- Extracts the facial Region of Interest (ROI).

This ensures that enhancement resources are focused on the most important area of the image.

### Stage 2: RetinexFormer-Based Enhancement

The extracted facial region is passed through RetinexFormer for enhancement.

RetinexFormer performs:

- Illumination estimation
- Adaptive brightness restoration
- Noise reduction
- Contrast enhancement
- Detail reconstruction

The enhanced facial image is then saved as the final output.

---

# Why RetinexFormer?

RetinexFormer addresses key shortcomings of earlier low-light enhancement methods.

### Existing CNN-based Methods

- Limited receptive field
- Poor global feature modeling
- Difficulty preserving fine details

### Traditional Transformers

- High memory consumption
- Quadratic self-attention complexity
- Slow inference on high-resolution images

### RetinexFormer Advantages

- Illumination-guided self-attention
- Efficient transformer design
- Better facial detail restoration
- Lower computational cost
- Simultaneous enhancement and denoising

---

# RetinexFormer Architecture

## 1. Illumination Estimator

The Illumination Estimator learns lighting information from the input image.

Outputs:

- Illumination Features (`illu_fea`)
- Illumination Map (`illu_map`)

Purpose:

- Identify underexposed regions
- Estimate missing illumination

### Intuition

Instead of blindly brightening the entire image, the model first learns where light is lacking.

---

## 2. Illumination-Guided Multi-Head Self-Attention (IG-MSA)

The key innovation of RetinexFormer.

Traditional self-attention:

- Treats all regions equally
- Consumes large computational resources

IG-MSA:

- Uses illumination information as guidance
- Focuses attention on dark regions
- Improves restoration quality
- Reduces computational complexity

### Intuition

The model learns:

> Where should attention be focused to restore visibility?

instead of wasting attention uniformly across the image.

---

## 3. Illumination Guided Attention Blocks (IGAB)

Each IGAB contains:

- IG-MSA
- Feed Forward Network
- Residual Connections
- Layer Normalization

These blocks progressively refine image features and improve restoration quality.

---

## 4. Denoiser Network

A U-Net style encoder-decoder architecture.

### Encoder

- Extracts deep hierarchical features
- Reduces spatial dimensions
- Applies IGAB blocks

### Bottleneck

- Captures global image context
- Learns long-range dependencies

### Decoder

- Restores image resolution
- Reconstructs facial details
- Produces enhanced output

---

## 5. RetinexFormer Single Stage

Workflow:

```text
Input Image
      │
      ▼
Illumination Estimator
      │
      ▼
Illumination Map
      │
      ▼
Denoiser Network
      │
      ▼
Enhanced Image
```

---

## 6. Multi-Stage RetinexFormer

Multiple enhancement stages are stacked sequentially.

Each stage:

- Refines illumination
- Suppresses remaining noise
- Improves visual quality

The final output is a bright, clean, and detail-preserving image.

---

# System Architecture

```text
Low-Light Face Image
          │
          ▼
   Face Detection
 (Haar Cascade Classifier)
          │
          ▼
  ROI Face Extraction
          │
          ▼
 Dynamic Reflection Padding
          │
          ▼
      RetinexFormer
          │
 ┌────────┼────────┐
 │        │        │
 ▼        ▼        ▼
Illum.   IG-MSA  Denoiser
Est.     Blocks   Network
          │
          ▼
 Enhanced Face Image
          │
          ▼
      Save Output
```

---

# Custom Engineering Contributions

This project extends the original RetinexFormer framework with several practical enhancements.

---

## 1. Face-Aware Enhancement Pipeline

A custom inference pipeline was developed that performs:

- Face localization
- ROI extraction
- Face-specific enhancement

instead of applying enhancement on the entire image.

### Benefits

- Better facial restoration
- Reduced computation
- Improved enhancement quality

---

## 2. Dynamic Tensor Padding

### Problem

RetinexFormer uses a multi-scale U-Net architecture.

Arbitrary cropped dimensions such as:

```text
435 × 287
533 × 361
```

cause tensor size mismatches during upsampling.

### Solution

Implemented dynamic reflection padding:

```python
F.pad()
```

Steps:

1. Round dimensions to nearest multiple of 16.
2. Apply reflection padding.
3. Run inference.
4. Remove padding after enhancement.

Result:

- Stable execution
- No dimension mismatch errors

---

## 3. Fail-Safe Fallback Mechanism

Extremely dark images may fail face detection.

Implemented fallback logic:

```text
Face Detected?
      │
 ┌────┴────┐
 Yes      No
  │        │
 Face ROI  Full Image
```

This guarantees:

- Zero processing failures
- Robust inference

---

## 4. Automatic Resource Management

Implemented:

- Auto-download of Haar Cascade XML file
- Relative path management
- Platform-independent execution

This improves reproducibility and portability.

---

# Repository Structure

```text
Low-Light-Face-Enhancement/
│
├── basicsr/
│   ├── models/
│   ├── archs/
│   ├── data/
│   └── metrics/
│
├── Enhancement/
│   ├── infer_faces.py
│   ├── train.py
│   ├── test.py
│   └── test_from_dataset.py
│
├── Options/
│   └── RetinexFormer_LOL_v1.yml
│
├── pretrained_weights/
│   └── LOL_v1.pth
│
├── data/
│   └── CustomFaces/
│
├── results/
│   └── CustomFaces_Enhanced/
│
├── requirements.txt
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Low-Light-Face-Enhancement.git

cd Low-Light-Face-Enhancement
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Download Pretrained Weights

Place pretrained weights inside:

```text
pretrained_weights/
```

Example:

```text
pretrained_weights/
└── LOL_v1.pth
```

---

# Dataset

## LOL-v1 Dataset

Directory structure:

```text
LOL-v1/
├── eval15/
│   ├── low/
│   └── high/
│
└── our485/
    ├── low/
    └── high/
```

---

# Running Face Enhancement

Place input images in:

```text
data/CustomFaces/Test/input/
```

Execute:

```bash
python Enhancement/infer_faces.py
```

Enhanced outputs will be saved in:

```text
results/CustomFaces_Enhanced/
```

---

# Training

```bash
python Enhancement/train.py \
--opt Options/RetinexFormer_LOL_v1.yml
```

---

# Testing

```bash
python Enhancement/test_from_dataset.py \
--opt Options/RetinexFormer_LOL_v1.yml \
--weights pretrained_weights/LOL_v1.pth \
--dataset LOL_v1
```

---

# Performance Metrics

The model is evaluated using:

## PSNR

Peak Signal-to-Noise Ratio

Measures reconstruction quality.

Higher is better.

---

## SSIM

Structural Similarity Index

Measures visual similarity between enhanced and ground truth images.

Higher is better.

---

# Applications

- Face Recognition Systems
- Smart Surveillance
- Security Monitoring
- Mobile Photography
- Video Conferencing
- Biometric Authentication
- Low-Light Object Detection

---

# Hardware Configuration

| Component             | Specification                      |
| --------------------- | ---------------------------------- |
| GPU                   | NVIDIA GeForce RTX 4050 Laptop GPU |
| Framework             | PyTorch                            |
| Backend               | CUDA                               |
| Vision Library        | OpenCV                             |
| Restoration Framework | BasicSR                            |
| Model                 | RetinexFormer                      |

---

# Key Contributions

✅ Face-Aware Enhancement Pipeline

✅ ROI-Based Processing

✅ Dynamic Tensor Padding

✅ Automatic Cascading Resource Download

✅ Fail-Safe Full Image Processing

✅ Cross-Platform Execution

✅ RetinexFormer Integration for Facial Restoration

---

# Future Work

- Real-time video enhancement
- Face-aware attention mechanisms
- Face recognition integration
- Edge-device deployment
- Custom facial low-light dataset training
- YOLO/MTCNN based face detection

---

# Authors

**Aniket Sonker**

M.Tech Project

Low-Light Face Image Enhancement using RetinexFormer

---

# Acknowledgements

- RetinexFormer (ICCV 2023)
- LOL-v1 Dataset
- BasicSR Framework
- PyTorch
- OpenCV Community

---

# License

This project is intended for academic and research purposes. Please refer to the licenses of RetinexFormer, LOL-v1, and other third-party dependencies before commercial use.
