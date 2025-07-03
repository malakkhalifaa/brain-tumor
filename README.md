
# Brain Tumor Detection Using Deep Learning

This project focuses on building a deep learning-based image classification system to detect brain tumors using MRI scans. The model identifies four classes: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. Leveraging transfer learning, the system utilizes VGG16 pretrained on ImageNet for feature extraction, with custom classification layers for accurate prediction.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Performance Metrics](#performance-metrics)
- [Web Deployment (In Progress)](#web-deployment-in-progress)

---

## Overview

Accurate and early detection of brain tumors is critical for effective treatment. This project uses convolutional neural networks (CNNs) to automate tumor classification, assisting radiologists and medical practitioners with reliable, real-time predictions.

---

## Dataset

- Source: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Format: MRI images grouped into four directories:
  - `glioma`
  - `meningioma`
  - `pituitary`
  - `notumor`
- Preprocessing: Grayscale conversion, resizing to 224×224, normalization

---

## Model Architecture

The architecture builds on **VGG16**, a 16-layer CNN pretrained on ImageNet.

- **Base Model**: VGG16 with frozen convolutional layers
- **Added Layers**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(4, activation='softmax')`
- **Optimizer**: Adam (`lr=1e-4`)
- **Loss Function**: Categorical Crossentropy
- **Activation**: ReLU (hidden), Softmax (output)

---

## Training Details

- Platform: Google Colab (GPU runtime)
- Epochs: 5
- Batch Size: 32
- Augmentation: Rotation, flipping, zoom
- Validation split: 20%

---

## Performance Metrics

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Glioma        | 0.96      | 0.94   | 0.95     |
| Meningioma    | 0.95      | 0.93   | 0.94     |
| Pituitary     | 0.96      | 0.97   | 0.96     |
| No Tumor      | 0.99      | 0.98   | 0.98     |

- **Overall Accuracy**: 97.5%
- **ROC AUC**: Plotted for all classes
- **Confusion Matrix**: Available in notebook

---

## Web Deployment (In Progress)

A lightweight web application is under development using **FastAPI** and **HTML/CSS**:

- Upload MRI scan → Receive instant prediction
- Deployment framework: FastAPI backend + static HTML frontend
- Future support for Docker & cloud deployment (Render/Heroku)

---


