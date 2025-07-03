# Brain Tumor Detection Using Deep Learning

This repository presents a deep learning pipeline for **automatic classification of brain tumors** from MRI scans. The model supports four target classes: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. It uses **transfer learning** with the **VGG16** architecture for robust feature extraction and is deployed through a fully functional **web interface** for real-time image predictions.

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset Description](#dataset-description)  
- [Preprocessing](#preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training Details](#training-details)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Flask and FastAPI Deployment](#flask-and-fastapi-deployment)  

---

## Project Overview

Brain tumor detection from MRI images is crucial for early diagnosis and treatment planning. Manual analysis is time-consuming, requires expert knowledge, and may lead to diagnostic delays. This project applies **Convolutional Neural Networks (CNNs)** to automate the classification process, enabling quick and reliable predictions for radiological scans.

---

## Dataset Description

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**:
  - `glioma`
  - `meningioma`
  - `pituitary`
  - `notumor`
- **Total Images**: 7,000+ across all categories
- **Format**: `.jpg` images organized into subfolders by class
- **Split**: 80% training, 20% validation

---

## Preprocessing

- Image resizing to **224×224 pixels**  
- Grayscale images converted to RGB  
- Normalization to `[0, 1]` range  
- One-hot encoding of labels  
- Data augmentation (only applied during training):
  - Random rotations
  - Horizontal/vertical flips
  - Zoom variations
  - Width/height shifts

---

## Model Architecture

A transfer learning approach was used based on the **VGG16** model:

- **Base Model**: `VGG16(weights='imagenet', include_top=False)`
  - All convolutional layers **frozen**
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(4, activation='softmax')`  
- **Optimizer**: Adam with learning rate = `1e-4`
- **Loss**: Categorical Crossentropy
- **Evaluation Metric**: Accuracy

---

## Training Details

- **Platform**: Google Colab (GPU-enabled runtime)
- **Epochs**: 5
- **Batch Size**: 32
- **Validation Split**: 20%
- **Augmentation**: Applied using `ImageDataGenerator` for training images
- **Callbacks**:
  - EarlyStopping
  - ModelCheckpoint (best model saved as `model.h5`)

---

## Evaluation Metrics

The model achieved strong performance on the validation set.

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Glioma        | 0.96      | 0.94   | 0.95     |
| Meningioma    | 0.95      | 0.93   | 0.94     |
| Pituitary     | 0.96      | 0.97   | 0.96     |
| No Tumor      | 0.99      | 0.98   | 0.98     |

- **Overall Accuracy**: 97.5%  
- **Confusion Matrix**: Visualized in the training notebook  
- **ROC AUC**: Computed and plotted for each class

---

## Flask and FastAPI Deployment

This project includes **two backend implementations**:

### 1. Flask App (`app.py`)
- Lightweight Python backend using Flask
- Accepts image uploads and returns predictions
- Displays the output on `index.html`

### 2. FastAPI App (`main.py`)
- High-performance, async-capable backend
- Uses Jinja2 for rendering
- Displays uploaded image + prediction
- Clean and responsive UI with prediction confidence

---
