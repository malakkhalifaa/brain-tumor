# Brain Tumor Detection Using Deep Learning

This repository presents a deep learning pipeline for **automatic classification of brain tumors** from MRI scans. The model supports four target classes: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. It uses **transfer learning** with the **VGG16** architecture for robust feature extraction and is deployed through a fully functional **web interface** for real-time image predictions.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset Description](#dataset-description)  
- [Preprocessing](#preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training Details](#training-details)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Web App Deployment](#web-app-deployment)  

---

## Project Overview

Brain tumor detection from MRI images is critical for early diagnosis and treatment. Manual analysis is time-consuming and error-prone. This project applies **Convolutional Neural Networks (CNNs)** to automate tumor classification. The solution includes:

- MRI classification into four categories  
- Transfer learning with VGG16  
- Augmented training for improved generalization  
- Flask and FastAPI web deployment  
- Clean UI for uploading images and displaying predictions  

---

## Dataset Description

- Source: Kaggle - Brain Tumor MRI Dataset  
- Total Images: Over 7,000 MRI scans  
- Classes: glioma, meningioma, pituitary, notumor  
- Format: .jpg images organized into folders by class  
- Split: 80% training, 20% testing  

---

## Preprocessing

- Image resizing to 128×128  
- Grayscale images converted to RGB  
- Normalization to [0, 1]  
- One-hot encoding of class labels  
- Augmentations used:
  - Random brightness and contrast
  - Rotation
  - Flipping
  - Zoom
  - Width/height shift

![image](https://github.com/user-attachments/assets/4218dd34-292b-4c70-b6f9-712eba35f495)

---

## Model Architecture

- Base: VGG16(weights='imagenet', include_top=False)  
- Layers Frozen: All except last few layers  
- Custom Head:
  - Flatten  
  - Dropout(0.3)  
  - Dense(128, activation='relu')  
  - Dropout(0.2)  
  - Dense(4, activation='softmax')  
- Loss: sparse_categorical_crossentropy  
- Optimizer: Adam with lr=1e-4  
- Metric: sparse_categorical_accuracy

![image](https://github.com/user-attachments/assets/f1591a79-2e1c-42d0-9027-3c1cdca734a2)

---

## Training Details

- Platform: Google Colab (GPU-enabled)  
- Epochs: 5  
- Batch Size: 20  
- Training Accuracy:
  - Epoch 1: 80.9%  
  - Epoch 5: 97.4%  
- Final Training Loss: 0.06  

![image](https://github.com/user-attachments/assets/4ba6d8be-d406-425d-b035-0b2ed18ccf20)
![image](https://github.com/user-attachments/assets/35d18b53-5cef-433b-8f3a-98c5623b648a)

## Model Confusion Matrix 
<img width="1641" height="1397" alt="image" src="https://github.com/user-attachments/assets/e5004a85-935d-41f7-9b36-1c44ba621fdb" />

## ROC Curve Plot
<img width="2203" height="1543" alt="image" src="https://github.com/user-attachments/assets/d814aa13-84d4-433f-9313-6ab75f4e9000" />

---

## Evaluation Metrics

Test Set: 1,311 images

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Glioma      | 0.74      | 0.96   | 0.84     | 300     |
| Meningioma  | 0.96      | 0.39   | 0.55     | 306     |
| Pituitary   | 0.89      | 1.00   | 0.94     | 405     |
| No Tumor    | 0.86      | 0.99   | 0.92     | 300     |

![image](https://github.com/user-attachments/assets/39b9903e-8aab-4091-871f-e3008a1a3d2d)

- Overall Accuracy: 85.0%  
- Macro F1 Score: 0.81  
- ROC AUC per class:  
  - Glioma: 0.94  
  - Meningioma: 0.87  
  - Pituitary: 0.98  
  - No Tumor: 0.96  

---

## Web App Deployment

### Flask App (`app.py`)
- Simple image upload and prediction return  
- Renders prediction on index.html  

### FastAPI App (`main.py`)
- Async backend with Jinja2  
- Displays uploaded image and prediction  
- Lightweight and fast UI  

### Frontend Features
- HTML/CSS styling  
- Upload form  
- Real-time class result and formatting  

![image](https://github.com/user-attachments/assets/18ee6606-d272-4457-be94-b4dfc6973e54)
![image](https://github.com/user-attachments/assets/b2c05fc1-78af-48b4-937b-8758a9d5c3c9)
