# Brain Tumor Detection Using Deep Learning

This repository presents a deep learning pipeline for **automatic classification of brain tumors** from MRI scans. The model supports four target classes: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. It uses **transfer learning** with the **VGG16** architecture for robust feature extraction and is deployed through a fully functional **web interface** for real-time image predictions.

---

## Table of Contents

- Project Overview  
- Dataset Description  
- Preprocessing  
- Model Architecture  
- Training Details  
- Evaluation Metrics  
- Web App Deployment  
- Directory Structure  
- Usage Instructions  
- Planned Improvements  

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

---


