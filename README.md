# Facial Expression Recognition using Deep Learning (PyTorch)

## Overview

This project presents a complete deep learning pipeline for **Facial Expression Recognition** using Convolutional Neural Networks (CNNs) implemented from scratch with **PyTorch**.

The repository contains two independent facial expression recognition models:

* **General Emotion Recognition Model** trained on the FER2013 dataset to classify seven facial emotions.
* **Pain Expression Recognition Model** trained on a custom dataset to classify facial expressions into **Pain**, **Distress**, and **Normal** categories.

The project demonstrates the complete deep learning workflow, including dataset preparation, preprocessing, CNN architecture design, model training, evaluation, and inference.

---

# Features

* Custom CNN architecture built entirely from scratch
* Facial emotion recognition (7 classes)
* Pain expression recognition (3 classes)
* Automated dataset cleaning
* Face detection and cropping using MediaPipe
* Image preprocessing and augmentation
* Training and validation dataset splitting
* Model evaluation using confusion matrix and classification report
* Image-based prediction scripts
* Automatic best model checkpointing
* Training loss and validation accuracy visualization

---

# Models Included

## Model 1 — General Facial Emotion Recognition

Classifies facial expressions into:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

Dataset:

* FER2013

Output Model:

```text
models/emotion_cnn.pth
```

---

## Model 2 — Pain Expression Recognition

Classifies facial expressions into:

* Pain
* Distress
* Normal

Dataset:

Custom dataset collected from Google Images and processed using MediaPipe face detection.

Output Model:

```text
models/pain_distress_cnn.pth
```

---

# Technologies Used

## Programming Language

* Python

## Deep Learning Framework

* PyTorch
* TorchVision

## Computer Vision

* OpenCV
* MediaPipe
* Pillow (PIL)

## Data Science

* NumPy
* Matplotlib
* Scikit-learn

## Development Environment

* VS Code
* Linux (ChromeOS Container)

---

# Datasets

## FER2013 Dataset

Used for general emotion recognition.

| Dataset    | Images |
| ---------- | -----: |
| Training   | 28,821 |
| Validation |  7,066 |
| Total      | 35,887 |

Emotion Classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## Custom Pain Dataset

Created manually using Google Images.

| Class     |  Images |
| --------- | ------: |
| Pain      |     123 |
| Distress  |     148 |
| Normal    |     134 |
| **Total** | **405** |

Dataset preparation pipeline:

```text
Google Images
      │
      ▼
Dataset Cleaning
      │
      ▼
Face Detection (MediaPipe)
      │
      ▼
Face Cropping
      │
      ▼
Image Resizing
      │
      ▼
Train / Validation Split
```

---

# Project Structure

```text
facial-expression-recognition/

│
├── dataset/
│   └── images/
│
├── pain_dataset/
│
├── processed_dataset/
│
├── final_dataset/
│
├── models/
│   ├── emotion_cnn.pth
│   └── pain_distress_cnn.pth
│
├── reports/
│   ├── confusion_matrix.png
│   ├── training_loss.png
│   └── validation_accuracy.png
│
├── model.py
├── train.py
├── predict.py
│
├── model_pain.py
├── train_pain.py
├── predict_pain.py
├── evaluate_pain.py
│
├── clean_dataset.py
├── crop_faces.py
├── preprocess_dataset.py
├── split_dataset.py
│
├── requirements.txt
└── README.md
```

---

# CNN Architecture

The custom CNN consists of:

### Feature Extraction

* Conv2D (3 → 32)

* Batch Normalization

* ReLU

* Max Pooling

* Conv2D (32 → 64)

* Batch Normalization

* ReLU

* Max Pooling

* Conv2D (64 → 128)

* Batch Normalization

* ReLU

* Max Pooling

* Conv2D (128 → 256)

* Batch Normalization

* ReLU

* Max Pooling

### Classification

* Adaptive Average Pooling
* Fully Connected Layer (4096 → 512)
* ReLU
* Dropout
* Fully Connected Layer (512 → 256)
* ReLU
* Dropout
* Output Layer

Outputs:

* 7 neurons for Emotion Recognition
* 3 neurons for Pain Recognition

---

# Data Augmentation

The following augmentation techniques are applied during training:

* Resize (64 × 64)
* Random Horizontal Flip
* Random Rotation (10°)
* Random Affine Transformation
* Brightness Adjustment
* Contrast Adjustment
* Tensor Conversion

---

# Training Configuration

| Parameter      |                                Value |
| -------------- | -----------------------------------: |
| Framework      |                              PyTorch |
| Image Size     |                              64 × 64 |
| Batch Size     | 32 (Pain Model) / 64 (Emotion Model) |
| Optimizer      |                                 Adam |
| Learning Rate  |                               0.0005 |
| Loss Function  |                     CrossEntropyLoss |
| Scheduler      |                    ReduceLROnPlateau |
| Early Stopping |                              Enabled |

---

# Results

## General Emotion Recognition

| Metric                   |  Value |
| ------------------------ | -----: |
| Training Accuracy        | 95.58% |
| Best Validation Accuracy | 61.66% |

The experiment demonstrates successful learning but also highlights overfitting, emphasizing the importance of regularization and evaluation.

---

## Pain Expression Recognition

| Metric                            |       Value |
| --------------------------------- | ----------: |
| Best Validation Accuracy Achieved | **80.00%*** |

* Accuracy may vary slightly between training runs because of random initialization and data augmentation.

Evaluation includes:

* Confusion Matrix
* Classification Report
* Precision
* Recall
* F1-Score
* Training Loss Graph
* Validation Accuracy Graph

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Emotion Recognition Model

```bash
python train.py
```

---

## Train Pain Recognition Model

```bash
python train_pain.py
```

---

## Evaluate Pain Recognition Model

```bash
python evaluate_pain.py
```

---

## Predict Emotion

```bash
python predict.py image.jpg
```

---

## Predict Pain Expression

```bash
python predict_pain.py "image.jpg"
```

---

# Learning Outcomes

This project explores the following deep learning concepts:

* Convolutional Neural Networks (CNNs)
* Image Classification
* Facial Expression Recognition
* Data Augmentation
* Face Detection using MediaPipe
* Batch Normalization
* Dropout Regularization
* Learning Rate Scheduling
* Early Stopping
* Hyperparameter Tuning
* Confusion Matrix Analysis
* Precision, Recall, and F1-Score
* Deep Learning using PyTorch

---

# Future Improvements

Potential enhancements include:

* Larger and more diverse custom pain dataset
* Transfer Learning (ResNet, EfficientNet, MobileNet)
* Attention-based CNN architectures
* Real-time webcam inference
* Web deployment using Flask or Streamlit
* Mobile deployment using TensorFlow Lite or ONNX

---

# Author

**Anubhab Chakraborty**

This project was developed as part of deep learning research in facial expression recognition using PyTorch. It demonstrates both traditional facial emotion recognition and custom pain/distress recognition through end-to-end CNN development, dataset preparation, training, evaluation, and inference.
