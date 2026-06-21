# Facial Expression Recognition & Video Expression Tagging using Custom CNN

## Overview

This project implements a complete Facial Expression Recognition System using Convolutional Neural Networks (CNNs) built entirely from scratch with PyTorch.

The project began as a seven-class facial emotion recognition system trained on facial expression images and was later extended with a custom three-class model capable of recognizing **Pain**, **Distress**, and **Normal** facial expressions using a manually collected dataset.

To further enhance the project, an automated **video facial expression recognition pipeline** was developed. The system processes videos frame-by-frame, detects faces using MediaPipe, predicts facial expressions using the trained CNN models, and generates a tagged output video containing bounding boxes, predicted expressions, and confidence scores.

The project demonstrates the complete deep learning workflow including dataset collection, preprocessing, CNN architecture design, model training, evaluation, image prediction, and video-based inference.

---

# Features

* Custom CNN architecture built completely from scratch
* Seven-class facial emotion recognition
* Three-class Pain / Distress / Normal recognition
* Manual dataset collection and preprocessing
* Automatic face detection using MediaPipe
* Image-based facial expression prediction
* Video-based facial expression recognition
* Automatic expression tagging in videos
* Bounding box visualization
* Confidence score display
* Training loss visualization
* Validation accuracy visualization
* Confusion matrix generation
* Modular project architecture

---

# Emotion Recognition Classes

The primary emotion recognition model classifies:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

# Pain Recognition Classes

The specialized pain recognition model classifies:

* Pain
* Distress
* Normal

---

# Technologies Used

## Programming Language

* Python

## Deep Learning

* PyTorch
* Torchvision

## Computer Vision

* OpenCV
* MediaPipe

## Image Processing

* Pillow (PIL)
* NumPy

## Development Environment

* VS Code
* Linux (ChromeOS Container)

---

# Dataset

## Emotion Recognition Dataset

The original model uses a facial emotion dataset containing seven emotion categories.

### Dataset Statistics

| Dataset    | Images |
| ---------- | ------ |
| Training   | 28,821 |
| Validation | 7,066  |
| Total      | 35,887 |

---

## Pain Recognition Dataset

A custom dataset was manually collected from publicly available facial images.

### Dataset Statistics

| Class     | Images  |
| --------- | ------- |
| Pain      | 123     |
| Distress  | 148     |
| Normal    | 134     |
| **Total** | **405** |

Dataset preprocessing included:

* Image validation
* Face cropping
* Dataset cleaning
* Train/Validation splitting

---

# Project Structure

```text
facial-expression-recognition/

├── dataset/
│   └── images/
│       ├── train/
│       └── validation/
│
├── final_dataset/
│   ├── train/
│   │   ├── pain/
│   │   ├── distress/
│   │   └── normal/
│   └── validation/
│       ├── pain/
│       ├── distress/
│       └── normal/
│
├── models/
│   ├── emotion_cnn.pth
│   ├── pain_distress_cnn.pth
│   └── pain_distress_cnn_80pct.pth
│
├── reports/
│   ├── confusion_matrix.png
│   ├── training_loss.png
│   └── validation_accuracy.png
│
├── video/
│   ├── detector.py
│   ├── predictor.py
│   ├── video_processor.py
│   └── __init__.py
│
├── model.py
├── model_pain.py
├── train.py
├── train_pain.py
├── predict.py
├── predict_pain.py
├── predict_video.py
├── evaluate_pain.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

# CNN Architecture

Both models use a custom CNN consisting of:

## Feature Extraction

* Conv2D
* Batch Normalization
* ReLU
* Max Pooling

Repeated across four convolutional blocks.

---

## Classification Layers

* Fully Connected Layer
* ReLU
* Dropout
* Fully Connected Layer
* ReLU
* Dropout
* Output Layer

Output neurons:

* EmotionCNN → 7 classes
* PainCNN → 3 classes

---

# Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Image Size    | 64 × 64          |
| Batch Size    | 64               |
| Optimizer     | Adam             |
| Learning Rate | 0.0005           |
| Loss Function | CrossEntropyLoss |
| Epochs        | 30               |

---

# Data Preprocessing

The following preprocessing techniques were applied:

* Image resizing (64 × 64)
* Face cropping
* Tensor conversion
* Dataset cleaning
* Train/Validation split

---

# Results

## Emotion Recognition Model

| Metric                   | Value |
| ------------------------ | ----- |
| Classes                  | 7     |
| Best Validation Accuracy | ~60%  |

---

## Pain Recognition Model

| Metric                   | Value      |
| ------------------------ | ---------- |
| Classes                  | 3          |
| Best Validation Accuracy | **80.00%** |

---

# Video Facial Expression Recognition

The project supports automatic facial expression recognition from videos.

## Pipeline

Input Video

↓

Frame Extraction

↓

Face Detection (MediaPipe)

↓

Face Cropping

↓

Image Preprocessing

↓

CNN Prediction

↓

Confidence Calculation

↓

Bounding Box & Label Generation

↓

Tagged Output Video

Each detected face is automatically annotated with:

* Bounding Box
* Predicted Expression
* Confidence Score

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Emotion Model

```bash
python train.py
```

---

## Train Pain Recognition Model

```bash
python train_pain.py
```

---

## Predict Emotion from Image

```bash
python predict.py path/to/image.jpg
```

Example:

```bash
python predict.py dataset/images/validation/happy/10019.jpg
```

---

## Predict Pain / Distress

```bash
python predict_pain.py path/to/image.jpg
```

---

## Evaluate Pain Model

```bash
python evaluate_pain.py
```

---

## Process a Video

```bash
python predict_video.py \
    --input input.mp4 \
    --output tagged_video.mp4
```

The generated video automatically contains:

* Face Detection
* Facial Expression Recognition
* Bounding Boxes
* Emotion Labels
* Confidence Scores

---

# Performance Analysis

The project demonstrates the complete lifecycle of a computer vision system.

The original seven-class emotion recognition model achieved satisfactory classification performance and established the foundation for facial expression analysis.

The specialized pain recognition model achieved a best validation accuracy of **80%**, demonstrating improved performance on the targeted three-class problem.

The system was further extended with an automated video processing pipeline capable of recognizing and tagging facial expressions frame-by-frame using the trained CNN models.

---

# Learning Outcomes

This project provided practical experience with:

* Deep Learning
* Convolutional Neural Networks
* Image Classification
* Dataset Collection
* Data Cleaning
* Face Detection
* MediaPipe
* OpenCV
* PyTorch
* Video Processing
* Model Evaluation
* Performance Analysis
* CNN Deployment
* Computer Vision Pipeline Development

---

# Future Improvements

* Face Tracking
* Multi-person Recognition
* Prediction Smoothing
* Confidence Thresholding
* Real-time Webcam Recognition
* Flask Deployment
* Streamlit Deployment
* Mobile Deployment
* ONNX Optimization
* TensorRT Acceleration
* Transformer-based Vision Models

---

# Author

**Anubhab Chakraborty**

Built as a deep learning and computer vision project to explore facial expression recognition, pain detection, and automated video facial expression tagging using custom CNN architectures developed entirely with PyTorch.
