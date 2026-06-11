# Facial Expression Recognition using Custom CNN

## Overview

This project implements a Facial Expression Recognition system using a Convolutional Neural Network (CNN) built and trained from scratch with PyTorch. The model classifies facial images into seven different emotion categories and demonstrates the complete deep learning workflow, including data preprocessing, model design, training, evaluation, and inference.

The primary objective of this project was to understand the fundamentals of deep learning by building a CNN architecture manually rather than relying on pre-trained models or transfer learning techniques.

---

## Features

* Custom CNN architecture built from scratch
* Facial emotion classification into 7 categories
* Data preprocessing and augmentation
* Model training and evaluation using PyTorch
* Image-based emotion prediction
* Training and validation performance analysis
* Overfitting analysis through training and validation metrics

---

## Emotion Classes

The model classifies images into the following categories:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* PyTorch
* Torchvision
* Pillow (PIL)
* NumPy

### Development Environment

* VS Code
* Linux (ChromeOS Container)

---

## Dataset

The project uses a facial expression dataset containing images categorized into seven emotion classes.

### Dataset Statistics

| Dataset        | Images |
| -------------- | ------ |
| Training Set   | 28,821 |
| Validation Set | 7,066  |
| Total          | 35,887 |

---

## Project Structure

```text
facial-expression-recognition/
│
├── dataset/
│   └── images/
│       ├── train/
│       └── validation/
│
├── models/
│   └── emotion_cnn.pth
│
├── model.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
└── screenshots/
```

---

## CNN Architecture

The model consists of:

### Feature Extraction Layers

* Conv2D (3 → 32)

* Batch Normalization

* ReLU Activation

* Max Pooling

* Conv2D (32 → 64)

* Batch Normalization

* ReLU Activation

* Max Pooling

* Conv2D (64 → 128)

* Batch Normalization

* ReLU Activation

* Max Pooling

* Conv2D (128 → 256)

* Batch Normalization

* ReLU Activation

* Max Pooling

### Classification Layers

* Fully Connected Layer (4096 → 512)

* ReLU Activation

* Dropout

* Fully Connected Layer (512 → 256)

* ReLU Activation

* Dropout

* Output Layer (256 → 7)

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Image Size    | 64 × 64          |
| Batch Size    | 64               |
| Learning Rate | 0.0005           |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |
| Epochs        | 20               |

---

## Data Augmentation

To improve generalization, the following transformations were applied:

* Resize (64 × 64)
* Random Horizontal Flip
* Random Rotation (10°)
* Tensor Conversion

---

## Results

### Experiment 1

Custom CNN with Batch Normalization and Data Augmentation.

| Metric                   | Value  |
| ------------------------ | ------ |
| Best Validation Accuracy | 61.66% |

### Experiment 2

Extended training and optimization to analyze model learning behavior.

| Metric                    | Value  |
| ------------------------- | ------ |
| Training Accuracy         | 95.58% |
| Best Validation Accuracy  | 60.76% |
| Final Validation Accuracy | 59.93% |
| Training Loss             | 0.1264 |

---

## Performance Analysis

The model achieved a training accuracy of 95.58%, demonstrating its ability to learn and fit the training dataset effectively.

However, validation accuracy remained around 60%, indicating overfitting. This experiment highlights the difference between memorization and generalization in deep learning systems and demonstrates the importance of model regularization and performance evaluation on unseen data.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Predict Emotion for an Image

```bash
python predict.py path/to/image.jpg
```

Example:

```bash
python predict.py dataset/images/validation/happy/10019.jpg
```

Output:

```text
Predicted Emotion: happy
```

---

## Learning Outcomes

Through this project, the following concepts were explored:

* Convolutional Neural Networks (CNNs)
* Image Classification
* Data Augmentation
* Batch Normalization
* Model Training and Evaluation
* Hyperparameter Tuning
* Overfitting Analysis
* Deep Learning using PyTorch

---

## Future Improvements

Potential enhancements include:

* Early Stopping
* Learning Rate Scheduling
* Confusion Matrix Analysis
* Real-Time Webcam Emotion Detection
* Attention-Based CNN Architectures
* Transfer Learning using ResNet or EfficientNet
* Deployment using Flask or Streamlit

---

## Author

**Anubhab Chakraborty**

Built as a deep learning project to explore facial expression recognition and CNN-based image classification using PyTorch.
