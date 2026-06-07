# Facial Expression Recognition Using Convolutional Neural Networks (CNN)

## Overview

This project implements a Facial Expression Recognition System using Deep Learning and Computer Vision techniques. The system is capable of classifying human facial expressions into seven different emotion categories by analyzing facial images.

A custom Convolutional Neural Network (CNN) was designed and trained from scratch using PyTorch on a facial expression dataset containing thousands of labeled images. The trained model can predict emotions from previously unseen facial images.

---

## Features

* Custom CNN architecture built from scratch
* Emotion classification into seven categories
* Training and validation pipeline using PyTorch
* Image preprocessing and augmentation support
* Model evaluation using validation accuracy
* Trained model export and reuse
* Emotion prediction from new facial images

---

## Emotion Classes

The model classifies images into the following emotions:

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

### Machine Learning Framework

* PyTorch
* TorchVision

### Image Processing

* Pillow (PIL)

### Development Environment

* Visual Studio Code
* Linux

---

## Dataset

The project uses a Facial Expression Recognition dataset containing facial images categorized into seven emotions.

Dataset Structure:

```text
dataset/
└── images/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    │
    └── validation/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```

Dataset Statistics:

* Training Images: 28,821
* Validation Images: 7,066
* Total Images: 35,887

---

## CNN Architecture

The custom CNN consists of:

```text
Input Image (96 × 96 × 3)
        │
        ▼
Conv2D (32 Filters)
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling
        │
        ▼
Conv2D (64 Filters)
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling
        │
        ▼
Conv2D (128 Filters)
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling
        │
        ▼
Flatten
        │
        ▼
Fully Connected Layer
        │
        ▼
Dropout
        │
        ▼
Output Layer (7 Classes)
```

---

## Project Structure

```text
facial-expression-recognition/
│
├── dataset/
│
├── models/
│   └── emotion_cnn.pth
│
├── model.py
├── train.py
├── predict.py
│
├── requirements.txt
├── README.md
│
└── screenshots/
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Anubhab36/Facial-Expression-Recognition

cd facial-expression-recognition
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the CNN model:

```bash
python train.py
```

The training process:

1. Loads the dataset
2. Preprocesses images
3. Trains the CNN
4. Evaluates on validation data
5. Saves the trained model

The trained model is saved as:

```text
models/emotion_cnn.pth
```

---

## Model Performance

Training Results:

| Epoch | Training Loss | Validation Accuracy |
| ----- | ------------- | ------------------- |
| 1     | 1.6736        | 41.15%              |
| 2     | 1.4679        | 46.53%              |
| 3     | 1.3487        | 49.87%              |

Final Validation Accuracy:

```text
49.87%
```

---

## Running Predictions

To predict emotion from a facial image:

```bash
python predict.py image.jpg
```

Example:

```bash
python predict.py sample.jpg
```

Output:

```text
Predicted Emotion: happy
```

---

## Sample Workflow

```text
Input Face Image
        │
        ▼
Image Preprocessing
        │
        ▼
CNN Model
        │
        ▼
Emotion Classification
        │
        ▼
Predicted Emotion
```

---

## Learning Outcomes

Through this project, the following concepts were implemented and explored:

* Deep Learning Fundamentals
* Convolutional Neural Networks (CNNs)
* Image Classification
* Computer Vision
* Model Training and Validation
* Hyperparameter Tuning
* PyTorch Framework
* Dataset Handling using DataLoader
* Model Deployment for Inference

---

## Future Improvements

Possible enhancements include:

* Real-time webcam emotion detection
* Transfer learning using ResNet or EfficientNet
* Accuracy optimization through data augmentation
* Confusion matrix visualization
* Training and validation loss graphs
* Web application deployment using Streamlit

---

## Author

Anubhab Chakraborty

Bachelor of Technology (B.Tech)

Artificial Intelligence & Machine Learning

---

## License

This project is developed for educational and learning purposes.
