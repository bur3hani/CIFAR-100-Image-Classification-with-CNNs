# CIFAR-100 Image Classification Project

This repository contains a Convolutional Neural Network (CNN) implementation to classify images in the **CIFAR-100** dataset. The CIFAR-100 dataset consists of 60,000 color images (32x32 pixels) categorized into 100 classes.

## Project Overview

1. **Objective**:  
   - Develop a CNN to accurately classify images into one of 100 categories.
   - Demonstrate a full deep learning workflow, from data loading to model evaluation.

2. **Dataset**:  
   - **Source**: TensorFlow/Keras built-in dataset or [Kaggle CIFAR-100](https://www.kaggle.com/datasets/fedesoriano/cifar100).
   - **Size**: 50,000 training images, 10,000 testing images.
   - **Image Dimensions**: 32×32 pixels, 3 color channels (RGB).

3. **Model Architecture**:  
   - A Sequential CNN featuring:
     - **Conv2D** layers for feature extraction.
     - **MaxPooling** layers to reduce spatial dimensionality.
     - **Dropout** layers to prevent overfitting.
     - A **fully connected** layer with softmax activation for classification.

4. **Training**:
   - **Optimizer**: Adam  
   - **Loss Function**: Sparse Categorical Crossentropy  
   - **Metrics**: Accuracy  
   - **Typical Hyperparameters**:  
     - Epochs: 15  
     - Batch Size: 64  

5. **Results**:
   - **Test Accuracy**: ~43% (with a simple CNN architecture and 15 epochs).
   - **Loss**: ~2.24 on the test set.

## Getting Started

### Prerequisites
- Python 3.7+
- Packages: `tensorflow`, `numpy`, `matplotlib`

```bash
pip install tensorflow numpy matplotlib
