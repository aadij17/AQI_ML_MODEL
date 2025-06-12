# Image-Based Estimation of Air Quality (AQI)

This project presents a machine learning approach to predict Air Quality Index (AQI) using sky images captured via mobile phones. It combines traditional feature-based techniques with deep learning to provide a scalable, low-cost alternative to conventional AQI monitoring systems.

## Overview

* Input: Sky images
* Output: Predicted AQI level
* Algorithms: Random Forest and Convolutional Neural Network (CNN)
* Best Accuracy: 86.14% using CNN
* Tools: Python, OpenCV, Scikit-learn, TensorFlow

## Problem Statement

Traditional AQI monitoring stations are limited by fixed locations and infrastructure costs. This project aims to bridge that gap using a visual estimation system based on widely available smartphone cameras.

## Methodology

### Data Collection

* Custom dataset of sky images from various times and locations
* Ground-truth AQI values from official monitoring stations

### Preprocessing

* Resizing, normalization, noise reduction
* Color space transformation for feature enhancement

### Feature Extraction

* Transmission Graph
* Sky Gradient
* Entropy of Blueness

### Model Development

* Random Forest trained on extracted features
* CNN trained directly on image pixels
* CNN achieved superior performance (86.14% accuracy)

### Evaluation

* Comparative analysis showed CNN outperformed Random Forest
* Feature importance identified sky gradient and transmission as key indicators

## Future Work

* Expand dataset with varied geographies and timeframes
* Integrate meteorological and sensor data
* Deploy model as a mobile or web application for public use

## Author
Aditya Jain

