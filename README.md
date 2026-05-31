# Face Recognition Using PCA and ANN

## Overview
This project implements a Face Recognition System using Principal Component Analysis (PCA) and an Artificial Neural Network (ANN). PCA is used to extract eigenface features, and ANN is used for classification.

## Technologies Used
- Python
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib

## Features
- Face dataset preprocessing
- PCA-based feature extraction
- ANN-based face classification
- Accuracy comparison for different values of k
- Imposter detection for unknown faces

## Results
The system was trained using 60% of the dataset and tested on 40% of the dataset. Accuracy was evaluated for multiple PCA feature dimensions (k values), and the results were plotted in accuracy_vs_k.png.

## How to Run

Training:
python3 train.py

Testing:
python3 test.py