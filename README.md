# Credit Card Fraud Detection Using TensorFlow/Keras

## Project Overview

This project uses a neural network to classify credit card transactions as either legitimate or fraudulent. The goal is to build a TensorFlow/Keras model that can identify fraudulent transactions from tabular transaction data.

## Problem Statement

Credit card fraud is a major issue for banks, credit card companies, and customers. Fraudulent transactions can result in financial losses and security concerns. This project aims to create a binary classification model that predicts whether a transaction is fraudulent or legitimate.

## Dataset

The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle.

Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains credit card transactions made by European cardholders. The target column is `Class`.

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

Most features are anonymized for privacy and labeled as `V1` through `V28`. The dataset also includes `Time` and `Amount`.

## Project Type

Option 3: Tabular Classification with a Dense Neural Network

## Tools and Libraries

- Python
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Planned Workflow

1. Load the dataset
2. Explore the data
3. Preprocess the features
4. Split data into training and testing sets
5. Build a Dense neural network
6. Train the model
7. Evaluate performance
8. Improve the model
9. Present results and limitations

## Model Plan

The project will use a Dense neural network because the dataset is tabular numerical data.

Planned architecture:

- Input layer
- Dense hidden layers with ReLU activation
- Dropout layer to reduce overfitting
- Sigmoid output layer for binary classification

## Evaluation Metrics

Because the dataset is highly imbalanced, accuracy alone is not enough. The model will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Improvement Attempt

At least one improvement attempt will be included, such as:

- Adding dropout
- Using class weights
- Changing the learning rate
- Adjusting the prediction threshold
- Comparing model architectures

## Team Responsibilities

### Partner 1: Data and Preprocessing Lead

- Dataset explanation
- Data exploration
- Class distribution analysis
- Preprocessing
- Visualizations

### Partner 2: Model and Evaluation Lead

- Neural network architecture
- Model training
- Evaluation metrics
- Confusion matrix
- Improvement attempt
