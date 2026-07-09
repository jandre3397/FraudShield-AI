# Credit Card Fraud Detection with Deep Learning

## Overview

Credit card fraud is a growing challenge for financial institutions worldwide, resulting in billions of dollars in losses each year. As digital payment systems continue to expand, organizations require intelligent systems capable of identifying potentially fraudulent transactions quickly and accurately while minimizing disruptions to legitimate customers.

This project develops a deep learning solution using TensorFlow and Keras to classify credit card transactions as either legitimate or fraudulent. By leveraging a feedforward neural network trained on historical transaction data, the model learns complex patterns associated with fraudulent activity and provides real-time fraud risk predictions.

The primary objective is to build a robust binary classification model that maximizes fraud detection while minimizing false positives, helping financial institutions reduce financial losses and improve customer trust.

---

## Business Problem

Traditional rule-based fraud detection systems struggle to identify evolving fraud patterns and often generate large numbers of false alarms. Fraudulent transactions represent only a very small percentage of all credit card activity, making fraud detection a highly imbalanced classification problem.

Financial institutions need intelligent models capable of:

- Detecting fraudulent transactions before they are approved
- Reducing financial losses caused by fraud
- Minimizing false positives that inconvenience legitimate customers
- Adapting to changing fraud patterns through machine learning

This project explores how deep learning can improve fraud detection by learning complex relationships within transaction data.

---

## Solution

This project implements a supervised deep learning model using TensorFlow and Keras.

The model analyzes historical credit card transaction data and predicts whether a new transaction is:

- Legitimate
- Fraudulent

The project follows a complete machine learning workflow:

1. Data Exploration
2. Data Cleaning and Preprocessing
3. Feature Scaling
4. Model Development
5. Model Training
6. Performance Evaluation
7. Model Optimization
8. Error Analysis

---

## Dataset

The model is trained using the publicly available **Credit Card Fraud Detection** dataset published on Kaggle.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset characteristics:

- 284,807 credit card transactions
- 492 fraudulent transactions
- 30 numerical features
- Binary classification target

To protect customer privacy, the majority of features have been transformed using Principal Component Analysis (PCA). Only the **Time** and **Amount** features remain in their original form.

---

## Technologies

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Machine Learning Pipeline

The project follows a production-style workflow:

### Data Collection

Load and inspect transaction data.

### Exploratory Data Analysis

- Analyze class imbalance
- Visualize fraud distribution
- Explore feature relationships

### Data Preprocessing

- Scale numerical features
- Split training and testing data
- Handle severe class imbalance

### Model Development

Build a feedforward neural network using TensorFlow/Keras.

The baseline architecture includes:

- Dense input layer
- Hidden Dense layers
- Dropout regularization
- Sigmoid output layer

---

## Model Evaluation

The model will be evaluated using metrics appropriate for highly imbalanced datasets.

Metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Special emphasis is placed on **Recall**, since missing fraudulent transactions can result in significant financial loss.

---

## Model Improvement

Several optimization strategies will be explored, including:

- Hyperparameter tuning
- Additional hidden layers
- Dropout regularization
- Class weighting
- Learning rate optimization
- Decision threshold adjustment

Performance improvements will be compared against the baseline model.

---

## Expected Outcome

The completed model will be capable of predicting whether a transaction is likely fraudulent based on historical transaction patterns.

Although this project is intended for educational purposes, it mirrors the workflow used by machine learning engineers and data scientists developing fraud detection systems within the financial industry.

---

## Repository Structure

```text
credit-card-fraud-detection/

├── data/
├── notebooks/
├── src/
├── models/
├── images/
├── presentation/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Future Enhancements

Future versions of this project may include:

- Real-time prediction API using FastAPI
- Docker containerization
- Cloud deployment (AWS or Azure)
- Automated model retraining pipeline
- Explainable AI using SHAP
- Interactive dashboard with Streamlit

---

## Contributors

- Jaël Andre
- Amaya Washington
