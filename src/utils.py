import random

import numpy as np
import tensorflow as tf


def set_random_seed(seed=42):
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_model_results(results):
    """
    Print evaluation metrics in a readable format.
    """
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nClassification Report:")
    print(results["classification_report"])