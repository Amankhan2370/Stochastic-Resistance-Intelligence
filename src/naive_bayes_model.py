"""
Naïve Bayes classifier for predicting Not_MDR status from Ampicillin and Penicillin features.
Uses Gaussian Naïve Bayes from scikit-learn.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the antimicrobial resistance dataset from CSV.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with Ampicillin, Penicillin, and Not_MDR columns.
    """
    df = pd.read_csv(filepath)
    return df


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Train a Gaussian Naïve Bayes classifier and evaluate on test set.

    Args:
        df: DataFrame with Ampicillin, Penicillin, Not_MDR columns.

    Returns:
        Dictionary with training_size, testing_size, accuracy, and model.
    """
    # Define features (X) and target (y)
    X = df[["Ampicillin", "Penicillin"]]
    y = df["Not_MDR"]

    # Split dataset: 75% training, 25% testing with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train Gaussian Naïve Bayes classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "training_size": len(X_train),
        "testing_size": len(X_test),
        "accuracy": accuracy,
        "model": model,
    }


def run_naive_bayes(data_path: str = "data/amr_ds.csv") -> dict:
    """
    Main entry point: load data, train model, and print results.

    Args:
        data_path: Relative path to dataset from project root.

    Returns:
        Dictionary with results for use by main.py.
    """
    # Resolve path relative to project root
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(script_dir, data_path)

    df = load_dataset(full_path)
    results = train_and_evaluate(df)

    print("\n" + "=" * 60)
    print("PART 1: Naïve Bayes Classification")
    print("=" * 60)
    print(f"Training size: {results['training_size']}")
    print(f"Testing size:  {results['testing_size']}")
    print(f"Accuracy:      {results['accuracy']:.4f}")
    print()

    return results


if __name__ == "__main__":
    run_naive_bayes()
