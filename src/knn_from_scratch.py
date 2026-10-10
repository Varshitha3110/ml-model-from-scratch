"""A small K-Nearest Neighbors classifier implemented with NumPy.

This is a teaching example: it intentionally favors clarity over speed.
It uses the Iris dataset, scales features, and implements distance + voting.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def predict_one(X_train, y_train, x, k=5):
    # Euclidean distance from x to every training row
    distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))

    # Indices of the k closest training examples
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]

    # Majority vote: most common class among the nearest neighbors
    labels, counts = np.unique(nearest_labels, return_counts=True)
    return labels[np.argmax(counts)]


def predict_many(X_train, y_train, X_test, k=5):
    return np.array([predict_one(X_train, y_train, row, k) for row in X_test])


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler only on training data to avoid test-data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    predictions = predict_many(X_train, y_train, X_test, k=5)
    print("KNN-from-scratch test accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("First 10 actual labels:   ", y_test[:10])
    print("First 10 predicted labels:", predictions[:10])
    print("Class names:", list(iris.target_names))


if __name__ == "__main__":
    main()
