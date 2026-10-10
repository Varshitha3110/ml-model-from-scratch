"""End-to-end Iris classification demo for the workshop."""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # 1. Load data
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    print("Dataset shape:", X.shape)
    print("\nFirst five rows:")
    print(X.head())
    print("\nFeatures:", list(X.columns))
    print("Classes:", list(iris.target_names))

    # 2. Split before fitting preprocessing/model steps
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    # 3. Build and train a pipeline
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
    model.fit(X_train, y_train)

    # 4. Predict and evaluate
    predictions = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, target_names=iris.target_names
    ))

    # 5. Try one sample (measurements in cm)
    sample = pd.DataFrame([{
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }])
    predicted_class = model.predict(sample)[0]
    print("Sample measurements:", sample.iloc[0].to_dict())
    print("Predicted species:", iris.target_names[predicted_class])


if __name__ == "__main__":
    main()
