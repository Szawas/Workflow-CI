import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=["Outcome"])
    y_train = train_data["Outcome"]

    X_test = test_data.drop(columns=["Outcome"])
    y_test = test_data["Outcome"]

    return X_train, X_test, y_train, y_test


def main():
    train_path = os.path.join("diabetes_preprocessing", "train.csv")
    test_path = os.path.join("diabetes_preprocessing", "test.csv")

    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Karena dijalankan lewat mlflow run, run sudah dibuat otomatis.
    # Jadi cukup log metric, parameter, dan model tanpa mlflow.start_run().
    mlflow.set_tag("mlflow.runName", "RandomForest_CI_Training")

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 1)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print("CI model training selesai.")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    main()