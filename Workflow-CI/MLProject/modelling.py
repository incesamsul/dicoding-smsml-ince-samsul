import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("SMSML CI Breast Cancer")

    data = load_breast_cancer(as_frame=True).frame
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = {"n_estimators": 160, "random_state": 42, "class_weight": "balanced"}
    model = RandomForestClassifier(**params)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_dict(report, "classification_report.json")

        Path("metrics.json").write_text(json.dumps({"accuracy": accuracy, "report": report}, indent=2))
        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")
        mlflow.log_artifact("metrics.json")

    print(classification_report(y_test, predictions))
