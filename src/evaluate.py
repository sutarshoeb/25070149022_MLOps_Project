import pandas as pd
import numpy as np
import joblib
import torch
import yaml
import os
import mlflow
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_traditional():
    params = load_params()
    processed_path = params["data"]["processed_path"]

    test_df = pd.read_csv(os.path.join(processed_path, "test.csv"))
    X_test = test_df["cleaned_review"]
    y_test = test_df["sentiment"]

    results = []
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl") and "Vectorizer" not in f]

    for model_file in model_files:
        model_name = model_file.replace(".pkl", "")
        model = joblib.load(f"models/{model_file}")

        if "Tfidf" in model_name:
            vectorizer = joblib.load("models/TfidfVectorizer.pkl")
        else:
            vectorizer = joblib.load("models/CountVectorizer.pkl")

        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "f1_score": f1
        })

        print(f"\n{model_name}:")
        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df)
    results_df.to_csv("models/evaluation_results.csv", index=False)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    evaluate_traditional()