import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_traditional():
    params = load_params()
    processed_path = params["data"]["processed_path"]
    max_features = params["data"]["max_features"]
    experiment_name = params["mlflow"]["experiment_name"]

    mlflow.set_experiment(experiment_name)

    train_df = pd.read_csv(os.path.join(processed_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_path, "test.csv"))

    X_train = train_df["cleaned_review"]
    y_train = train_df["sentiment"]
    X_test = test_df["cleaned_review"]
    y_test = test_df["sentiment"]

    vectorizers = {
        "CountVectorizer": CountVectorizer(max_features=max_features),
        "TfidfVectorizer": TfidfVectorizer(max_features=max_features)
    }

    models = {
        "LogisticRegression": LogisticRegression(max_iter=5000),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(),
        "RandomForest": RandomForestClassifier()
    }

    os.makedirs("models", exist_ok=True)

    for vec_name, vectorizer in vectorizers.items():
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        for model_name, model in models.items():
            if model_name == "MultinomialNB" and vec_name == "TfidfVectorizer":
                continue

            run_name = f"{model_name}_{vec_name}"
            print(f"\nTraining {run_name}...")

            with mlflow.start_run(run_name=run_name):
                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                mlflow.log_param("model", model_name)
                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("max_features", max_features)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.sklearn.log_model(model, run_name)

                print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

                joblib.dump(model, f"models/{run_name}.pkl")
                joblib.dump(vectorizer, f"models/{vec_name}.pkl")

    print("\nAll traditional models trained and logged to MLflow!")

if __name__ == "__main__":
    train_traditional()