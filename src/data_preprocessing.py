import pandas as pd
import numpy as np
import re
import yaml
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def text_preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word, pos="v") for word in words]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_data():
    params = load_params()
    raw_path = params["data"]["raw_path"]
    processed_path = params["data"]["processed_path"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    print("Loading dataset...")
    df = pd.read_csv(raw_path)
    df = df.drop_duplicates()
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    print("Preprocessing text...")
    df["cleaned_review"] = df["review"].apply(text_preprocess)

    os.makedirs(processed_path, exist_ok=True)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_df.to_csv(os.path.join(processed_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_path, "test.csv"), index=False)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data()