import joblib
import yaml
import os
import re
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

app = Flask(__name__)

def text_preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w, pos="v") for w in words]
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

model = joblib.load("models/LogisticRegression_TfidfVectorizer.pkl")
vectorizer = joblib.load("models/TfidfVectorizer.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "IMDB Sentiment Analysis API",
        "endpoints": {
            "/predict": "POST - Predict sentiment of a movie review"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review = data.get("review", "")
        if not review:
            return jsonify({"error": "No review provided"}), 400
        cleaned = text_preprocess(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return jsonify({
            "review": review,
            "sentiment": sentiment,
            "confidence": "High"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)