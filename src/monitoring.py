import logging
import os
import json
from datetime import datetime

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("IMDB_MLOps")

def log_prediction(review, sentiment, confidence):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "review_length": len(review),
        "sentiment": sentiment,
        "confidence": confidence
    }
    logger.info(f"PREDICTION: {json.dumps(log_entry)}")
    return log_entry

def log_model_metrics(model_name, accuracy, f1_score):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "accuracy": accuracy,
        "f1_score": f1_score
    }
    logger.info(f"MODEL_METRICS: {json.dumps(log_entry)}")

def log_data_drift(feature_name, expected_mean, actual_mean):
    drift = abs(expected_mean - actual_mean)
    status = "DRIFT_DETECTED" if drift > 0.1 else "STABLE"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "feature": feature_name,
        "expected_mean": expected_mean,
        "actual_mean": actual_mean,
        "drift": drift,
        "status": status
    }
    logger.warning(f"DATA_DRIFT: {json.dumps(log_entry)}")

if __name__ == "__main__":
    log_prediction("This movie was great!", "Positive", "High")
    log_model_metrics("LogisticRegression_TfidfVectorizer", 0.8843, 0.8843)
    log_data_drift("review_length", 150.0, 165.0)
    print("Monitoring logs created in logs/app.log")