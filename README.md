# 25070149022 - MLOps Mini Project
## IMDB Sentiment Analysis - End-to-End MLOps Pipeline

**Student:** Shoeb Shakil Sutar
**Roll No:** 25070149022
**Course:** Essentials of MLOps
**Faculty:** Mr. Shridhar Shende

---

## 📌 Problem Definition
This project implements a complete MLOps pipeline for IMDB Movie Review Sentiment Analysis. The primary focus is on the MLOps infrastructure rather than model performance. Traditional ML models are trained, tracked, versioned, and deployed following industry-standard MLOps practices.

> Note: A Custom Transformer Encoder is implemented from scratch for research comparison but is not deployed due to computational constraints.

---

## 🏗️ Architecture

Raw Data (IMDB CSV - 50,000 reviews) → Data Versioning (DVC → AWS S3) → Data Preprocessing (NLTK) → Model Training → Experiment Tracking (MLflow) → Best Model Selected (Logistic Regression + TF-IDF → 88.43%) → Flask REST API → Docker Container → CI/CD (GitHub Actions) → Auto Deploy → AWS EC2 Deployment → Monitoring & Logging

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| DVC | Data versioning |
| AWS S3 | Remote data storage |
| MLflow | Experiment tracking |
| GitHub Actions | CI/CD automation |
| Docker | Containerization |
| AWS EC2 | Cloud deployment |
| Flask | REST API |
| Scikit-learn | Traditional ML models |
| PyTorch | Custom Transformer |
| NLTK | Text preprocessing |

---

## 📊 Model Results

| Model | Vectorizer | Accuracy |
|---|---|---|
| **Logistic Regression** | **TF-IDF** | **88.43%** ✅ |
| Linear SVC | TF-IDF | 87.90% |
| Logistic Regression | CountVec | 86.68% |
| Linear SVC | CountVec | 85.77% |
| Multinomial NB | CountVec | 84.62% |
| Random Forest | TF-IDF | 83.95% |
| Random Forest | CountVec | 83.91% |
| Custom Transformer | BERT Tokenizer | 85.79% |

---

## 🚀 How to Run

### 1. Clone the repository
git clone https://github.com/sutarshoeb/25070149022_MLOps_Project.git
cd 25070149022_MLOps_Project

### 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Pull data from DVC
dvc pull

### 5. Run preprocessing
python3 src/data_preprocessing.py

### 6. Train traditional models
python3 src/train_traditional.py

### 7. View MLflow experiments
mlflow ui
Open http://127.0.0.1:5000

### 8. Run Flask API locally
python3 app.py

### 9. Test API
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"review": "This movie was amazing!"}'

---

## 🐳 Docker

### Build image
docker build -t imdb-sentiment-api .

### Run container
docker run -p 5000:5000 imdb-sentiment-api

---

## ⚙️ CI/CD Pipeline

Every push to main branch automatically:
1. Sets up Python 3.10
2. Installs dependencies
3. Tests preprocessing module
4. Tests Flask import
5. Checks project structure
6. Deploys to AWS EC2 if all tests pass

---

## ☁️ Live API

The API is deployed on AWS EC2 and accessible at:
http://3.109.243.234:5000

### Endpoints:

| Endpoint | Method | Description |
|---|---|---|
| / | GET | API information |
| /predict | POST | Predict sentiment |
| /health | GET | Health check |

### Sample Request:
POST http://3.109.243.234:5000/predict
{"review": "This movie was absolutely amazing!"}

### Sample Response:
{"review": "This movie was absolutely amazing!", "sentiment": "Positive", "confidence": "High"}

---

## 📁 Project Structure

25070149022_MLOps_Project/
├── .github/workflows/ci_cd.yml
├── .dvc/config
├── data/raw/
├── data/processed/
├── models/
├── src/data_preprocessing.py
├── src/train_traditional.py
├── src/train_transformer.py
├── src/evaluate.py
├── src/monitoring.py
├── logs/
├── app.py
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
└── requirements.txt

---

## 📝 Monitoring

Every prediction is logged with timestamp, review length, predicted sentiment and confidence level. Logs are stored in logs/app.log

---

## 🔗 GitHub Repository
https://github.com/sutarshoeb/25070149022_MLOps_Project
