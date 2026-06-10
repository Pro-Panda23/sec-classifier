from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = FastAPI(
    title="10-K SEC Filing Risk Classifier",
    description="Classifies financial filings as high/low/medium risk",
    version="1.0.0"
)

# Load model and preprocessors 
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
le         = joblib.load("models/label_encoder.pkl")
model      = joblib.load("models/xgboost_model.pkl")

#  Input/Output schemas 
class InputText(BaseModel):
    text: str

class PredictionResult(BaseModel):
    label: str
    confidence: float

# Helper functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def extract_custom_features(text):
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) + 1
    high_risk_words = ['risk', 'loss', 'decline', 'uncertainty', 'litigation',
                       'default', 'bankruptcy', 'adverse', 'volatile', 'lawsuit',
                       'failure', 'unable', 'difficult', 'negative', 'weak']
    low_risk_words  = ['growth', 'profit', 'strong', 'improved', 'success',
                       'revenue', 'positive', 'stable', 'opportunity', 'increase',
                       'expand', 'leading', 'innovative', 'strength', 'advantage']
    risk_count     = sum(text_lower.count(w) for w in high_risk_words)
    positive_count = sum(text_lower.count(w) for w in low_risk_words)
    total_sentiment = risk_count + positive_count + 1
    return [[
        total_words,
        risk_count,
        positive_count,
        risk_count / total_sentiment,
        (positive_count - risk_count) / total_sentiment,
        len(set(words)) / total_words
    ]]

#  Endpoints
@app.get("/")
def root():
    return {"message": "10-K SEC Filing Risk Classifier API", "docs": "/docs"}

@app.post("/predict", response_model=PredictionResult)
def predict(input: InputText):
    cleaned      = clean_text(input.text)
    tfidf_feats  = vectorizer.transform([cleaned])
    custom_feats = csr_matrix(extract_custom_features(cleaned))
    features     = hstack([tfidf_feats, custom_feats])
    
    pred       = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    label      = le.inverse_transform([int(pred)])[0]
    confidence = round(float(np.max(proba)), 2)
    
    return {"label": label, "confidence": confidence}