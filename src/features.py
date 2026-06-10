import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import joblib

print("Loading data...")
df = pd.read_csv("data/labeled_filings.csv")


df = df[df['clean_text'].notna() & (df['clean_text'].str.strip() != '')]
print("Rows after cleaning:", len(df))

#  TF-IDF Vectorization 
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
print("TF-IDF shape:", X_tfidf.shape)

#  Custom Features 

print("Creating custom features...")

high_risk_words = ['risk', 'loss', 'decline', 'uncertainty', 'litigation',
                   'default', 'bankruptcy', 'adverse', 'volatile', 'lawsuit',
                   'failure', 'unable', 'difficult', 'negative', 'weak']
low_risk_words  = ['growth', 'profit', 'strong', 'improved', 'success',
                   'revenue', 'positive', 'stable', 'opportunity', 'increase',
                   'expand', 'leading', 'innovative', 'strength', 'advantage']

def extract_custom_features(text):
    if not isinstance(text, str):
        text = ""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) + 1

    risk_count     = sum(text_lower.count(w) for w in high_risk_words)
    positive_count = sum(text_lower.count(w) for w in low_risk_words)
    total_sentiment = risk_count + positive_count + 1

    return {
        'doc_length'      : total_words,                        # how long is the doc
        'risk_word_count' : risk_count,                         # raw risk word count
        'pos_word_count'  : positive_count,                     # raw positive word count
        'risk_ratio'      : risk_count / total_sentiment,       # % of sentiment that is risky
        'sentiment_score' : (positive_count - risk_count) / total_sentiment,  # -1 to +1 score
        'lexical_density' : len(set(words)) / total_words,      # vocabulary richness
    }

custom_features = df['clean_text'].apply(extract_custom_features)
X_custom = pd.DataFrame(custom_features.tolist()).values
print("Custom features shape:", X_custom.shape)
print("Custom feature names: doc_length, risk_word_count, pos_word_count, risk_ratio, sentiment_score, lexical_density")

#  Combine TF-IDF + Custom Features 

X_combined = hstack([X_tfidf, csr_matrix(X_custom)])
print("\nFinal feature matrix shape:", X_combined.shape)

#  Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])
print("Labels:", le.classes_)

#  Save everything 
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(le,         "models/label_encoder.pkl")
joblib.dump(X_combined, "models/X_features.pkl")
joblib.dump(y,          "models/y_labels.pkl")

print("\nDone! All saved to models./")