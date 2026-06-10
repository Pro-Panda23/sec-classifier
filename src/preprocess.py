from datasets import load_dataset
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

#  Load dataset 
print("Loading dataset...")
dataset = load_dataset("winterForestStump/10-K_sec_filings", verification_mode="no_checks")

keys = list(dataset.keys())[:10]
all_dfs = [pd.DataFrame(dataset[key]) for key in keys]
df = pd.concat(all_dfs, ignore_index=True)
print("Total rows:", len(df))

# Clean text 
def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Combine Business + MD&A for features 
text_cols = [
    'Business',
    "Management's Discussion and Analysis of Financial Condition and Results of Operations"
]

print("Combining and cleaning sections...")
combined = []
for col in text_cols:
    if col in df.columns:
        combined.append(df[col].astype(str))

df['clean_text'] = pd.concat(combined, axis=1).apply(
    lambda row: clean_text(' '.join(row.values)), axis=1
)

#  Assign labels based on economic era(Dotcom era crash/boom and everything outside of these) 

print("Assigning labels based on economic era...")

df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
df['filing_year'] = df['filing_date'].dt.year

def assign_era_label(year):
    if pd.isna(year):
        return 'medium_risk'
    year = int(year)
    if 2001 <= year <= 2002:
        return 'high_risk'
    elif 1999 <= year <= 2000:
        return 'low_risk'
    else:
        return 'medium_risk'

df['label'] = df['filing_year'].apply(assign_era_label)

print("\nLabel distribution:")
print(df['label'].value_counts())

# Save 
df_final = df[['cik', 'company_name', 'filing_date', 'clean_text', 'label']]
df_final.to_csv("data/labeled_filings.csv", index=False)
print("\nDone! Saved to data/labeled_filings.csv")