# 10-K SEC Filing Risk Classifier

An end-to-end document intelligence system that classifies financial SEC filings as high, medium, or low risk.

## Project Structure
sec-classifier/
├── data/          # raw and processed datasets
├── src/
│   ├── preprocess.py   # data loading, cleaning, labeling
│   ├── features.py     # TF-IDF + custom feature engineering
│   ├── train.py        # model training (XGBoost, AdaBoost, CatBoost)
│   └── evaluate.py     # metrics and confusion matrix
├── api/
│   └── app.py          # FastAPI prediction endpoint
├── models/             # saved model files
└── requirements.txt
Model folder contains the pkl file for best model while data folder contains a representative sample dataset for demonstration. Full SEC data set was used for training but is omitted due to size constraints.


## Classification Setup
Labels are assigned based on economic era of filing date:
- **high_risk** → 2001-2002 (dot-com crash)
- **low_risk** → 1999-2000 (dot-com boom)
- **medium_risk** → everything else

## Features Used
- TF-IDF vectorization (5000 features)
- Custom features: document length, risk word count, positive word count, risk ratio, sentiment score, lexical density

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| XGBoost | 0.617 | 0.639 | 0.617 | 0.612 |
| AdaBoost | 0.498 | 0.544 | 0.498 | 0.485 |
| CatBoost | 0.590 | 0.620 | 0.590 | 0.579 |

**Best Model: XGBoost** with F1 score of 0.612

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```


### Run pipeline
``` bash
python src/preprocess.py
python src/features.py
python src/train.py
python src/evaluate.py
```

### Start API
```bash
uvicorn api.app:app --reload
```

### Test endpoint
http://localhost:8000/predict
{"text": "your financial text here"}

## API Response(example)
```json
{
  "label": "high_risk",
  "confidence": 0.37
}
```
