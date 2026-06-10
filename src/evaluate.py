import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from catboost import CatBoostClassifier

# Load Test Data
print("Loading test data...")
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")
le     = joblib.load("models/label_encoder.pkl")

#  Load models....
xgb = joblib.load("models/xgboost_model.pkl")
ada = joblib.load("models/adaboost_model.pkl")
cat = CatBoostClassifier()
cat.load_model("models/catboost_model.cbm")

models = {
    "XGBoost" : (xgb, False),   # False = don't need .toarray()
    "AdaBoost": (ada, True),    # True  = needs .toarray()
    "CatBoost": (cat, True),
}

#  Evaluate each model 
results = {}
for name, (model, needs_dense) in models.items():
    X_input = X_test.toarray() if needs_dense else X_test
    preds   = model.predict(X_input)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec  = recall_score(y_test, preds, average='weighted')
    f1   = f1_score(y_test, preds, average='weighted')
    cm   = confusion_matrix(y_test, preds)

    results[name] = {'Accuracy': acc, 'Precision': prec, 
                     'Recall': rec, 'F1': f1}

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  Classes: {le.classes_}")
    print(cm)
    print(f"\n  Detailed Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

# Create a summary table
print("\n" + "="*50)
print("  FINAL COMPARISON TABLE")
print("="*50)
print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*50)
for name, m in results.items():
    print(f"{name:<12} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f}")

# The best model
best = max(results, key=lambda x: results[x]['F1'])
print(f"\n Best Model: {best} with F1 Score of {results[best]['F1']:.4f}")