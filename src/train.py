import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
import time

# Load features 
print("Loading features...")
X = joblib.load("models/X_features.pkl")
y = joblib.load("models/y_labels.pkl")

#  Train/Test Split
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Save test set for evaluate.py
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")

#  XGBoost
print("\nTraining XGBoost...")
start = time.time()
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_train, y_train)
print(f"XGBoost done in {round(time.time()-start, 1)}s")
joblib.dump(xgb, "models/xgboost_model.pkl")

#  AdaBoost 
print("\nTraining AdaBoost...")
start = time.time()
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X_train.toarray(), y_train)
print(f"AdaBoost done in {round(time.time()-start, 1)}s")
joblib.dump(ada, "models/adaboost_model.pkl")

#  CatBoost
print("\nTraining CatBoost...")
start = time.time()
cat = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=0,
    random_state=42
)
cat.fit(X_train.toarray(), y_train)
print(f"CatBoost done in {round(time.time()-start, 1)}s")
cat.save_model("models/catboost_model.cbm")

print("\nAll 3 models trained and saved!")