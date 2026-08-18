"""
train_models.py
----------------
Trains and evaluates 5 classification models on the Breast Cancer Wisconsin
(Diagnostic) dataset:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Gaussian Naive Bayes Classifier
  5. Random Forest Classifier (Ensemble)

For each model it computes: Accuracy, AUC, Precision, Recall, F1, MCC.
Saves each trained model as a .pkl file (for the Streamlit app to load) and
saves the held-out test split as test_data.csv (features + true label column
`target`) so the Streamlit app has real data to demo predictions on.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
REPO_ROOT = os.path.dirname(BASE_DIR)

RANDOM_STATE = 7

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class distribution:\n{y.value_counts()}\n")

# ---------------------------------------------------------------------
# 2. Train/test split (stratified)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save test split as CSV for the Streamlit app / repo deliverable
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(REPO_ROOT, "test_data.csv"), index=False)
print(f"Saved test_data.csv with {test_df.shape[0]} rows.\n")

# ---------------------------------------------------------------------
# 3. Define models (each wrapped with scaling where it matters)
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": Pipeline([
        ("clf", GaussianNB())
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ]),
}

# ---------------------------------------------------------------------
# 4. Train, evaluate, save
# ---------------------------------------------------------------------
results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    # Probability of positive class for AUC
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipe.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results[name] = metrics

    # Save model
    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))

    print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

# ---------------------------------------------------------------------
# 5. Save results table (used by app.py and README)
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
results_df = results_df.round(4)
results_df.to_csv(os.path.join(MODEL_DIR, "results.csv"))

with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nComparison table:\n")
print(results_df)
