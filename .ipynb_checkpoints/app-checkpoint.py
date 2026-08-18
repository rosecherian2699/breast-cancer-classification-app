"""
app.py
------
Streamlit app for demoing 5 classification models trained on the
Breast Cancer Wisconsin (Diagnostic) dataset.

Features:
  - CSV upload (test data)
  - Model selection dropdown
  - Evaluation metrics display
  - Confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

TARGET_COL = "target"


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


st.title("🔬 Breast Cancer Classification — Model Demo")
st.markdown(
    "This app demonstrates **5 classification models** trained on the "
    "**Breast Cancer Wisconsin (Diagnostic)** dataset "
    "(569 instances, 30 features, binary classification: malignant vs benign)."
)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help="Upload a CSV with the same feature columns as the training data, "
         "plus a 'target' column with true labels (0 = malignant, 1 = benign)."
)

model_choice = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {df.shape[0]} rows from uploaded file.")
else:
    default_path = os.path.join(BASE_DIR, "test_data.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.sidebar.info(f"No file uploaded — using bundled test_data.csv ({df.shape[0]} rows).")
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

st.subheader("Preview of Test Data")
st.dataframe(df.head(10))

if TARGET_COL not in df.columns:
    st.error(f"The uploaded CSV must contain a '{TARGET_COL}' column with true labels.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL]

# ---------------------------------------------------------------------
# Load model and predict
# ---------------------------------------------------------------------
model = load_model(MODEL_FILES[model_choice])

try:
    y_pred = model.predict(X)
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.decision_function(X)
except Exception as e:
    st.error(f"Error running predictions — check that uploaded CSV columns match "
              f"the training features. Details: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------
st.subheader(f"Evaluation Metrics — {model_choice}")

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_proba),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

# ---------------------------------------------------------------------
# Confusion matrix + classification report
# ---------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with c2:
    st.markdown("**Classification Report**")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

# ---------------------------------------------------------------------
# Compare all models (if results.csv is available)
# ---------------------------------------------------------------------
results_path = os.path.join(MODEL_DIR, "results.csv")
if os.path.exists(results_path):
    st.subheader("All Models — Comparison (on original held-out test split)")
    results_df = pd.read_csv(results_path)
    st.dataframe(results_df)
