# ML Assignment 2 — Classification Models with Streamlit Deployment

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple machine
learning classification models on a real-world dataset, and to expose the
results through an interactive Streamlit web application. Specifically, the
task is a **binary classification problem**: predicting whether a breast mass
is **malignant** or **benign** based on quantitative measurements derived from
a digitized image of a fine needle aspirate (FNA) of the mass.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set
**Source:** UCI Machine Learning Repository / scikit-learn built-in dataset
(`sklearn.datasets.load_breast_cancer`), originally from the UCI ML Repository.

- **Instances:** 569
- **Features:** 30 numeric features (mean, standard error, and "worst"
  value of 10 real-valued measurements per cell nucleus — e.g. radius,
  texture, perimeter, area, smoothness, compactness, concavity, concave
  points, symmetry, fractal dimension)
- **Target variable:** Diagnosis — binary (0 = malignant, 1 = benign)
- **Class balance:** 212 malignant / 357 benign
- **Train/test split:** 80% / 20%, stratified by class, `random_state=42`

This dataset satisfies the assignment's minimum requirements (≥12 features,
≥500 instances) and is a standard binary classification benchmark.

## c. GitHub Repository Link

`<PASTE YOUR GITHUB REPO LINK HERE AFTER YOU PUSH>`

## d. Models Used

All 5 models were trained on the same 80/20 stratified train/test split of the
dataset described above. Numeric feature scaling (`StandardScaler`) was
applied for the distance/gradient-based models (Logistic Regression, kNN).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9974 | 0.9857 | 0.9583 | 0.9718 | 0.9260 |
| Decision Tree | 0.9737 | 0.9742 | 0.9859 | 0.9722 | 0.9790 | 0.9439 |
| kNN | 0.9737 | 0.9950 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9211 | 0.9821 | 0.9437 | 0.9306 | 0.9371 | 0.8313 |
| Random Forest (Ensemble) | 0.9561 | 0.9960 | 0.9855 | 0.9444 | 0.9645 | 0.9085 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong on ranking quality (highest AUC of all models, 0.9974) and precision (0.9857), but has the lowest recall (0.9583) of the top models — it misses slightly more true malignant cases than kNN or Decision Tree do on this split. |
| Decision Tree | Surprisingly strong on this split — ties for the highest accuracy (0.9737) and posts a high MCC (0.9439), showing that a single tree can perform competitively when the split happens to align well with its axis-aligned decision boundaries, though it typically generalizes less reliably than ensemble methods across different splits. |
| kNN | Best overall balance: ties for highest accuracy (0.9737), highest recall (a perfect 1.0000 — no malignant cases missed), highest F1 (0.9796), and highest MCC (0.9442). Distance-based methods like kNN benefit directly from the feature scaling applied here. |
| Naive Bayes | Weakest performer across all metrics (Accuracy 0.9211, MCC 0.8313). Its independence assumption between the 30 correlated cell-nucleus measurements is a poor fit for this dataset, since many of the features (e.g. radius, perimeter, area) are highly correlated with each other. |
| Random Forest (Ensemble) | Achieves the highest AUC of all models (0.9960) and strong precision (0.9855), but its recall (0.9444) is the lowest among the top 3 models — it's the most conservative/confident classifier but misses more true positives than kNN or Decision Tree here. |
| **Overall Winner for your dataset?** | **kNN** — it's the only model with a perfect recall (critical in a medical diagnosis context, where missing a malignant case is costlier than a false alarm), while also tying for the best accuracy and posting the highest F1 and MCC of all 5 models. |

## Repository Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
    │-- train_models.py
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest.pkl
    │-- results.csv
    │-- results.json
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live Streamlit App Link

`<PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE>`
