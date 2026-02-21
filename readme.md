# Student Dropout Prediction Model

A binary classification machine learning project that predicts whether a student is at risk of dropping out, using academic and socio-economic features. Built with Python and Scikit-learn, achieving **94.7% accuracy** and **0.98 ROC-AUC**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Key Concepts](#key-concepts)

---

## Overview

Student dropout is a major challenge in higher education. Early identification of at-risk students allows institutions to intervene before it's too late. This project builds an end-to-end ML pipeline that:

- Handles real-world class imbalance using **SMOTE**
- Tunes hyperparameters automatically using **GridSearchCV**
- Evaluates performance with **Confusion Matrix**, **ROC-AUC**, and **F1 Score**
- Explains predictions using **Permutation Importance** and **SHAP-style analysis**

---

## Dataset

**Source:** [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

| Property | Value |
|---|---|
| Total records | 4,424 students |
| Features | 35 (academic + socio-economic) |
| Target classes | Graduate (0) vs Dropout (1) |
| Class imbalance | ~3:1 (Graduate : Dropout) |

**Key features include:**
- Curricular units approved / grades (1st & 2nd semester)
- Age at enrollment
- Tuition fees status, scholarship, debtor flag
- Previous qualification grade, admission grade
- Macroeconomic indicators (GDP, unemployment rate, inflation)

---

## Project Structure

```
dropout_project/
│
├── run_all.py                   # ▶ Run everything with one command
│
├── step1_load_explore.py        # Load data + exploratory analysis
├── step2_preprocessing.py       # Imputation, scaling, SMOTE
├── step3_train_model.py         # Random Forest + GridSearchCV tuning
├── step4_evaluate.py            # Metrics, confusion matrix, ROC curve
├── step5_feature_importance.py  # Feature importance + individual explanations
│
├── requirements.txt             # All dependencies
│
├── data/                        # Auto-created on first run
│   ├── cleaned_data.csv
│   ├── X_train.npy / X_test.npy
│   ├── y_train.npy / y_test.npy
│   └── feature_names.csv
│
├── models/                      # Auto-created on first run
│   ├── best_random_forest.pkl
│   ├── preprocessor.pkl
│   └── eval_metrics.json
│
└── plots/                       # Auto-created on first run
    ├── step1_eda.png
    ├── step2_smote.png
    ├── step3_gridsearch.png
    ├── step4_evaluation.png
    └── step5_feature_importance.png
```

---

## Pipeline

```
Raw Data
   │
   ▼
Step 1 — EDA          →  Understand distribution, spot imbalance
   │
   ▼
Step 2 — Preprocessing →  Impute → Scale → Train/Test Split → SMOTE
   │
   ▼
Step 3 — Training      →  Random Forest + GridSearchCV (5-fold CV)
   │
   ▼
Step 4 — Evaluation    →  Accuracy / F1 / ROC-AUC / Confusion Matrix
   │
   ▼
Step 5 — Explainability →  Feature Importance + Individual Explanation
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **94.7%** |
| ROC-AUC | **0.98** |
| Dropout Precision | 96% |
| Dropout Recall | 82% |
| Dropout F1-Score | 89% |

**Best Hyperparameters found by GridSearchCV:**
```
n_estimators    : 100
max_depth       : 15
min_samples_leaf: 1
```

**Top dropout risk factors (by feature importance):**
1. Curricular units without evaluations (1st sem)
2. Curricular units approved (2nd sem)
3. Curricular units grade (2nd sem)
4. Age at enrollment
5. Tuition fees up to date

---

## How to Run

### 1. Clone / download the project
```bash
git clone https://github.com/YOUR-USERNAME/student-dropout-prediction.git
cd student-dropout-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python run_all.py
```

This runs all 5 steps in order and saves plots to the `plots/` folder.

### 4. Or run steps individually
```bash
python step1_load_explore.py
python step2_preprocessing.py
python step3_train_model.py
python step4_evaluate.py
python step5_feature_importance.py
```

> **Note:** Steps must be run in order — each step saves outputs that the next step reads.

### 5. Use the real UCI dataset (recommended)
By default the project uses synthetic data with the same structure as the real dataset. To use the real data, replace the data generation block in `step1_load_explore.py` with:
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=697)
X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)
df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
df["Target"] = df["Target"].map({"Dropout": 1, "Graduate": 0})
```

### 6. Enable full SHAP values (optional)
In `step5_feature_importance.py`, replace the permutation section with:
```python
import shap
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(pd.DataFrame(X_test, columns=feature_names))
shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns=feature_names))
```

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11      # for real SMOTE
shap>=0.44                  # for full SHAP values
matplotlib>=3.7
seaborn>=0.12
ucimlrepo>=0.0.3            # to download real dataset
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Key Concepts

**SMOTE** (Synthetic Minority Over-sampling Technique)
Fixes class imbalance by generating synthetic minority-class samples. It picks a minority sample, finds its nearest neighbours, and interpolates new points between them — not just duplicates.

**Random Forest**
An ensemble of decision trees. Each tree is trained on a random data subset and sees random features at each split. Final prediction = majority vote across all trees. Reduces overfitting compared to a single decision tree.

**GridSearchCV**
Exhaustive hyperparameter search with cross-validation. Tries every combination in a grid, evaluates each with k-fold CV, returns the best-performing settings.

**ROC-AUC**
Area Under the Receiver Operating Characteristic curve. Measures how well the model separates classes across all classification thresholds. 1.0 = perfect, 0.5 = random guessing.

**Permutation Importance**
Measures feature importance by shuffling one feature at a time and recording the accuracy drop. A large drop means the feature was important. More reliable than built-in impurity importance.

---

## Author

**Deep Kumar Singh**  
B.Tech, Materials and Metallurgical Engineering  
Maulana Azad National Institute of Technology, Bhopal  
📧 deepsingh00492@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/deep-kumar-singh/) · [GitHub](https://github.com/DEEP2909)
