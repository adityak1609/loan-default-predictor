# 🏦 LoanGuard AI — Loan Default Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![AUC--ROC](https://img.shields.io/badge/AUC--ROC-0.716-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning system that predicts whether a borrower will default on a loan, with explainable AI-powered risk assessment. Built on real-world LendingClub data — **2.2 million loans across 11 years (2007–2018)**.

> 💡 Every prediction includes a SHAP waterfall plot explaining *why* the model made its decision — critical for real-world financial applications where decisions must be justifiable.

---

## 📊 Results

| Metric | Value |
|---|---|
| Model | LightGBM |
| AUC-ROC | 0.716 |
| Training Rows | 1,700,000 |
| Features | 30 |
| Default Rate | 19.73% |
| Class Imbalance Handling | scale_pos_weight = 4.07 |

---

## 🎯 Problem Statement

Banks and lending platforms need to assess borrower risk before approving a loan. This project builds a binary classification model that predicts default probability from a borrower's financial profile — and explains *why* each prediction was made using SHAP values.

**Target variable:** `loan_status` → binary (Fully Paid = 0, Charged Off/Default = 1)

---

## 🔄 ML Pipeline

```
Raw Data (2.2M rows, 150+ columns)
        ↓
Feature Selection (150+ → 30 columns)
        ↓
Cleaning (emp_length, int_rate, revol_util)
        ↓
Encoding (ordinal for grade, one-hot for categoricals)
        ↓
Train / Val / Test Split (70 / 15 / 15, stratified)
        ↓
Class Imbalance Handling (scale_pos_weight = 4.07)
        ↓
Baseline: Logistic Regression
        ↓
Main Model: LightGBM (AUC-ROC = 0.716)
        ↓
SHAP Explainability (global + local)
        ↓
Streamlit Deployment + MLflow Tracking
```

---

## 💡 Key Features

### 🔍 Explainable Predictions
Every prediction comes with a SHAP waterfall plot showing exactly which features drove the decision. In finance, explainability isn't optional — it's a regulatory requirement.

### 🚦 Three-Tier Risk Classification
| Risk Level | Default Probability |
|---|---|
| 🟢 Low Risk | Below 30% |
| 🟡 Medium Risk | 30% – 50% |
| 🔴 High Risk | Above 50% |

### ⚖️ Class Imbalance Handling
The dataset has an 80/20 class split. Using `scale_pos_weight = 4.07` prevents the model from simply predicting "no default" for everything and achieving a misleading 80% accuracy.

### 📦 MLflow Experiment Tracking
All experiments are logged with parameters, metrics, and model artifacts — fully reproducible.

---

## 🔍 Key Findings

- **Interest rate** and **loan grade** are the strongest predictors of default
- **FICO score** and **debt-to-income ratio (DTI)** are the strongest protective factors
- Borrowers taking loans for **small business** default more frequently than those consolidating debt
- The model correctly ranks high-risk borrowers above low-risk borrowers **71.6% of the time** (AUC-ROC)

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Pandas, NumPy |
| Modeling | LightGBM, Scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Web App | Streamlit |
| Environment | Conda, Python 3.10 |

---

## 🗂️ Project Structure

```
loan-default-predictor/
├── 01_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_modeling.ipynb
├── 04_plots.ipynb
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/adityak1609/loan-default-predictor.git
cd loan-default-predictor
```

**2. Create conda environment**
```bash
conda create -n loanproject python=3.10
conda activate loanproject
pip install -r requirements.txt
```

**3. Download the dataset**

Get the [LendingClub dataset from Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place the CSV in the project root.

**4. Run notebooks in order**
```
01_eda.ipynb
02_feature_engineering.ipynb
03_modeling.ipynb
04_plots.ipynb
```

**5. Launch the Streamlit app**
```bash
streamlit run streamlit_app.py
```

---

## 📈 Why These Choices?

**Why LightGBM?**
Gradient boosting that builds trees sequentially — each correcting the previous one's mistakes. Faster than XGBoost on large datasets, excellent SHAP compatibility, and strong performance on tabular financial data.

**Why AUC-ROC over Accuracy?**
With an 80/20 class split, accuracy is misleading — a model predicting "no default" for everything scores 80% while being completely useless. AUC-ROC measures the model's ability to rank defaulters above non-defaulters regardless of threshold.

**Why SHAP?**
Black-box models can't be deployed in regulated financial environments. SHAP assigns each feature a contribution value for every individual prediction, making the model auditable and trustworthy.

---

## 👨‍💻 Author

**Aditya Khanna** — 2nd Year B.Tech IT, Manipal Institute of Technology

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/aditya-khanna-564896269)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/adityak1609) 
