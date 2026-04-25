# 🏦 LoanGuard AI — Loan Default Predictor

An end-to-end machine learning system that predicts whether a borrower will default on a loan, with explainable AI powered risk assessment.

Built using real-world LendingClub data — 2.2 million loans from 2007 to 2018.

---

## 🎯 Problem Statement

Banks and lending platforms need to assess the risk of a borrower defaulting before approving a loan. This project builds a binary classification model that predicts default probability given a borrower's financial profile, and explains *why* each prediction was made using SHAP.

---

## 📊 Results

| Metric | Value |
|---|---|
| Model | LightGBM |
| AUC-ROC | 0.7159 |
| Training Rows | 854,088 |
| Features | 30 |
| Default Rate | 19.73% |

---

## 🗂️ Dataset

**Source:** LendingClub Loan Data (Kaggle)  
**Size:** 2.2M rows, 150+ columns  
**Years:** 2007–2018  
**Target:** loan_status → binary (Fully Paid = 0, Charged Off/Default = 1)

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

## 🏗️ Project Structure
```
loan-default-predictor/
├── 01_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_modeling.ipynb
├── 4_Plots.ipynb
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🔄 ML Pipeline
```
Raw Data (2.2M rows)
      ↓
Feature Selection (150+ → 15 columns)
      ↓
Cleaning (emp_length, int_rate, revol_util)
      ↓
Encoding (ordinal for grade, one-hot for categoricals)
      ↓
Train/Val/Test Split (70/15/15, stratified)
      ↓
Class Imbalance Handling (scale_pos_weight = 4.07)
      ↓
Baseline: Logistic Regression
      ↓
Main Model: LightGBM (AUC = 0.71)
      ↓
SHAP Explainability
      ↓
Streamlit Deployment
```

---

## 💡 Key Features

**Explainable Predictions**  
Every prediction comes with a SHAP waterfall plot showing exactly which features drove the decision — critical for real-world financial applications where decisions must be justifiable.

**Three-Tier Risk Classification**  
- 🟢 Low Risk — below 30% default probability  
- 🟡 Medium Risk — 30 to 50% default probability  
- 🔴 High Risk — above 50% default probability  

**Handles Class Imbalance**  
Dataset has 80/20 class split. Used scale_pos_weight of 4.07 so the model does not just predict not default for everything.

**MLflow Experiment Tracking**  
All experiments logged with parameters, metrics, and model artifacts for reproducibility.

---

## 🚀 How to Run

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
Get the LendingClub dataset from Kaggle and place the CSV in the project folder.

**4. Run notebooks in order**
```
01_eda.ipynb
02_feature_engineering.ipynb
03_modeling.ipynb
4_Plots.ipynb
```

**5. Launch the app**
```bash
streamlit run streamlit_app.py
```

---

## 📈 Model Details

**Why LightGBM?**  
Gradient boosting model that builds trees sequentially — each tree corrects mistakes of the previous one. Faster than XGBoost, better SHAP compatibility, and comparable performance on tabular data.

**Why AUC-ROC?**  
With 80/20 class imbalance, accuracy is misleading — a model predicting not default for everything gets 80% accuracy while being useless. AUC-ROC measures the model's ability to rank defaulters above non-defaulters regardless of threshold.

**Why SHAP?**  
In finance, explainability is a regulatory requirement. A model that cannot explain its decisions cannot be deployed in production. SHAP assigns each feature a contribution value for every individual prediction.

---

## 🔍 Key Findings

- Interest rate and loan grade are the strongest predictors of default
- FICO score and DTI are the strongest protective factors
- Borrowers taking loans for small business default more than those consolidating debt
- The model correctly ranks high risk borrowers above low risk borrowers 71% of the time

---

## 👨‍💻 Author

**Aditya** — 2nd Year BTech CS Student  
