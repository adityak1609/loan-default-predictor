# Loan Default Predictor

End-to-end ML system to predict loan defaults using real LendingClub data.

## Results
- Dataset: 2.2M loans (2007–2018)
- Model: LightGBM
- AUC-ROC: 0.71 on test set
- Features: 30 (after encoding)

## Tech Stack
Python, LightGBM, SHAP, Scikit-learn, Pandas, Streamlit

## How to Run
pip install -r requirements.txt
streamlit run streamlit_app.py
