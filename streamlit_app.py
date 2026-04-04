import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="LoanGuard AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Hero Header */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #0a0a0f 50%, #0d1117 100%);
    border: 1px solid #1e1e2e;
    border-radius: 20px;
    padding: 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 12px 0;
    line-height: 1.1;
}
.hero-sub {
    color: #6b7280;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* Section Headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4b5563;
    margin: 0 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e2e;
}

/* Input Cards */
.input-card {
    background: #0d0d1a;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Result Cards */
.result-card {
    background: #0d0d1a;
    border: 1px solid #1e1e2e;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 20px;
}
.result-card.high {
    border-color: rgba(239, 68, 68, 0.4);
    background: linear-gradient(135deg, #0d0d1a, #1a0d0d);
}
.result-card.medium {
    border-color: rgba(245, 158, 11, 0.4);
    background: linear-gradient(135deg, #0d0d1a, #1a150d);
}
.result-card.low {
    border-color: rgba(16, 185, 129, 0.4);
    background: linear-gradient(135deg, #0d0d1a, #0d1a12);
}

.prob-display {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}
.prob-display.high { color: #ef4444; }
.prob-display.medium { color: #f59e0b; }
.prob-display.low { color: #10b981; }

.risk-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 8px;
}
.risk-label.high { color: #ef4444; }
.risk-label.medium { color: #f59e0b; }
.risk-label.low { color: #10b981; }

.risk-desc {
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 8px;
    line-height: 1.5;
}

/* Progress Bar */
.prob-bar-container {
    background: #1e1e2e;
    border-radius: 100px;
    height: 6px;
    margin: 20px 0;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s ease;
}
.prob-bar.high { background: linear-gradient(90deg, #ef4444, #dc2626); }
.prob-bar.medium { background: linear-gradient(90deg, #f59e0b, #d97706); }
.prob-bar.low { background: linear-gradient(90deg, #10b981, #059669); }

/* Stat Pills */
.stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 20px;
}
.stat-pill {
    background: #161622;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 10px 16px;
    flex: 1;
    min-width: 100px;
}
.stat-pill-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4b5563;
    margin-bottom: 4px;
}
.stat-pill-value {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e8e8f0;
}

/* Predict Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.5) !important;
}

/* Streamlit element overrides */
.stSelectbox > div > div {
    background: #0d0d1a !important;
    border-color: #1e1e2e !important;
    color: #e8e8f0 !important;
}
.stSlider > div > div > div {
    background: #6366f1 !important;
}
label { color: #9ca3af !important; font-size: 0.85rem !important; }
.stNumberInput > div > div > input {
    background: #0d0d1a !important;
    border-color: #1e1e2e !important;
    color: #e8e8f0 !important;
}

/* SHAP section */
.shap-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8e8f0;
    margin-bottom: 4px;
}
.shap-sub {
    color: #4b5563;
    font-size: 0.8rem;
    margin-bottom: 20px;
}

/* Divider */
hr { border-color: #1e1e2e !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('processed/lgbm_model.pkl')

model = load_model()

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Risk Assessment</div>
    <p class="hero-title">LoanGuard AI</p>
    <p class="hero-sub">Predict loan default probability with explainable machine learning.<br>
    Trained on 2.2M real LendingClub loans (2007–2018).</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<p class="section-header">Borrower Profile</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        loan_amnt  = st.number_input("Loan Amount ($)", 1000, 40000, 10000, step=500)
        annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
        fico       = st.slider("FICO Credit Score", 580, 850, 700)
        emp_length = st.selectbox("Employment (years)", [0,1,2,3,4,5,6,7,8,9,10])
        open_acc   = st.number_input("Open Accounts", 1, 40, 10)
        mort_acc   = st.number_input("Mortgage Accounts", 0, 20, 1)

    with c2:
        int_rate   = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, step=0.1)
        dti        = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0, step=0.5)
        revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)
        revol_bal  = st.number_input("Revolving Balance ($)", 0, 100000, 15000, step=500)
        pub_rec    = st.number_input("Public Records", 0, 10, 0)
        grade      = st.selectbox("Loan Grade", ['A','B','C','D','E','F','G'])

    home_ownership = st.selectbox("Home Ownership", ['RENT','OWN','MORTGAGE','OTHER'])
    purpose = st.selectbox("Loan Purpose", [
        'debt_consolidation','credit_card','home_improvement','other',
        'major_purchase','small_business','car','medical','moving',
        'vacation','house','wedding','renewable_energy','educational'])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyze Default Risk →", type="primary")

# ── Right Panel ──────────────────────────────────────────────
with right:
    st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style="
            background: #0d0d1a;
            border: 1px dashed #1e1e2e;
            border-radius: 20px;
            padding: 60px 32px;
            text-align: center;
        ">
            <div style="font-size: 3rem; margin-bottom: 16px;">🏦</div>
            <p style="font-family: 'Syne', sans-serif; font-size: 1.1rem;
               font-weight: 600; color: #374151; margin: 0 0 8px 0;">
               Awaiting Analysis
            </p>
            <p style="color: #374151; font-size: 0.8rem; margin: 0;">
               Fill in borrower details and click analyze
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Build input
        grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
        data = {
            'loan_amnt': loan_amnt, 'int_rate': int_rate,
            'annual_inc': annual_inc, 'dti': dti,
            'grade': grade_map[grade], 'emp_length': emp_length,
            'fico_range_low': fico, 'open_acc': open_acc,
            'revol_util': revol_util, 'revol_bal': revol_bal,
            'mort_acc': mort_acc, 'pub_rec': pub_rec,
        }
        for val in ['MORTGAGE','OTHER','OWN','RENT']:
            data[f'home_ownership_{val}'] = 1 if home_ownership == val else 0
        for val in ['car','credit_card','debt_consolidation','educational',
                    'home_improvement','house','major_purchase','medical',
                    'moving','other','renewable_energy','small_business',
                    'vacation','wedding']:
            data[f'purpose_{val}'] = 1 if purpose == val else 0

        input_df = pd.DataFrame([data])
        model_cols = model.booster_.feature_name()
        for col in model_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_cols]

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        pct  = prob * 100

        if prob > 0.5:
            tier, desc = "high", "High probability of default. This loan carries significant risk."
        elif prob > 0.3:
            tier, desc = "medium", "Moderate default risk. Review carefully before approval."
        else:
            tier, desc = "low", "Low default risk. Borrower profile looks financially stable."

        # Result Card
        st.markdown(f"""
        <div class="result-card {tier}">
            <p class="prob-display {tier}">{pct:.1f}%</p>
            <p class="risk-label {tier}">
                {"🔴 HIGH RISK" if tier=="high" else "🟡 MEDIUM RISK" if tier=="medium" else "🟢 LOW RISK"}
            </p>
            <p class="risk-desc">{desc}</p>
            <div class="prob-bar-container">
                <div class="prob-bar {tier}" style="width: {pct}%"></div>
            </div>
            <div class="stat-row">
                <div class="stat-pill">
                    <div class="stat-pill-label">Default Prob</div>
                    <div class="stat-pill-value">{pct:.1f}%</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Loan Grade</div>
                    <div class="stat-pill-value">{grade}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">FICO Score</div>
                    <div class="stat-pill-value">{fico}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Int. Rate</div>
                    <div class="stat-pill-value">{int_rate}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP
        st.markdown("""
        <div class="shap-header">Why this prediction?</div>
        <div class="shap-sub">SHAP values showing feature contributions to this result</div>
        """, unsafe_allow_html=True)

        with st.spinner("Computing explanations..."):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            sv = shap_values[0] if isinstance(shap_values, list) else shap_values

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0d0d1a')
            ax.set_facecolor('#0d0d1a')

            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value[0],
                sv[0],
                input_df.iloc[0],
                show=False
            )

            plt.gcf().set_facecolor('#0d0d1a')
            for text in plt.gcf().findobj(matplotlib.text.Text):
                text.set_color('#9ca3af')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e1e2e')
            ax.tick_params(colors='#6b7280')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()