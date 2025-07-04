import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Loan Default Predictor",   
    page_icon="💰",                       
    layout="wide",                        
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <h1 >
         Loan Default Risk Predictor
    </h1>
    <hr style='border: 1px solid #ccc; margin-top: -10px; margin-bottom: 30px;'>
    """,
    unsafe_allow_html=True
)

st.markdown("""
This tool uses a machine learning model trained on borrower data to predict the risk of loan default.
""")

st.subheader ("""
Enter the information and click **Predict** to see the result.
""")

# Load model
model = joblib.load("app/model.pkl")

# User Inputs 
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Monthly Income (₹)", min_value=0)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=2.0, step=0.01)

with col2:
    education = st.selectbox("Education", ["Bachelor's", "High School", "Master's", "PhD"])
    employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    months_employed = st.number_input("Months Employed", min_value=0)
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
    has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
    has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    has_cosigner = st.selectbox("Has Co-signer?", ["No", "Yes"])

# --- One-hot encoding (manual) ---

# Binary
has_mortgage = 1 if has_mortgage == "Yes" else 0
has_dependents = 1 if has_dependents == "Yes" else 0
has_cosigner = 1 if has_cosigner == "Yes" else 0

# Education (4 columns)
edu_bach = int(education == "Bachelor's")
edu_hs = int(education == "High School")
edu_masters = int(education == "Master's")
edu_phd = int(education == "PhD")

# Employment (4 columns)
emp_full = int(employment == "Full-time")
emp_part = int(employment == "Part-time")
emp_self = int(employment == "Self-employed")
emp_unemp = int(employment == "Unemployed")

# Marital Status (3 columns)
mar_div = int(marital_status == "Divorced")
mar_married = int(marital_status == "Married")
mar_single = int(marital_status == "Single")

# Loan Purpose (5 columns)
purpose_auto = int(loan_purpose == "Auto")
purpose_business = int(loan_purpose == "Business")
purpose_education = int(loan_purpose == "Education")
purpose_home = int(loan_purpose == "Home")
purpose_other = int(loan_purpose == "Other")

# --- Final feature vector (28 features) ---
input_data = np.array([[
    age, income, loan_amount, credit_score, months_employed,
    num_credit_lines, interest_rate, loan_term, dti_ratio, has_mortgage,
    has_dependents, has_cosigner,
    edu_bach, edu_hs, edu_masters, edu_phd,
    emp_full, emp_part, emp_self, emp_unemp,
    mar_div, mar_married, mar_single,
    purpose_auto, purpose_business, purpose_education, purpose_home, purpose_other
]])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    result = "🟢 Low Risk of Default" if prediction == 0 else "🔴 High Risk of Default"
    st.subheader(f"Prediction: {result}")
    st.info(f"🔍 Model Confidence (Default Risk): **{proba * 100:.2f}%**")

# Footer
st.markdown("---")
st.markdown("Built by [Nishu Mehta](https://github.com/NishuMehta) | Powered by Streamlit 💻")
