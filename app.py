import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("credit_risk_model.pkl")

st.title("Credit Risk Prediction App")

st.write("Enter applicant details below:")

# User Inputs
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
person_emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
loan_grade = st.selectbox("Loan Grade (1=A, 7=G)", [1,2,3,4,5,6,7])
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.2)
cb_person_default_on_file = st.selectbox("Previous Default?", [0,1])
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)

# Dummy values for encoded columns (simplified assumption)
person_home_ownership_OTHER = 0
person_home_ownership_OWN = 0
person_home_ownership_RENT = 1  # defaulting to RENT

loan_intent_EDUCATION = 0
loan_intent_HOMEIMPROVEMENT = 0
loan_intent_MEDICAL = 0
loan_intent_PERSONAL = 1  # default
loan_intent_VENTURE = 0

# Create DataFrame
input_data = pd.DataFrame([[
    person_age,
    person_income,
    person_emp_length,
    loan_grade,
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    cb_person_default_on_file,
    cb_person_cred_hist_length,
    person_home_ownership_OTHER,
    person_home_ownership_OWN,
    person_home_ownership_RENT,
    loan_intent_EDUCATION,
    loan_intent_HOMEIMPROVEMENT,
    loan_intent_MEDICAL,
    loan_intent_PERSONAL,
    loan_intent_VENTURE
]], columns=[
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE"
])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Risk of Default (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Default (Probability: {probability:.2f})")