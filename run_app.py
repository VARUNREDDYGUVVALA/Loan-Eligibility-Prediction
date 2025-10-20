# run_app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ✅ Set relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "loan_eligibility_status_DATA_SET.csv")
BG_IMAGE = os.path.join(BASE_DIR, "dataset-cover.jpg")

# 🎨 Background image (optional)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("file://{BG_IMAGE}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Loan Eligibility Prediction")

# 📝 Load model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# 📝 Example UI
st.header("Enter Applicant Details:")
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])
gender = st.selectbox("Gender", ["Male", "Female"])

# Predict button
if st.button("Predict"):
    # Prepare input for model
    import numpy as np
    X = np.array([[applicant_income, coapplicant_income, loan_amount, loan_term, credit_history]])
    prediction = model.predict(X)[0]
    st.success(f"Loan Eligibility: {'Approved' if prediction == 1 else 'Rejected'}")
