# run_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ───── Paths ─────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")  # updated
DATASET_PATH = os.path.join(BASE_DIR, "data", "loan_eligibility_status_DATA_SET.csv")
BG_IMAGE = os.path.join(BASE_DIR, "dataset-cover.jpg")

# ───── Background Image ─────
if os.path.exists(BG_IMAGE):
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

# ───── Load Model ─────
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error("Model file not found! Make sure 'best_model.pkl' is in the 'model/' folder.")
        st.stop()

model = load_model(MODEL_PATH)

# ───── Streamlit UI ─────
st.header("Enter Applicant Details:")

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])
gender = st.selectbox("Gender", ["Male", "Female"])

# ───── Predict Button ─────
if st.button("Predict"):
    X = np.array([[applicant_income, coapplicant_income, loan_amount, loan_term, credit_history]])
    try:
        prediction = model.predict(X)[0]
        st.success(f"Loan Eligibility: {'Approved' if prediction == 1 else 'Rejected'}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
