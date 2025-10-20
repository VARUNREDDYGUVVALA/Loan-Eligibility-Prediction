import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# 1️⃣ Locate model files
# -------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(base_dir), "model", "best_model.pkl")
scaler_path = os.path.join(os.path.dirname(base_dir), "model", "scaler.pkl")

# -------------------------------
# 2️⃣ Load model and scaler
# -------------------------------
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("✅ Model and Scaler loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model/scaler: {e}")
    st.stop()

# -------------------------------
# 3️⃣ Page Configuration
# -------------------------------
st.set_page_config(page_title="Loan Eligibility Prediction", layout="centered")

# -------------------------------
# 4️⃣ Add Background Image
# -------------------------------
bg_image_path = r"D:\Drive\projects\Loan-Eligibility-Prediction\dataset-cover.jpg"

st.markdown(f"""
    <style>
    .stApp {{
        background: url("file:///{bg_image_path.replace("\\", "/")}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.3); /* dark overlay */
        z-index: 0;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    .form-box {{
        background-color: rgba(255, 255, 255, 0.92);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        max-width: 800px;
        margin: auto;
        margin-top: 50px;
    }}
    h1, h3 {{
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 5px #000;
    }}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 5️⃣ Sidebar Information
# -------------------------------
st.sidebar.header("📘 About the App")
st.sidebar.info("""
This app predicts whether a loan applicant is eligible for a loan 
based on their financial and demographic details.
""")
st.sidebar.markdown("**Model Used:** Gradient Boosting")
st.sidebar.markdown("**Test Accuracy:** 73.5%")
st.sidebar.markdown("**F1 Score:** 0.82")
st.sidebar.markdown("📂 [View GitHub Repo](https://github.com/yourusername/Loan-Eligibility-Prediction)")

# -------------------------------
# 6️⃣ App Title
# -------------------------------
st.markdown("<h1>🏦 Loan Eligibility Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Fill out your details below to check your loan eligibility:</h3>", unsafe_allow_html=True)

# -------------------------------
# 7️⃣ Input Form
# -------------------------------
with st.container():
    with st.form("loan_form", clear_on_submit=False):
        st.markdown('<div class="form-box">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", [0, 1, 2, 3])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_emp = st.selectbox("Self Employed", ["Yes", "No"])

        with col2:
            loan_amt = st.number_input("Loan Amount", min_value=0.0)
            loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=480, value=360)
            credit_history = st.selectbox("Credit History", ["Good", "Bad"])
            property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
            income = st.number_input("Total Income (Applicant + Coapplicant)", min_value=0.0)

        submitted = st.form_submit_button("🔮 Predict Loan Eligibility")

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 8️⃣ Prediction
# -------------------------------
if submitted:
    data = pd.DataFrame({
        "Gender": [1 if gender == "Male" else 0],
        "Married": [1 if married == "Yes" else 0],
        "Dependents": [dependents],
        "Education": [1 if education == "Graduate" else 0],
        "Self_Employed": [1 if self_emp == "Yes" else 0],
        "LoanAmount": [loan_amt],
        "Loan_Amount_Term": [loan_term / 12],
        "Credit_History": [1 if credit_history == "Good" else 0],
        "Property_Area": [0 if property_area == "Rural" else 1 if property_area == "Semiurban" else 2],
        "Income": [income],
    })

    try:
        scaled = scaler.transform(data)
        pred = model.predict(scaled)[0]

        if pred == 1:
            st.markdown("""
                <div style='background-color:#C8E6C9; color:#2E7D32; padding:15px; 
                            border-radius:10px; text-align:center; font-size:22px;
                            box-shadow:0 4px 10px rgba(0,0,0,0.2);'>
                    ✅ Congratulations! You are eligible for a loan.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color:#FFCDD2; color:#C62828; padding:15px;
                            border-radius:10px; text-align:center; font-size:22px;
                            box-shadow:0 4px 10px rgba(0,0,0,0.2);'>
                    ❌ Sorry, you are not eligible for a loan.
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -------------------------------
# 9️⃣ Footer
# -------------------------------
st.markdown("""
    <hr>
    <div style='text-align:center; color:white; font-size:16px;'>
        Developed by <b>Varun Reddy</b> • 
        <a href='https://github.com/yourusername/Loan-Eligibility-Prediction' target='_blank' style='color:#FFD700;'>GitHub Repository</a>
    </div>
""", unsafe_allow_html=True)
