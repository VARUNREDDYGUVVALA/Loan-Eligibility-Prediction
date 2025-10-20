# 🏦 Loan Eligibility Prediction

This project is a **machine learning web application** that predicts whether an applicant is eligible for a loan based on their details.  
It uses a trained classification model integrated into a **Streamlit** web interface for easy interaction.

---

## 🚀 Features
- Interactive and user-friendly UI built with Streamlit  
- Machine learning model trained using Scikit-learn  
- Automatic input preprocessing using saved scaler  
- Dynamic result prediction (Eligible / Not Eligible)  
- Attractive layout with a custom background image  

---

## 🧰 Tech Stack
- **Python 3.x**
- **Streamlit**
- **Scikit-learn**
- **Pandas**, **NumPy**
- **Pickle** (for model persistence)

---

## 📂 Project Structure
Loan-Eligibility-Prediction/
│
├── app/
│ └── loan_ui_app.py # Streamlit UI file
│
├── model/
│ ├── train_model.py # Script to train ML model
│ ├── best_model.pkl # Trained model
│ └── scaler.pkl # Scaler for input normalization
│
├── data/
│ └── loan_eligibility_status_DATA_SET.csv # Dataset
│
├── notebooks/
│ └── LOAN ELIGIBILITY STATUS PROJECT.ipynb # Model training notebook
│
├── run_app.py # Entry point to run the app
├── dataset-cover.jpg # Background image for UI
├── requirements.txt
└── README.md

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/VARUNREDDYGUVVALA/Loan-Eligibility-Prediction.git
cd Loan-Eligibility-Prediction




pip install -r requirements.txt

streamlit run app/loan_ui_app.py
The model uses classification techniques (e.g., Logistic Regression, Random Forest, etc.) trained on applicant demographic and financial details to predict loan approval outcomes.
It preprocesses features using a saved scaler and outputs the prediction instantly through the UI.


