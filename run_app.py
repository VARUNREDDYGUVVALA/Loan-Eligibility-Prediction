# run_app.py
import os
import sys
import subprocess

# ✅ Define relative path to Streamlit app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(BASE_DIR, "app", "loan_ui_app.py")

# 🚨 Check if file exists
if not os.path.exists(APP_PATH):
    print(f"❌ Can't find app file at: {APP_PATH}")
    print("💡 Make sure your project folder structure looks like this:")
    print("""
    Loan-Eligibility-Prediction/
    ├── app/
    │   └── loan_ui_app.py
    ├── model/
    │   └── train_model.py
    ├── data/
    │   └── loan_eligibility_status_DATA_SET.csv
    └── run_app.py
    """)
    sys.exit(1)

# 🚀 Launch Streamlit from Python
print("\n🚀 Launching Streamlit app...")
try:
    subprocess.run(["streamlit", "run", APP_PATH], check=True)
except KeyboardInterrupt:
    print("\n🛑 App stopped manually.")
except Exception as e:
    print(f"❌ Error running Streamlit app: {e}")
