# model/train_model.py

import pandas as pd
import numpy as np
import joblib, os, warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# 1️⃣ Load dataset
data_path = r"D:\Drive\projects\Loan-Eligibility-Prediction\data\loan_eligibility_status_DATA_SET.csv"
data = pd.read_csv(data_path)

# 2️⃣ Preprocessing
data.drop(columns=['Loan_ID'], inplace=True, errors='ignore')
data['Dependents'] = data['Dependents'].replace({'3+': 3}).astype(float)

# Fill missing values
for col in ['Gender', 'Married', 'Self_Employed', 'Credit_History']:
    data[col] = data[col].fillna(data[col].mode()[0])
for col in ['LoanAmount', 'Loan_Amount_Term', 'Dependents']:
    data[col] = data[col].fillna(data[col].median())

# Encode categorical
encode = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Loan_Status': {'Y': 1, 'N': 0}
}
for col in encode:
    if col in data.columns:
        data[col] = data[col].map(encode[col])

# 3️⃣ Feature Engineering (New + Improved)
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['LoanAmount_log'] = np.log1p(data['LoanAmount'])
data['EMI'] = data['LoanAmount'] / (data['Loan_Amount_Term'] + 1)
data['Balance_Income'] = data['Total_Income'] - (data['EMI'] * 1000)
data['Credit_Income_Ratio'] = data['Credit_History'] * data['Total_Income']

# 🔥 New engineered features
data['Income_to_Loan'] = data['Total_Income'] / (data['LoanAmount'] + 1)
data['EMI_to_Income'] = data['EMI'] / (data['Total_Income'] + 1)
data['Credit_to_Loan'] = data['Credit_History'] * data['LoanAmount']

data.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True, errors='ignore')
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# 4️⃣ Split dataset
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Handle imbalance
sm = SMOTE(sampling_strategy=0.8, random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# 7️⃣ Base Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

print("🔍 Training and Evaluating Models...\n")
results = []
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    results.append((name, acc, f1, cv))
    print(f"{name:<20} | Test Accuracy: {acc:.4f} | F1: {f1:.4f} | CV: {cv:.4f}")

results_df = pd.DataFrame(results, columns=["Model", "Test Accuracy", "F1 Score", "CV Accuracy"])
print("\n📊 Summary of Models:\n", results_df)

# 8️⃣ Fine-Tune Gradient Boosting
print("\n⚙️ Fine-Tuning Gradient Boosting for better accuracy...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train_res, y_train_res)

best_gb = grid.best_estimator_
print("✅ Best Gradient Boosting Params:", grid.best_params_)

y_pred_gb = best_gb.predict(X_test_scaled)
acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
print(f"🎯 Tuned Gradient Boosting -> Accuracy: {acc_gb:.4f}, F1: {f1_gb:.4f}")

# 9️⃣ Stacked Ensemble
print("\n🤖 Training Stacked Ensemble...")
stack_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', best_gb),
        ('cat', CatBoostClassifier(verbose=0, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=500),
    n_jobs=-1
)

stack_model.fit(X_train_res, y_train_res)
y_pred_stack = stack_model.predict(X_test_scaled)
acc_stack = accuracy_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack)
print(f"🏗️ Stacked Ensemble -> Accuracy: {acc_stack:.4f}, F1: {f1_stack:.4f}")

# 🔟 Pick Best Model
final_results = {
    "Gradient Boosting Tuned": acc_gb,
    "Stacked Ensemble": acc_stack
}
best_name = max(final_results, key=final_results.get)
best_model = best_gb if best_name == "Gradient Boosting Tuned" else stack_model

print(f"\n🏆 Final Best Model: {best_name}")

# 🔹 Save model & scaler
model_dir = r"D:\Drive\projects\Loan-Eligibility-Prediction\model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

print(f"\n✅ Saved best model ({best_name}) to: {model_dir}")
print("\n✅ Training completed successfully!")
