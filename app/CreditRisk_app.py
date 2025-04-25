# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Paths
model_path = "model"
outputs_path = "outputs"

# Load assets
model = joblib.load(f"model/best_model_XGBoost.pkl")
scaler = joblib.load(f"model/scaler.pkl")
features = joblib.load(f"model/features.pkl")  # Full feature list
selected_features = joblib.load(f"model/selected_features.pkl")  # Used during training
label_encoders = joblib.load(f"model/label_encoders.pkl")  # Consistent label encoding

# App title
st.title("Credit Risk Prediction App")
st.header("Enter Applicant Details")


def user_input():
    age = st.number_input("Age", min_value=18, max_value=100)
    sex = st.selectbox("Sex", label_encoders['Sex'].classes_)
    job = st.selectbox("Job", [0, 1, 2, 3])
    housing = st.selectbox("Housing", label_encoders['Housing'].classes_)
    saving_accounts = st.selectbox("Saving accounts", label_encoders['Saving accounts'].classes_)
    checking_account = st.selectbox("Checking account", label_encoders['Checking account'].classes_)
    credit_amount = st.number_input("Credit amount (in DM)", min_value=0.0)
    duration = st.number_input("Duration (in months)", min_value=1)
    purpose = st.selectbox("Purpose", label_encoders['Purpose'].classes_)

    data = {
        "Age": age,
        "Sex": label_encoders["Sex"].transform([sex])[0],
        "Job": job,
        "Housing": label_encoders["Housing"].transform([housing])[0],
        "Saving accounts": label_encoders["Saving accounts"].transform([saving_accounts])[0],
        "Checking account": label_encoders["Checking account"].transform([checking_account])[0],
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": label_encoders["Purpose"].transform([purpose])[0]
    }

    df = pd.DataFrame([data])

    # Ensure all features are present and ordered
    for col in features:
        if col not in df:
            df[col] = 0
    df = df[features]

    return df


input_df = user_input()

if st.button("Predict"):
    # Filter to selected features used in model
    input_selected = input_df[selected_features]

    # Scale input
    input_scaled = scaler.transform(input_selected)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    st.subheader("Prediction")
    st.success("Good Credit Risk" if prediction == 1 else "Bad Credit Risk")

    st.subheader("Confidence")
    st.write(f"Probability of good credit: {probability:.2f}")

# Visuals
st.header("Model Insights")

if os.path.exists(f"outputs/roc_auc_comparison.png"):
    st.subheader("ROC AUC Comparison")
    st.image(f"outputs/roc_auc_comparison.png")

if os.path.exists(f"outputs/mutual_information.png"):
    st.subheader("Mutual Information Scores")
    st.image(f"outputs/mutual_information.png")

if os.path.exists(f"outputs/correlation_matrix.png"):
    st.subheader("Feature Correlation Heatmap")
    st.image(f"outputs/correlation_matrix.png")
