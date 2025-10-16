import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

# Fallback model setup
@st.cache_resource
def get_fallback_model():
    model = LogisticRegression()
    X_dummy = [[0]*6, [1]*6]
    y_dummy = [0, 1]
    model.fit(X_dummy, y_dummy)
    return {"Fallback_Model": model}

# Load test data
@st.cache_data
def load_data():
    path = Path("models/test_set.csv")
    return pd.read_csv(path) if path.exists() else None

# Sidebar — logo
with st.sidebar:
    logo_path = Path("assets/banner.png")
    if logo_path.exists():
        st.image(Image.open(logo_path), width=250)
    else:
        st.markdown("### Fraud Model Explainer")

# Load model
models_dict = get_fallback_model()
model_name = st.sidebar.selectbox("Choose a model", list(models_dict.keys()))
model = models_dict[model_name]

# Features
features = [
    "Transaction_Amount", "Account_Balance", "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count", "Risk_Score", "Is_Weekend"
]

# Sidebar — inputs
st.sidebar.header("Input Features")
inputs = {
    "Transaction_Amount": st.sidebar.number_input("Transaction Amount", value=0.0),
    "Account_Balance": st.sidebar.number_input("Account Balance", value=0.0),
    "Previous_Fraudulent_Activity": int(st.sidebar.checkbox("Previous Fraudulent Activity")),
    "Daily_Transaction_Count": st.sidebar.number_input("Daily Transaction Count", value=0),
    "Risk_Score": st.sidebar.slider("Risk Score", 0.0, 1.0, 0.5),
    "Is_Weekend": int(st.sidebar.checkbox("Is Weekend"))
}
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# Build input row
X_user = pd.DataFrame([inputs], columns=features).fillna(0.0)

# Display inputs
st.subheader("User Inputs")
st.dataframe(X_user)

# Explain Risk_Score
with st.expander("What is Risk_Score?"):
    st.markdown("""
    **Risk_Score** reflects how suspicious a transaction looks based on:
    - Velocity checks
    - Device/IP anomalies
    - Rule-based flags
    - Historical behavior
    """)

# Predict
proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"

st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# Evaluation
df = load_data()
if df is not None and "is_fraud" in df.columns:
    X_test = df[features]
    y_test = df["is_fraud"]
    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2%}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2%}")
