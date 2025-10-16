import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

@st.cache_data
def load_data():
    path = Path("models/test_set.csv")
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource
def load_models(models_dir="models"):
    models = {}
    for p in Path(models_dir).glob("*.pkl"):
        name = p.stem
        try:
            models[name] = joblib.load(p)
        except:
            st.warning(f"Could not load model: {p.name}")
    return models

with st.sidebar:
    logo_path = Path("assets/banner.png")
    if logo_path.exists():
        st.image(Image.open(logo_path), width=250)
    else:
        st.markdown("### Fraud Model Explainer")

models_dict = load_models()
if not models_dict:
    st.error("No models found in /models. Please add at least one .pkl file.")
    st.stop()

model_name = st.sidebar.selectbox("Choose a model", list(models_dict.keys()))
model = models_dict.get(model_name)
if model is None:
    st.error(f"Selected model '{model_name}' not found.")
    st.stop()

default_features = [
    "Transaction_Amount", "Account_Balance", "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count", "Risk_Score", "Is_Weekend"
]

st.sidebar.header("Input Features")
selected_features = {
    "Transaction_Amount": st.sidebar.number_input("Transaction Amount", value=0.0),
    "Account_Balance": st.sidebar.number_input("Account Balance", value=0.0),
    "Previous_Fraudulent_Activity": int(st.sidebar.checkbox("Previous Fraudulent Activity")),
    "Daily_Transaction_Count": st.sidebar.number_input("Daily Transaction Count", value=0),
    "Risk_Score": st.sidebar.slider("Risk Score", 0.0, 1.0, 0.5),
    "Is_Weekend": int(st.sidebar.checkbox("Is Weekend"))
}
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

X_user = pd.DataFrame([{
    feat: float(val) for feat, val in selected_features.items()
}], columns=default_features).fillna(0.0)

st.subheader("User Inputs")
st.dataframe(X_user[selected_features.keys()])

with st.expander("What is Risk_Score?"):
    st.markdown("""
    **Risk_Score** reflects how suspicious a transaction looks based on:
    - Velocity checks
    - Device/IP anomalies
    - Rule-based flags
    - Historical behavior
    """)

proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"

st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

df = load_data()
if df is not None and "is_fraud" in df.columns:
    X_test = df[default_features]
    y_test = df["is_fraud"]
    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2%}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2%}")

