import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Setup
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

# Load model and features
@st.cache_resource
def load_models_and_features():
    models = {p.stem: joblib.load(p) for p in MODELS_DIR.glob("*.pkl") if p.stem != "feature_names"}
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return models, feature_names

models_dict, feature_names = load_models_and_features()
model_name = st.sidebar.selectbox("Choose a model", list(models_dict.keys()))
model = models_dict[model_name]

# Inputs
selected_features = {
    "Transaction_Amount": st.sidebar.number_input("Transaction Amount", value=12.50),
    "Account_Balance": st.sidebar.number_input("Account Balance", value=150.0),
    "Previous_Fraudulent_Activity": int(st.sidebar.checkbox("Previous Fraudulent Activity")),
    "Daily_Transaction_Count": st.sidebar.number_input("Daily Transaction Count", value=1),
    "Risk_Score": st.sidebar.slider("Risk Score", 0.0, 1.0, 0.92),
    "Is_Weekend": int(st.sidebar.checkbox("Is Weekend", value=True))
}
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# Build input row
X_user = pd.DataFrame([0.0] * len(feature_names), index=feature_names).T
for k, v in selected_features.items():
    X_user.at[0, k] = float(v)

# Prediction
proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"
st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# Adaptation block
with st.expander("Model Adaptation Example"):
    st.markdown("Retraining on 50 synthetic fraud cases based on your input.")
    df_fraud = pd.concat([X_user] * 50, ignore_index=True)
    df_fraud["Fraud_Label"] = 1
    X_new = df_fraud[feature_names]
    y_new = df_fraud["Fraud_Label"]
    adapted_model = LogisticRegression()
    adapted_model.fit(X_new, y_new)
    new_proba = adapted_model.predict_proba(X_user)[0, 1]
    new_label = "Fraud" if new_proba >= threshold else "Legit"
    st.write(f"**Adapted Model Prediction:** {new_label} ({new_proba:.2%})")



