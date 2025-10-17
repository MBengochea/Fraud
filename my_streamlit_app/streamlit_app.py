import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
)
import matplotlib.pyplot as plt
from PIL import Image

# Page config
st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

# Project directories
BASE_DIR   = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

# Sidebar — logo
with st.sidebar:
    logo_path = ASSETS_DIR / "banner.png"
    if not logo_path.exists():
        st.error("Missing `assets/banner.png` – add it to GitHub and redeploy.")
        st.stop()
    st.image(Image.open(logo_path), width=250)

# Load models and feature names
@st.cache_resource
def load_models_and_features():
    models = {}
    for p in MODELS_DIR.glob("*.pkl"):
        if p.stem == "feature_names":
            continue
        models[p.stem] = joblib.load(p)
    if not models:
        st.error("No model .pkl files found in `models/`. Add them to GitHub and redeploy.")
        st.stop()

    fn = MODELS_DIR / "feature_names.pkl"
    if not fn.exists():
        st.error("Missing `models/feature_names.pkl`. Add it to GitHub and redeploy.")
        st.stop()
    feature_names = joblib.load(fn)

    return models, feature_names

models_dict, feature_names = load_models_and_features()

# Sidebar — model selector
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models_dict.keys()))
model = models_dict[model_name]

# Sidebar — input features
st.sidebar.header("Input Features")
selected_features = {
    "Transaction_Amount": st.sidebar.number_input("Transaction Amount", value=0.0),
    "Account_Balance": st.sidebar.number_input("Account Balance", value=0.0),
    "Previous_Fraudulent_Activity": int(st.sidebar.checkbox("Previous Fraudulent Activity", value=False)),
    "Daily_Transaction_Count": st.sidebar.number_input("Daily Transaction Count", value=0),
    "Risk_Score": st.sidebar.slider("Risk Score", 0.0, 1.0, 0.0),
    "Is_Weekend": int(st.sidebar.checkbox("Is Weekend", value=False))
}

# Sidebar — decision threshold
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# Build user input DataFrame with full feature schema
X_user_full = pd.DataFrame(columns=feature_names)
X_user_full.loc[0] = 0.0
for feat, val in selected_features.items():
    X_user_full.at[0, feat] = float(val)
X_user = X_user_full.copy()

# Display user inputs
st.subheader("User Inputs")
st.dataframe(X_user[list(selected_features.keys())])

# Explain Risk_Score
with st.expander("What is Risk_Score?"):
    st.markdown("""
    **Risk_Score** is a precomputed feature from upstream systems.  
    It reflects how suspicious a transaction looks based on:
    - Velocity checks
    - Device/IP anomalies
    - Rule-based flags
    - Historical behavior
    """)

# Original prediction
proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"

st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# Adaptation block — retrain on synthetic data
with st.expander("Model Adaptation Example"):
    st.markdown("Retraining on 100 synthetic examples (50 fraud, 50 legit) based on your input.")

    # 50 synthetic fraud cases
    df_fraud = pd.concat([X_user] * 50, ignore_index=True)
    df_fraud["Fraud_Label"] = 1

    # 50 synthetic legit cases with lower risk
    df_legit = pd.concat([X_user] * 50, ignore_index=True)
    df_legit["Fraud_Label"] = 0
    if "Risk_Score" in df_legit.columns:
        df_legit["Risk_Score"] = df_legit["Risk_Score"] * 0.5

    # Combine into adaptation set
    df_adapt = pd.concat([df_fraud, df_legit], ignore_index=True)
    X_new = df_adapt[feature_names]
    y_new = df_adapt["Fraud_Label"]

    # Retrain a fresh model
    adapted_model = LogisticRegression()
    adapted_model.fit(X_new, y_new)

    # Predict again
    new_proba = adapted_model.predict_proba(X_user)[0, 1]
    new_label = "Fraud" if new_proba >= threshold else "Legit"

    st.write(f"**Adapted Model Prediction:** {new_label} ({new_proba:.2%})")
    st.write(f"Original: {proba:.2%} → Adapted: {new_proba:.2%}")
    st.write(f"Original Class: {'Fraud' if proba >= threshold else 'Legit'} → Adapted Class: {'Fraud' if new_proba >= threshold else 'Legit'}")
    st.caption("Retrained on 100 synthetic examples. Real deployments use thousands.")

# Load test set and evaluate
@st.cache_data
def load_test_set():
    path = MODELS_DIR / "test_set.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    X_test = df[feature_names]
    y_test = df["Fraud_Label"]
    return df, X_test, y_test

df_test, X_test, y_test = load_test_set()
if df_test is not None:
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    st.subheader("Model Performance on Test Set")
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    st.table(pd.DataFrame(metrics, index=["Score"]).T)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(2, 2))
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Legit", "Fraud"])
    ax_cm.set_yticklabels(["Legit", "Fraud"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax_cm.text(j, i, str(val), ha="center", va="center", fontsize=6)
    st.pyplot(fig_cm)

    fpr, tpr, _ = roc_curve(y_test, probs)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

    with col2:
        st.subheader("Precision–Recall Curve")
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall_vals, precision_vals)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        st.pyplot(fig_pr)

    st.subheader("Real Fraud Examples")
    cols_to_show = list(selected_features.keys()) + ["Fraud_Label"]
    fraud_cases = df_test[df_test["Fraud_Label"] == 1].head(5)
    st.dataframe(fraud_cases[cols_to_show])

    if {"LogisticOversampled", "LogisticRegression"}.issubset(models_dict):
        st.subheader("Model Comparison")
        o = models_dict["LogisticOversampled"].predict_proba(X_user)[0, 1]
        r = models_dict["LogisticRegression"].predict_proba(X_user)[0, 1]
        st.write(f"**Oversampled Model:** {o:.2%}")
        st.write(f"**Original Model:**  {r:.2%}")
        if o >= threshold > r:
            st.warning("Oversampled flags fraud; original does not.")
        elif r >= threshold > o:
            st.warning("Original flags fraud; oversampled does not.")
        else:
            st.info("Both models agree.")

