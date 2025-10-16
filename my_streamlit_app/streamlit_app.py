import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

# -- DEBUG: list model files
models_dir = Path("models")
found = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
st.sidebar.markdown(f"**Model files found:** {', '.join(p.name for p in found) or 'None'}")

# -- Load or fallback model
@st.cache_resource
def load_models():
    md = {}
    for p in found:
        try:
            md[p.stem] = joblib.load(p)
        except Exception as e:
            st.sidebar.warning(f"Failed to load {p.name}: {e}")
    if not md:
        lr = LogisticRegression()
        lr.fit([[0]*6, [1]*6], [0, 1])
        md["Fallback_Model"] = lr
    return md

models_dict = load_models()
model_name = st.sidebar.selectbox("Choose a model", list(models_dict.keys()))
model = models_dict[model_name]

# -- Inputs
features = [
    "Transaction_Amount", "Account_Balance", "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count", "Risk_Score", "Is_Weekend"
]
st.sidebar.header("Input Features")
inputs = {
    f: float(st.sidebar.number_input(f.replace("_", " "), value=0.0))
    if f not in ["Previous_Fraudulent_Activity", "Is_Weekend"]
    else int(st.sidebar.checkbox(f.replace("_", " "), value=False))
    for f in features
}
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# -- Banner
with st.sidebar:
    logo = Path("assets/banner.png")
    if logo.exists(): st.image(Image.open(logo), width=250)
    else: st.sidebar.markdown("### Fraud Model Explainer")

# -- Build DataFrame
X_user = pd.DataFrame([inputs], columns=features).fillna(0.0)
st.subheader("User Inputs")
st.dataframe(X_user)

# -- Explain
with st.expander("What is Risk_Score?"):
    st.write("Precomputed feature from velocity checks, anomalies, rule‐flags, behavior.")

# -- Predict
proba = model.predict_proba(X_user)[0,1]
label = "Fraud" if proba >= threshold else "Legit"
st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# -- Evaluation (if test_set exists)
@st.cache_data
def load_data():
    fp = Path("models/test_set.csv")
    return pd.read_csv(fp) if fp.exists() else None

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
