import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# Page config
st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

# Resolve project structure
BASE_DIR     = Path(__file__).resolve().parent
MODELS_DIR   = BASE_DIR / "models"
ASSETS_DIR   = BASE_DIR / "assets"

# Debug: list model files found in GitHub repo
model_files = sorted(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []
st.sidebar.markdown(f"**Found .pkl files:** {', '.join(p.name for p in model_files) or 'None'}")
if not model_files:
    st.error("No model .pkl files found in `models/`. Confirm they are committed to GitHub.")
    st.stop()

# Load feature names
feature_names_path = MODELS_DIR / "feature_names.pkl"
if not feature_names_path.exists():
    st.error("Missing feature_names.pkl in `models/`. Add and redeploy.")
    st.stop()
feature_names = joblib.load(feature_names_path)

# Load models
models = {}
for p in model_files:
    if p.stem == "feature_names":
        continue
    try:
        models[p.stem] = joblib.load(p)
    except Exception as e:
        st.sidebar.warning(f"Failed to load {p.name}: {e}")
if not models:
    st.error("No valid models loaded. Check your .pkl files.")
    st.stop()

# Sidebar — logo
with st.sidebar:
    logo_path = ASSETS_DIR / "banner.png"
    if logo_path.exists():
        st.image(Image.open(logo_path), width=250)
    else:
        st.markdown("### Fraud Model Explainer")

# Sidebar — model selector
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# Sidebar — input features
st.sidebar.header("Input Features")
inputs = {}
for feat in feature_names:
    label = feat.replace("_", " ")
    if feat in ("Previous_Fraudulent_Activity", "Is_Weekend"):
        inputs[feat] = int(st.sidebar.checkbox(label))
    else:
        inputs[feat] = float(st.sidebar.number_input(label, value=0.0))

# Sidebar — decision threshold
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# Build user DataFrame
X_user = pd.DataFrame([inputs], columns=feature_names).fillna(0.0)

# Display user inputs
st.subheader("User Inputs")
st.dataframe(X_user)

# Explain Risk_Score
with st.expander("What is Risk_Score?"):
    st.markdown("""
    **Risk_Score** is a precomputed feature reflecting transaction risk based on:
    - Velocity checks  
    - Device/IP anomalies  
    - Rule-based flags  
    - Historical behavior  
    """)

# Run prediction
proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"

st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# Load and evaluate on test set
test_set_path = MODELS_DIR / "test_set.csv"
if test_set_path.exists():
    df = pd.read_csv(test_set_path)
    if "is_fraud" in df.columns:
        X_test  = df[feature_names]
        y_test  = df["is_fraud"]
        y_pred  = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        st.write(f"Precision: {precision_score(y_test, y_pred):.2%}")
        st.write(f"Recall: {recall_score(y_test, y_pred):.2%}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred):.2%}")
