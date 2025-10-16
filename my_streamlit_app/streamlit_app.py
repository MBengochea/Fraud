import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# Page config
st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

# Resolve directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Debug: list .pkl files in /models
model_files = list(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []
st.sidebar.markdown(f"**Found model files:** {', '.join(p.name for p in model_files) or 'None'}")
if not model_files:
    st.error("No .pkl files found in /models. Add your model files and redeploy.")
    st.stop()

# Load models (skip feature_names.pkl)
models = {}
for f in model_files:
    if f.stem == "feature_names":
        continue
    try:
        models[f.stem] = joblib.load(f)
    except Exception as e:
        st.sidebar.warning(f"Failed to load {f.name}: {e}")
if not models:
    st.error("No valid models loaded. Check your .pkl files.")
    st.stop()

# Load feature names
fn_file = MODELS_DIR / "feature_names.pkl"
if fn_file.exists():
    feature_names = joblib.load(fn_file)
else:
    st.error("Missing feature_names.pkl in /models")
    st.stop()

# Sidebar logo
with st.sidebar:
    logo_file = ASSETS_DIR / "banner.png"
    if logo_file.exists():
        st.image(Image.open(logo_file), width=250)
    else:
        st.markdown("### Fraud Model Explainer")

# Model selector
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# Input features
st.sidebar.header("Input Features")
inputs = {}
for feat in feature_names:
    label = feat.replace("_", " ")
    if feat in ["Previous_Fraudulent_Activity", "Is_Weekend"]:
        inputs[feat] = int(st.sidebar.checkbox(label, value=0))
    else:
        inputs[feat] = float(st.sidebar.number_input(label, value=0.0))
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# Build user DataFrame
X_user = pd.DataFrame([inputs], columns=feature_names).fillna(0.0)

# Display inputs
st.subheader("User Inputs")
st.dataframe(X_user)

# Prediction
proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"
st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# Model evaluation if test_set.csv exists
test_file = MODELS_DIR / "test_set.csv"
if test_file.exists():
    df = pd.read_csv(test_file)
    if "is_fraud" in df.columns:
        X_test = df[feature_names]
        y_test = df["is_fraud"]
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        st.write(f"Precision: {precision_score(y_test, y_pred):.2%}")
        st.write(f"Recall: {recall_score(y_test, y_pred):.2%}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred):.2%}")
