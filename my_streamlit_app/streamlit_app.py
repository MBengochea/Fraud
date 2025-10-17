with st.expander("Model Adaptation Example"):
    st.markdown("""
    This simulation retrains the model on 50 synthetic fraud cases and 50 legit cases based on your input.  
    It shows how the model can adapt to new patterns.
    """)

    # Create 50 synthetic fraud and 50 legit examples
    df_fraud = pd.concat([X_user] * 50, ignore_index=True)
    df_fraud["Fraud_Label"] = 1

    df_legit = pd.concat([X_user] * 50, ignore_index=True)
    df_legit["Fraud_Label"] = 0

    df_adapt = pd.concat([df_fraud, df_legit], ignore_index=True)
    X_new = df_adapt[feature_names]
    y_new = df_adapt["Fraud_Label"]

    # Retrain model
    adapted_model = LogisticRegression()
    adapted_model.fit(X_new, y_new)

    # Predict again
    new_proba = adapted_model.predict_proba(X_user)[0, 1]
    new_label = "Fraud" if new_proba >= threshold else "Legit"

    st.write(f"**Adapted Model Prediction:** {new_label} ({new_proba:.2%})")
    st.caption("Retrained on 100 synthetic examples (50 fraud, 50 legit). Real deployments use thousands.")
