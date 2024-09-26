# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load('credit_fraud_rf_model.pkl')  # Replace with the correct path

# Streamlit app layout
st.title("Credit Card Fraud Detection")

st.write("Input the details of the transaction below:")

# User input fields for the transaction details
amount = st.number_input('Amount', min_value=0.0, step=0.01)
time = st.number_input('Time', min_value=0.0, step=0.01)
v1 = st.number_input('V1 (PCA-reduced feature)')
# Add inputs for other PCA-reduced features V2 to V28
# Example for V2:
v2 = st.number_input('V2 (PCA-reduced feature)')
v3 = st.number_input('V3 (PCA-reduced feature)')
v4 = st.number_input('V4 (PCA-reduced feature)')
v5 = st.number_input('V5 (PCA-reduced feature)')
v6 = st.number_input('V6 (PCA-reduced feature)')
v7 = st.number_input('V7 (PCA-reduced feature)')
v8 = st.number_input('V8 (PCA-reduced feature)')
v9 = st.number_input('V9 (PCA-reduced feature)')
v10 = st.number_input('V10 (PCA-reduced feature)')
v11 = st.number_input('V11 (PCA-reduced feature)')
v12 = st.number_input('V12 (PCA-reduced feature)')
v13 = st.number_input('V13 (PCA-reduced feature)')
v14 = st.number_input('V14 (PCA-reduced feature)')
v15 = st.number_input('V15 (PCA-reduced feature)')
v16 = st.number_input('V16 (PCA-reduced feature)')
v17 = st.number_input('V17 (PCA-reduced feature)')
v18 = st.number_input('V18 (PCA-reduced feature)')
v19 = st.number_input('V19 (PCA-reduced feature)')
v20 = st.number_input('V20 (PCA-reduced feature)')
v21 = st.number_input('V21 (PCA-reduced feature)')
v22 = st.number_input('V22 (PCA-reduced feature)')
v23 = st.number_input('V23 (PCA-reduced feature)')
v24 = st.number_input('V24 (PCA-reduced feature)')
v25 = st.number_input('V25 (PCA-reduced feature)')
v26 = st.number_input('V26 (PCA-reduced feature)')
v27 = st.number_input('V27 (PCA-reduced feature)')
v28 = st.number_input('V28 (PCA-reduced feature)')
# Add more fields as required

# Predict button
if st.button('Predict'):
    # Log-transform the 'Amount'
    amount_log = np.log1p(amount)

    # Prepare the feature array for prediction
    features = np.array([time, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11, v12,v13,v14,v15,v16,v17,v18,v19,v20,v21, v22,v23,v24,v25,v26,v27,v28,amount_log])  # Add V3, V4, ..., V28

    # Reshape the features to match the model input shape
    features = features.reshape(1, -1)
    df = pd.DataFrame(data=features[1:,1:])
    # Make a prediction
    prediction = model.predict(features)

    # Display result
    result = 'Fraudulent Transaction' if prediction[0] == 1 else 'Legitimate Transaction'
    st.write(f'Prediction: {result}')
