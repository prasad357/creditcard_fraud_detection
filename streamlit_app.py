# streamlit_app.py
import streamlit as st
import numpy as np
import joblib

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
# Add more fields as required

# Predict button
if st.button('Predict'):
    # Log-transform the 'Amount'
    amount_log = np.log1p(amount)

    # Prepare the feature array for prediction
    features = np.array([amount_log,time, v1, v2,v3,v4,v5,v6,v7,v8,v9,10])  # Add V3, V4, ..., V28

    # Reshape the features to match the model input shape
    features = features.reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)

    # Display result
    result = 'Fraudulent Transaction' if prediction[0] == 1 else 'Legitimate Transaction'
    st.write(f'Prediction: {result}')
