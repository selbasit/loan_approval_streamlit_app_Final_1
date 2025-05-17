
import streamlit as st
import pickle
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load model and transformer
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("column_transformer.pkl", "rb") as f:
    column_transformer = pickle.load(f)

st.title("Loan Approval Prediction App")

# Display evaluation results
st.header("📊 Model Evaluation Metrics")
if 'metrics.csv' in os.listdir():
    metrics_df = pd.read_csv('metrics.csv')
    st.dataframe(metrics_df)
else:
    st.warning("No metrics file found.")

# Input form
st.header("📝 Predict a New Application")
user_input = {}
features = column_transformer.feature_names_in_

for feature in features:
    user_input[feature] = st.text_input(f"Enter {feature}", "")

if st.button("Predict Loan Approval"):
    try:
        input_df = pd.DataFrame([user_input])
        input_transformed = column_transformer.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        st.success(f"✅ Prediction: {'Approved' if prediction == 1 else 'Not Approved'}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
