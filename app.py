import streamlit as st
import pickle
import numpy as np

# Load model
lor = pickle.load(open("model.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# 🔢 Numeric Inputs
age = st.number_input("Age", 18, 100)
tenure = st.slider("Tenure (Months)", 1, 72)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

# 🔤 Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox(
    "Payment Method",
    ["Bank transfer", "Credit card", "Electronic check", "Mailed check"]
)

# 🔄 Encoding (EXACT SAME AS TRAINING)

# Gender encoding
gender_map = {"Male": 1, "Female": 0, "Other": -1}
gender = gender_map[gender]

# Contract encoding
contract_map = {"Month-to-month": 1, "One year": 2, "Two year": 3}
contract = contract_map[contract]

# Payment dummy encoding
pay_credit = 1 if payment == "Credit card" else 0
pay_elec = 1 if payment == "Electronic check" else 0
pay_mail = 1 if payment == "Mailed check" else 0
# "Bank transfer" → all 0 (base category)

# ⚠️ ORDER MUST MATCH TRAINING DATA
input_data = np.array([[
    age,
    gender,
    tenure,
    monthly,
    contract,
    total,
    pay_credit,
    pay_elec,
    pay_mail
]])

# 🎯 Prediction
if st.button("Predict"):
    prediction = lor.predict(input_data)
    prob = lor.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"Customer will Churn ❌ (Probability: {prob:.2f})")
    else:
        st.success(f"Customer will Stay ✅ (Probability: {prob:.2f})")