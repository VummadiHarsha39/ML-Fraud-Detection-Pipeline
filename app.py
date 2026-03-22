import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("💳 ML Fraud Detection System")
st.markdown("Real-time fraud detection with explainability")

# ================= INPUT FORM =================

st.subheader("Enter Transaction Details")

amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0)
newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0)
oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0)
newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0)

transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])

# ================= PREDICT BUTTON =================

if st.button("Predict Fraud"):

    payload = {
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "type_TRANSFER": 1 if transaction_type == "TRANSFER" else 0,
        "type_CASH_OUT": 1 if transaction_type == "CASH_OUT" else 0
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        st.subheader("Prediction Result")

        if result.get("is_fraud") == 1:
            st.error(f"🚨 FRAUD DETECTED (Probability: {result['fraud_probability']:.4f})")
        else:
            st.success(f"✅ NOT FRAUD (Probability: {result['fraud_probability']:.4f})")

        # ================= EXPLANATION =================
        if "explanation" in result:
            st.subheader("Why this prediction?")

            for item in result["explanation"]:
                st.write(f"🔹 {item['reason']} (impact: {item['impact']:.2f})")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")
