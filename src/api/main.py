from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from src.explainability.shap_explainer import get_shap_explanation

# Load trained model
model = joblib.load("models/xgb_model.pkl")

THRESHOLD = 0.5

app = FastAPI()


# Input schema
class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_TRANSFER: int = 0
    type_CASH_OUT: int = 0


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # ================= STEP 1: BASE INPUT =================
        data = pd.DataFrame([{
            "amount": transaction.amount,
            "oldbalanceOrg": transaction.oldbalanceOrg,
            "newbalanceOrig": transaction.newbalanceOrig,
            "oldbalanceDest": transaction.oldbalanceDest,
            "newbalanceDest": transaction.newbalanceDest,
            "type": "TRANSFER" if transaction.type_TRANSFER == 1 else "CASH_OUT"
        }])

        # ================= STEP 2: MATCH TRAINING ENCODING =================
        data = pd.get_dummies(data, columns=["type"], drop_first=True)

        # ================= STEP 3: FEATURE ENGINEERING =================

        # Basic
        data["balance_diff_orig"] = data["newbalanceOrig"] - data["oldbalanceOrg"]
        data["balance_diff_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]
        data["log_amount"] = np.log1p(data["amount"])

        # Strong fraud signals
        data["error_balance_orig"] = data["oldbalanceOrg"] - data["amount"] - data["newbalanceOrig"]
        data["error_balance_dest"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]

        data["abs_error_orig"] = data["error_balance_orig"].abs()
        data["abs_error_dest"] = data["error_balance_dest"].abs()

        data["orig_zero"] = (data["oldbalanceOrg"] == 0).astype(int)
        data["dest_zero"] = (data["oldbalanceDest"] == 0).astype(int)

        data["amount_to_balance"] = data["amount"] / (data["oldbalanceOrg"] + 1)

        # Match training behavior
        data["is_large_txn"] = (data["amount"] > 50000).astype(int)

        # ================= STEP 4: ALIGN FEATURES =================

        expected_features = list(model.get_booster().feature_names)

        for col in expected_features:
            if col not in data.columns:
                data[col] = 0

        data = data[expected_features]

        # ================= STEP 5: PREDICT =================

        prob = model.predict_proba(data)[0][1]
        prediction = int(prob > THRESHOLD)

        # ================= STEP 6: SHAP EXPLANATION =================

        explanation = get_shap_explanation(data)

        return {
            "fraud_probability": float(prob),
            "is_fraud": prediction,
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}
