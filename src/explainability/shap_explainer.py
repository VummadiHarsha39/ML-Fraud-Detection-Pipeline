import shap
import joblib

MODEL_PATH = "models/xgb_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def interpret_feature(feature, value):
    """
    Convert SHAP feature → human readable explanation
    """

    if feature in ["error_balance_orig", "abs_error_orig"]:
        return "High inconsistency in sender balance"

    elif feature in ["error_balance_dest", "abs_error_dest"]:
        return "Unusual receiver balance behavior"

    elif feature == "balance_diff_orig":
        return "Abnormal change in sender balance"

    elif feature == "balance_diff_dest":
        return "Abnormal change in receiver balance"

    elif feature == "amount_to_balance":
        return "Transaction amount unusually large relative to balance"

    elif feature == "is_large_txn":
        return "Large transaction detected"

    elif feature == "orig_zero":
        return "Sender account has zero balance"

    elif feature == "dest_zero":
        return "Receiver account has zero balance"

    elif feature == "log_amount":
        return "Transaction amount is unusually high"

    else:
        return f"{feature} contributed to the prediction"


def get_shap_explanation(input_df):
    model = load_model()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    explanations = []

    for i, col in enumerate(input_df.columns):
        val = shap_values[0][i]

        # only keep important features
        if abs(val) > 0.5:
            explanations.append({
                "feature": col,
                "impact": float(val),
                "reason": interpret_feature(col, val)
            })

    # sort by importance
    explanations = sorted(explanations, key=lambda x: abs(x["impact"]), reverse=True)

    return explanations[:5]
