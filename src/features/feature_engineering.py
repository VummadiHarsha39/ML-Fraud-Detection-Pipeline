import numpy as np
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/clean_transactions.csv")
OUTPUT_PATH = Path("data/processed/featured_transactions.csv")


def load_data():
    print("📥 Loading cleaned data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Shape: {df.shape}")
    return df


def create_features(df):
    print("⚙️ Creating features...")

    # ================= BASIC FEATURES =================
    df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["log_amount"] = df["amount"].apply(lambda x: np.log1p(x))

    # ================= STRONG FRAUD FEATURES =================

    # Balance inconsistency (CRITICAL)
    df["error_balance_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Absolute error (very strong signal)
    df["abs_error_orig"] = df["error_balance_orig"].abs()
    df["abs_error_dest"] = df["error_balance_dest"].abs()

    # Zero balance flags
    df["orig_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["dest_zero"] = (df["oldbalanceDest"] == 0).astype(int)

    # Ratio feature
    df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # Large transaction flag
    threshold = df["amount"].quantile(0.95)
    df["is_large_txn"] = (df["amount"] > threshold).astype(int)

    # ================= ONE-HOT ENCODING =================
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    print("✅ Feature engineering completed")
    return df


def save_data(df):
    print("💾 Saving featured data...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")


def run_feature_engineering():
    df = load_data()
    df = create_features(df)
    save_data(df)


if __name__ == "__main__":
    run_feature_engineering()
