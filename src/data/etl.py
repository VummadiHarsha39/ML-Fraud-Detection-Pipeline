import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path

# Paths
RAW_DATA_PATH = Path("data/raw/transactions.csv")
PROCESSED_DATA_PATH = Path("data/processed/clean_transactions.csv")


def load_data(path):
    print("📥 Loading PaySim dataset...")
    df = pd.read_csv(path)

    # Rename columns for consistency
    df = df.rename(columns={
        "nameOrig": "user_id",
        "nameDest": "receiver_id",
        "isFraud": "is_fraud"
    })

    # Convert step (hours) → datetime
    df['transaction_time'] = pd.to_datetime(df['step'], unit='h')

    print(f"Shape: {df.shape}")
    return df


def check_missing_values(df):
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])


def impute_missing_values(df):
    print("\n🧠 Applying median imputation...")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


def validate_data(df):
    print("\n✅ Validating data...")

    if df.isnull().sum().sum() != 0:
        raise ValueError("❌ Still contains missing values!")

    print("✔ Data is clean")


def save_data(df, path):
    print("\n💾 Saving processed dataset...")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"Saved to: {path}")


def run_etl():
    df = load_data(RAW_DATA_PATH)

    check_missing_values(df)

    df = impute_missing_values(df)

    validate_data(df)

    save_data(df, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    run_etl()
