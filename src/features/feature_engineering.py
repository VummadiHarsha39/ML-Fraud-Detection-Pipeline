import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/processed/clean_transactions.csv")
OUTPUT_PATH = Path("data/processed/featured_transactions.csv")


def load_data(sample_frac=0.2):
    print("📥 Loading processed data...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    # Sample for faster processing (20%)
    df = df.sample(frac=sample_frac, random_state=42)

    print(f"Sampled shape: {df.shape}")
    return df


def add_time_features(df):
    print("⏱ Adding time features...")

    df['hour'] = df['transaction_time'].astype('datetime64[ns]').dt.hour
    df['is_weekend'] = df['transaction_time'].astype('datetime64[ns]').dt.weekday >= 5

    return df


def add_transaction_type_features(df):
    print("💳 Encoding transaction types...")

    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    return df


def add_balance_features(df):
    print("💰 Adding balance features...")

    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    return df


def add_amount_features(df):
    print("📊 Adding amount features...")

    df['log_amount'] = np.log1p(df['amount'])

    df['amount_zscore'] = (
        (df['amount'] - df['amount'].mean()) / df['amount'].std()
    )

    return df


def save_data(df):
    print("💾 Saving featured dataset...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to {OUTPUT_PATH}")


def run_feature_engineering():
    df = load_data()

    df['transaction_time'] = pd.to_datetime(df['transaction_time'])

    df = add_time_features(df)
    df = add_transaction_type_features(df)
    df = add_balance_features(df)
    df = add_amount_features(df)

    save_data(df)


if __name__ == "__main__":
    run_feature_engineering()
