import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

INPUT_PATH = Path("data/processed/featured_transactions.csv")
MODEL_PATH = Path("models/xgb_model.pkl")


def load_data():
    print("📥 Loading featured data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Shape: {df.shape}")
    return df


def prepare_data(df):
    print("🧹 Preparing data...")

    # Drop non-useful columns
    df = df.drop(columns=['user_id', 'receiver_id', 'transaction_time'], errors='ignore')

    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    return X, y


def split_data(X, y):
    print("✂️ Splitting data...")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    print("🚀 Training XGBoost...")

    # Handle class imbalance properly
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # 🔥 KEY FIX
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    print("📊 Evaluating model...")

    probs = model.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.5, 0.7]

    for threshold in thresholds:
        print(f"\n🔹 Threshold: {threshold}")

        preds = (probs > threshold).astype(int)

        print(classification_report(y_test, preds))

    print("📈 AUC:", roc_auc_score(y_test, probs))


def save_model(model):
    print("💾 Saving model...")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


def run_training():
    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    run_training()
