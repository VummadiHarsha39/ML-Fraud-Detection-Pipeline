# 💳 ML-Powered Fraud Detection Pipeline

An end-to-end machine learning pipeline designed to detect fraudulent financial transactions using behavioral patterns derived from transaction data.

This project focuses on building a structured workflow — from data processing to model inference — while emphasizing feature engineering and system design over purely model-driven performance.

---

## ✨ Overview

The objective of this project is to identify fraudulent transactions by learning **transactional inconsistencies and behavioral anomalies**, rather than relying on simple threshold-based rules.

The system integrates:

* Data preprocessing and validation
* Feature engineering based on financial behavior
* Model training using XGBoost
* Real-time inference through FastAPI
* Explainability using SHAP

---

## 🧠 Core Idea

Fraud detection is treated as a **behavioral problem**, not just a classification task.

Instead of focusing only on transaction amount, the model captures:

* Balance inconsistencies after transactions
* Abnormal account state transitions
* Transaction amount relative to available balance

This approach aligns more closely with real-world fraud detection systems.

---

## 📂 Project Structure

```
ML-Fraud-Detection-Pipeline/
│
├── data/                  # Raw and processed datasets
├── src/
│   ├── api/               # FastAPI service
│   ├── data/              # ETL pipeline
│   ├── features/          # Feature engineering
│   ├── models/            # Training & evaluation
│   ├── explainability/    # SHAP explanations
│   └── monitoring/        # Drift detection
│
├── models/                # Trained model artifacts
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline Breakdown

### 1. Data Processing

* Cleaned and validated transaction data
* Handled missing values
* Prepared structured dataset for modeling

---

### 2. Feature Engineering

The project emphasizes feature design to capture anomalies:

* `error_balance_orig` → mismatch in sender balance
* `abs_error_dest` → receiver inconsistency
* `amount_to_balance` → transaction intensity
* `is_large_txn` → threshold-based behavior

These features help expose patterns commonly associated with fraudulent activity.

---

### 3. Model Training

* Model: **XGBoost**
* Addressed class imbalance
* Evaluated using precision-recall trade-offs

The focus was on learning meaningful patterns rather than maximizing benchmark scores.

---

### 4. Explainability

Integrated SHAP to provide insight into model predictions.

Instead of returning only a classification, the system explains:

> which features contributed most to the decision

This is important for interpretability in financial systems.

---

### 5. API Layer

* Built using FastAPI
* Supports real-time transaction scoring
* Returns:

  * fraud probability
  * classification result
  * feature-level explanation

---

## 🧪 Sample Scenarios

### 🚨 Fraud-like Pattern

```json
{
  "amount": 90000,
  "oldbalanceOrg": 90000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 0,
  "type_CASH_OUT": 1
}
```

* High likelihood of fraud
* Signals: account draining and balance inconsistency

---

### ✅ Normal Transaction

```json
{
  "amount": 500,
  "oldbalanceOrg": 5000,
  "newbalanceOrig": 4500,
  "oldbalanceDest": 1000,
  "newbalanceDest": 1500,
  "type_TRANSFER": 1
}
```

* Low fraud probability
* Consistent balance updates

---

## 🛠️ Tech Stack

* Python
* XGBoost
* scikit-learn
* FastAPI
* SHAP
* Pandas / NumPy

---

## 🧠 Key Takeaways

* Fraud detection depends heavily on **behavioral feature engineering**
* Model performance is strongly influenced by how well anomalies are represented
* Explainability is essential for trust in financial systems
* Building an end-to-end pipeline improves understanding of real-world ML systems


---

## 🚀 Summary

A structured machine learning pipeline demonstrating how transaction data can be transformed into an explainable fraud detection system using feature-driven modeling.

