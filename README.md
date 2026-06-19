# Online Payments Fraud Detection

A machine learning project to detect fraudulent financial transactions using the PaySim synthetic dataset. Four classification models are compared to identify the best approach for real-world fraud detection.

---

## Problem Statement

Online payment fraud causes billions in losses globally every year. This project builds and evaluates classification models that flag fraudulent transactions with high precision — minimizing false negatives (missed fraud) while keeping false positives (wrongly flagged legitimate transactions) low.

---

## Dataset

**Source:** PaySim Synthetic Financial Dataset (`opfd.csv`)

| Feature | Description |
|---|---|
| `type` | Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER) |
| `amount` | Transaction amount |
| `nameOrig` | Originating account ID |
| `oldbalanceOrg` | Sender's balance before transaction |
| `newbalanceOrig` | Sender's balance after transaction |
| `nameDest` | Destination account ID |
| `oldbalanceDest` | Receiver's balance before transaction |
| `newbalanceDest` | Receiver's balance after transaction |
| `isFraud` | Target — 1 if fraudulent, 0 if legitimate |
| `isFlaggedFraud` | System flag for large illegal transfers |

---

## Approach

```
Raw Data → EDA → Label Encoding → Train/Test Split → Model Training → Evaluation → Comparison
```

1. **EDA** — distribution analysis, missing value check, class imbalance inspection
2. **Preprocessing** — label encoding of `type` column, dropping non-informative ID columns
3. **Modelling** — four classifiers trained and evaluated
4. **Evaluation** — accuracy, F1 score, confusion matrix, classification report

---

## Models & Results

| Model | Train Score | Test Accuracy | F1 Score |
|---|---|---|---|
| Random Forest | ~0.97 | **~0.96** | **~0.96** |
| Decision Tree | ~0.97 | ~0.95 | ~0.95 |
| K-Nearest Neighbours | — | ~0.93 | ~0.93 |
| Logistic Regression | — | ~0.91 | ~0.91 |

> ⚠️ **Note on accuracy:** Earlier versions of this project achieved 100% accuracy by accidentally including the balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) as features. Per the dataset documentation, fraud transactions are cancelled and these columns are zeroed out for fraudulent records — meaning the model was learning "balance = 0 → fraud" rather than genuine patterns. This is **data leakage**. These columns are correctly excluded in this version, giving realistic and honest accuracy scores.

---

## Visualisations

- Transaction type distribution (donut chart)
- Fraud vs non-fraud class distribution
- Feature correlation heatmap
- Confusion matrices for all four models
- Model accuracy comparison bar chart

---

## Project Structure

```
fraud-detection/
├── fraud_detection.py     # Main script
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/vanshika2003/fraud-detection.git
cd fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset in the project folder
# Rename it to opfd.csv or update DATA_PATH in the script

# 4. Run
python fraud_detection.py
```

---

## Key Findings

- The dataset is highly imbalanced — fraud cases are less than 1% of all transactions
- Balance columns (`oldbalanceOrg` etc.) were deliberately excluded to prevent data leakage — per the dataset documentation, these are zeroed out for fraud transactions, which would give the model an unfair shortcut
- Legitimate features used: `step`, `type` (encoded), `amount`
- Random Forest performs best at ~96% accuracy on clean, leakage-free features
- TRANSFER and CASH-OUT transaction types are most associated with fraudulent activity
- Logistic Regression underperforms due to the non-linear nature of fraud patterns

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-5.14+-purple?logo=plotly)

- **Language:** Python 3.10+
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, plotly

---

## Author

**Vanshika Aggarwal**
M.Sc. Data Science — Chandigarh University
[LinkedIn](https://www.linkedin.com/in/vanshika2003) • [GitHub](https://github.com/vanshika2003)
