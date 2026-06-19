# Online Payments Fraud Detection

A machine learning project to detect fraudulent financial transactions using the PaySim synthetic dataset. Four classification models are compared to identify the best approach for real-world fraud detection.

---

## Problem Statement

Online payment fraud causes billions in losses globally every year. This project builds and evaluates classification models that flag fraudulent transactions with high precision — minimizing false negatives (missed fraud) while keeping false positives (wrongly flagged legitimate transactions) low.

---

## Dataset

**Source:** PaySim Synthetic Financial Dataset (`opfd.csv`) or download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

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
| Random Forest | 0.9999 | 0.9987 | 0.9986 |
| Decision Tree | 1.0000 | 0.9987 | 0.9982 |
| K-Nearest Neighbours | 0.9988  | 0.8918 | 0.9415 |
| Logistic Regression | 0.8919 |  0.9987 | 0.9987 |

> ⚠️ **Note on accuracy:** Earlier versions of this project achieved 100% accuracy by accidentally including the balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) as features. But as per the dataset documentation available on kaggle , fraud transactions are cancelled and these columns are zeroed out for fraudulent records — meaning the model was learning "balance = 0 → fraud" rather than genuine patterns. This is **data leakage**. These columns are correctly excluded in this version, giving realistic and honest accuracy scores.

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
Online-Payments-Fraud-Detection-/
├── Online_Payments_Fraud_Detection_Dataset.py     # Main script
├── requirements.txt       # Python dependencies
├── .gitignore
├── .gitattributes
├── opfd.csv              #dataset 
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/vanshika-data-lab/Online-Payments-Fraud-Detection-.git
cd Online-Payments-Fraud-Detection-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset in the project folder
# Rename it to opfd.csv or update DATA_PATH in the script

# 4. Run
python Online_Payments_Fraud_Detection_Dataset.ipynb
```

---

## Key Findings

- The dataset is highly imbalanced — fraud cases are less than 1% of all transactions
- Balance columns (`oldbalanceOrg` etc.) were deliberately excluded to prevent data leakage — per the dataset documentation, these are zeroed out for fraud transactions, which would give the model an unfair shortcut
- Legitimate features used: `step`, `type` (encoded), `amount`
- Random Forest, KNN, Decision Tree performs best at ~99.87% accuracy on clean, leakage-free features
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


[LinkedIn](https://www.linkedin.com/in/vanshika2003)
